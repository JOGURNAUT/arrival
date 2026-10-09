"""Watching a model whose labels arrive forty minutes after its predictions.

Most monitoring writing assumes you can compare predictions to truth. Here you
cannot, not yet. The true duration of a trip is known when the trip ends, which
is half an hour or more after the prediction was served, so an accuracy
dashboard is permanently behind the present by exactly the time it takes to
deliver an order.

That has a consequence worth stating plainly, because it is the reason this
module exists and not a detail of it:

    A model that broke at 09:00 looks perfectly healthy at 09:30.

Not "looks slightly off". Healthy -- because every trip that has been scored by
then was predicted before the break. Whoever is watching accuracy sees a flat
line while every prediction being served is wrong, and by the time the line
moves the damage is an hour old. Three things follow from it:

  * Realised accuracy must be computed only over predictions whose labels have
    genuinely arrived. Scoring the ones that have not yet landed, by treating a
    missing label as anything at all, invents a number.
  * It must be reported with how far behind it is. An MAE without
    `labels_pending` and `label_lag_minutes` beside it reads as current, and
    that misreading is the whole failure.
  * The signals that are available immediately -- feature drift, prediction
    drift, fallback rate -- are not nice-to-haves. They are the only things
    that can catch a break while it is happening, which is why PSI is in here
    next to the accuracy maths rather than in a separate dashboard nobody
    opens.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from .contracts import ArrivalError, FeatureVector, Metrics, Prediction, Trip
from .serve import percentile

# Proportions are floored before taking a log: an empty bin on either side
# makes the ratio zero or infinite, and one unlucky bin would otherwise report
# infinite drift on a distribution that barely moved.
_PSI_FLOOR = 1e-6


# ------------------------------------------------------------------- drift

def psi(reference: Sequence[float], current: Sequence[float],
        bins: int = 10) -> float:
    """Population stability index between a reference and a current sample.

        PSI = sum over bins of (current_share - reference_share)
                               * ln(current_share / reference_share)

    Bins are the reference's own quantiles, so the reference is uniform across
    them by construction and every departure from uniformity in the current
    sample is the signal.

    The conventional reading:

        < 0.10   nothing to look at
        0.10-0.25 moderate shift, worth an explanation
        > 0.25   the population has moved; the model was fitted on a different
                 one

    Those bands are convention and not law. They were set for credit scorecards
    on monthly windows, and on a short window with few observations PSI is
    noisy enough to cross 0.1 on its own -- hence `n` reported beside it
    everywhere this is used.

    The harder caveat: a feature that is legitimately seasonal *will* fire.
    Trip distance on a public holiday, orders per hour during a promotion,
    hour-of-day on a window that starts at a different clock time -- all of
    these produce large, correct PSI. Drift detection says something changed.
    It does not say anything is wrong, and a team that treats every alert as a
    defect learns within a fortnight to ignore the alerts.
    """
    if len(reference) < 2 or not current:
        # One reference observation has no quantiles to cut on, and binning it
        # into a single bin would report zero drift wherever the current window
        # had moved to.
        raise ArrivalError("PSI needs at least two reference observations and "
                           "one current one")
    if bins < 2:
        raise ArrivalError("PSI needs at least two bins")

    edges = _quantile_edges(reference, bins)
    ref_shares = _shares(reference, edges)
    cur_shares = _shares(current, edges)

    total = 0.0
    for r, c in zip(ref_shares, cur_shares):
        r, c = max(r, _PSI_FLOOR), max(c, _PSI_FLOOR)
        total += (c - r) * math.log(c / r)
    return total


def _quantile_edges(values: Sequence[float], bins: int) -> list[float]:
    """Interior cut points of the reference, de-duplicated.

    De-duplication matters for features that are mostly one value -- a
    count of trips in the last hour is zero for most drivers most of the time
    -- where several quantiles coincide and the repeated edges would create
    empty bins that exist only to inflate the score.
    """
    cuts = statistics.quantiles(sorted(values), n=bins, method="inclusive")
    edges: list[float] = []
    for c in cuts:
        if not edges or c > edges[-1]:
            edges.append(float(c))
    return edges


def _shares(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    """Share of the sample in each bin, cutting as (-inf, e0], (e0, e1], ...

    A value sitting exactly on a cut point belongs to the bin below it, which
    is `bisect_left`. Putting it above instead looks like a convention and is
    not: a feature that is one value for most entities -- trips in the last
    hour, almost always zero -- has that value as its own lower cut, so sending
    it upwards puts the entire reference population in one bin together with
    everything above it, and the score then reads zero no matter where the
    current window sits.
    """
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect.bisect_left(edges, v)] += 1
    return [c / len(values) for c in counts]


def feature_drift(reference: Sequence[FeatureVector],
                  current: Sequence[FeatureVector],
                  bins: int = 10) -> dict[str, float]:
    """PSI per feature, over features both windows actually contain.

    A feature present in one window and not the other is reported as a missing
    feature rather than as drift, because the two have different causes: drift
    is the world changing, a missing feature is the pipeline changing, and the
    second one is usually a deploy somebody can roll back.
    """
    if not reference or not current:
        raise ArrivalError("feature drift needs vectors in both windows")

    shared = sorted(set(reference[0].values) & set(current[0].values))
    scores: dict[str, float] = {}
    for name in shared:
        ref = [v.values[name] for v in reference if name in v.values]
        cur = [v.values[name] for v in current if name in v.values]
        if ref and cur:
            scores[name] = psi(ref, cur, bins)
    return scores


def missing_features(reference: Sequence[FeatureVector],
                     current: Sequence[FeatureVector]) -> list[str]:
    """Features the reference window had and the current one does not.

    Usually a renamed column. The model keeps serving, the store keeps
    answering, and one input is quietly absent or defaulted.
    """
    if not reference or not current:
        return []
    return sorted(set(reference[0].values) - set(current[0].values))


# ------------------------------------------------------------- the records

@dataclass
class _Served:
    """One prediction as it was served, and the label when it turns up."""

    trip_id: str
    minutes: float
    served_at: datetime
    model_version: str
    variant: str
    fallback: bool
    promised_minutes: float
    actual_minutes: float | None = None
    label_at: datetime | None = None

    def label_has_arrived(self, as_of: datetime) -> bool:
        return (self.actual_minutes is not None
                and self.label_at is not None
                and self.label_at <= as_of)


@dataclass(frozen=True)
class HealthReport:
    """Everything that can be said about the model at one instant.

    Deliberately mixed: the immediate signals and the lagged ones side by side,
    with the lag stated. Reading `realised` without reading `labels_pending`
    and `label_lag_minutes` is the mistake this dataclass is shaped to prevent,
    which is why they are fields of the same object and not two endpoints.
    """

    as_of: datetime
    n_served: int
    fallback_rate: float
    labels_pending: int
    label_lag_minutes: float
    scored_through: datetime | None
    realised: Metrics | None
    prediction_psi: float | None
    feature_psi: dict[str, float] = field(default_factory=dict)
    missing_features: tuple[str, ...] = ()

    @property
    def blind_minutes(self) -> float:
        """How stale the accuracy figure is, in minutes.

        The window in which a broken model is invisible. If this reads 38 then
        `realised` describes the model as it was 38 minutes ago, and nothing in
        `realised` can tell you what it is doing now.
        """
        if self.scored_through is None:
            return float("inf")
        return (self.as_of - self.scored_through).total_seconds() / 60.0

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "as_of": self.as_of.isoformat(), "n_served": self.n_served,
            "fallback_rate": self.fallback_rate,
            "labels_pending": self.labels_pending,
            "label_lag_minutes": self.label_lag_minutes,
            "blind_minutes": self.blind_minutes,
            "prediction_psi": self.prediction_psi,
        }
        if self.realised is not None:
            row.update({f"realised_{k}": v
                        for k, v in self.realised.as_row().items()})
        row.update({f"psi_{k}": v for k, v in self.feature_psi.items()})
        return row


# ------------------------------------------------------------- the monitor

class Monitor:
    """Prediction log in, honest health out.

    `label_delay_minutes` is added on top of the delivery time. The label is
    *knowable* when the trip ends, but it is *available* when the warehouse has
    landed it, and scoring from the end of the trip rather than from the
    warehouse write flatters the dashboard by however long the pipeline takes.
    """

    def __init__(self, *,
                 reference_predictions: Sequence[float] | None = None,
                 reference_vectors: Sequence[FeatureVector] | None = None,
                 label_delay_minutes: float = 0.0,
                 psi_bins: int = 10) -> None:
        self.reference_predictions = list(reference_predictions or [])
        self.reference_vectors = list(reference_vectors or [])
        self.label_delay = timedelta(minutes=label_delay_minutes)
        self.psi_bins = psi_bins
        self._served: dict[str, _Served] = {}
        self._vectors: list[FeatureVector] = []

    # -- ingest -------------------------------------------------------------

    def record_prediction(self, prediction: Prediction, trip: Trip, *,
                          vector: FeatureVector | None = None) -> None:
        """Log a prediction at the moment it is served.

        The promise is kept beside the prediction so that breach rate can be
        computed later without going back to the trip table, and the fallback
        flag is read off the prediction if it carries one -- a `Prediction`
        from the contract does not, so an unflagged prediction counts as
        modelled rather than guessed at.
        """
        served_at = prediction.served_at or trip.assigned_at
        self._served[prediction.trip_id] = _Served(
            trip_id=prediction.trip_id,
            minutes=float(prediction.minutes),
            served_at=served_at,
            model_version=prediction.model_version,
            variant=prediction.variant,
            fallback=bool(prediction.fallback),
            promised_minutes=float(trip.promised_minutes),
        )
        if vector is not None:
            self._vectors.append(vector)

    def record_outcome(self, trip: Trip) -> None:
        """Attach the truth to a prediction once the trip has finished.

        An outcome for a trip that was never predicted is ignored rather than
        raised on: in production the prediction log and the trip table are
        joined across systems and will not agree perfectly, and a monitor that
        dies on the mismatch is a monitor that is off during the incident.
        """
        record = self._served.get(trip.trip_id)
        if record is None or not trip.is_labelled:
            return
        actual = float(trip.tat_minutes)
        record.actual_minutes = actual
        ends_at = trip.delivered_at or (
            trip.assigned_at + timedelta(minutes=actual))
        record.label_at = ends_at + self.label_delay

    # -- the lagged view ----------------------------------------------------

    def realised_accuracy(self, as_of: datetime) -> Metrics | None:
        """Accuracy over the predictions whose labels exist by `as_of`.

        Nothing else is scored. A prediction whose trip is still in flight
        contributes no information, and the temptations -- score it against the
        promise, score it against the prediction, assume it was fine -- all
        amount to filling the gap with the answer you were hoping for.

        `None` when nothing has been labelled yet, rather than a zero. An MAE
        of 0.0 over no trips renders on a chart as a perfect model.
        """
        scored = [r for r in self._served.values()
                  if r.label_has_arrived(as_of)]
        if not scored:
            return None
        return _score(scored)

    def labels_pending(self, as_of: datetime) -> int:
        """Predictions served but not yet scoreable.

        Rises when traffic rises and when deliveries slow down, so it is read
        together with `label_lag_minutes` rather than alarmed on alone.
        """
        return sum(1 for r in self._served.values()
                   if not r.label_has_arrived(as_of))

    def label_lag_minutes(self, as_of: datetime) -> float:
        """Median minutes from serving a prediction to its label arriving.

        Median rather than mean: a handful of trips that sat in a queue for
        three hours would otherwise set the expectation for all of them.
        Measured from labels that have actually arrived, so it is an
        observation about this pipeline and not the 30-40 minutes the design
        assumed.
        """
        lags = [(r.label_at - r.served_at).total_seconds() / 60.0
                for r in self._served.values() if r.label_has_arrived(as_of)]
        return statistics.median(lags) if lags else float("nan")

    def scored_through(self, as_of: datetime) -> datetime | None:
        """The most recent prediction that has a label.

        The edge of what is known. Everything served after this instant is
        unmeasured, which is the sentence the accuracy number needs attached to
        it.
        """
        scored = [r.served_at for r in self._served.values()
                  if r.label_has_arrived(as_of)]
        return max(scored) if scored else None

    # -- the immediate view -------------------------------------------------

    def prediction_drift(self, as_of: datetime,
                         window_minutes: float = 60.0) -> float | None:
        """PSI of the predictions themselves against the reference window.

        Available the instant a prediction is served, with no label involved,
        which makes it the earliest warning available. It cannot tell you the
        model got worse -- a model can shift its output distribution and be
        more right -- but a model that starts answering 19 minutes to
        everything shows up here immediately and in realised accuracy forty
        minutes later.
        """
        if not self.reference_predictions:
            return None
        current = [r.minutes for r in self._served.values()
                   if _within(r.served_at, as_of, window_minutes)]
        if len(current) < 2:
            return None
        return psi(self.reference_predictions, current, self.psi_bins)

    def feature_drift(self, as_of: datetime,
                      window_minutes: float = 60.0) -> dict[str, float]:
        """PSI per feature over the vectors recorded in the window."""
        if not self.reference_vectors:
            return {}
        current = [v for v in self._vectors
                   if _within(v.as_of, as_of, window_minutes)]
        if len(current) < 2:
            return {}
        return feature_drift(self.reference_vectors, current, self.psi_bins)

    @property
    def fallback_rate(self) -> float:
        if not self._served:
            return 0.0
        fallbacks = sum(r.fallback for r in self._served.values())
        return fallbacks / len(self._served)

    # -- everything at once -------------------------------------------------

    def health_report(self, as_of: datetime,
                      window_minutes: float = 60.0) -> HealthReport:
        current = [v for v in self._vectors
                   if _within(v.as_of, as_of, window_minutes)]
        return HealthReport(
            as_of=as_of,
            n_served=len(self._served),
            fallback_rate=self.fallback_rate,
            labels_pending=self.labels_pending(as_of),
            label_lag_minutes=self.label_lag_minutes(as_of),
            scored_through=self.scored_through(as_of),
            realised=self.realised_accuracy(as_of),
            prediction_psi=self.prediction_drift(as_of, window_minutes),
            feature_psi=self.feature_drift(as_of, window_minutes),
            missing_features=tuple(
                missing_features(self.reference_vectors, current)),
        )


def _within(moment: datetime, as_of: datetime, window_minutes: float) -> bool:
    return as_of - timedelta(minutes=window_minutes) < moment <= as_of


def _score(records: Sequence[_Served]) -> Metrics:
    """The shared `Metrics` shape, over labelled records.

    Computed here rather than imported from the training layer on purpose. The
    monitor has to keep working when the training package is mid-change, and a
    monitoring module that fails to import during a deploy is off at the one
    moment it is needed.

    `breach_rate` is the share of trips that took longer than the ETA given to
    the customer -- the promise the model itself made. A model can hold a
    respectable MAE while breaching constantly by being symmetrically wrong,
    and the customer only notices one of those two directions.
    """
    actual = [float(r.actual_minutes) for r in records]  # all labelled
    errors = [abs(r.minutes - a) for r, a in zip(records, actual)]
    breaches = sum(a > r.minutes for r, a in zip(records, actual))
    return Metrics(
        n=len(records),
        mae=statistics.fmean(errors),
        rmse=math.sqrt(statistics.fmean([e * e for e in errors])),
        p50_error=percentile(errors, 50),
        p90_error=percentile(errors, 90),
        breach_rate=breaches / len(records),
    )
