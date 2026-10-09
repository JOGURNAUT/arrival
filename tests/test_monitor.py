"""Monitoring a model you cannot score yet.

The test that matters here is
`test_a_broken_model_is_invisible_until_its_labels_arrive`. It asserts that the
accuracy dashboard is wrong -- that it reports a healthy model while every
prediction being served is thirty-five minutes out -- because that is true, and
a monitoring layer written as though it were not will be trusted during the one
hour it cannot be.
"""

from __future__ import annotations

import datetime as dt
import random

import pytest

from arrival.contracts import ArrivalError, FeatureVector, Prediction, Trip
from arrival.monitor import (Monitor, feature_drift, missing_features, psi)


EIGHT_AM = dt.datetime(2026, 9, 1, 8, 0, 0)
BROKE_AT = dt.datetime(2026, 9, 1, 9, 0, 0)

# Every trip takes thirty-five minutes, so every label is thirty-five minutes
# late. Constant on purpose: it makes the blind window a number the test can
# assert on rather than a distribution it has to reason about.
TRIP_MINUTES = 35.0


# ------------------------------------------------------------------ fixture

def _timeline(n: int = 120) -> list[Trip]:
    """One trip assigned every minute from 08:00, each taking 35 minutes."""
    trips = []
    for i in range(n):
        assigned = EIGHT_AM + dt.timedelta(minutes=i)
        trips.append(Trip(
            trip_id=f"T-{i:04d}", store_id="NORTHGATE",
            driver_id=f"D-{i % 7:04d}", assigned_at=assigned,
            distance_m=7000, promised_minutes=19,
            delivered_at=assigned + dt.timedelta(minutes=TRIP_MINUTES),
            tat_minutes=TRIP_MINUTES))
    return trips


def _prediction(trip: Trip, i: int,
                broke_at: dt.datetime | None) -> Prediction:
    """A model that is half a minute out, and then catastrophically out."""
    if broke_at is not None and trip.assigned_at >= broke_at:
        minutes = 70.0                      # double every ETA
        version = "eta-2.1.0-bad"
    else:
        minutes = TRIP_MINUTES + (0.5 if i % 2 else -0.5)
        version = "eta-2.0.0"
    return Prediction(trip_id=trip.trip_id, minutes=minutes,
                      model_version=version, served_at=trip.assigned_at)


def _replay(as_of: dt.datetime, *, broke_at: dt.datetime | None = BROKE_AT,
            reference: list[float] | None = None) -> Monitor:
    """The monitor as it would stand at `as_of`.

    Predictions are logged only for trips already served. Outcome rows are
    handed over for all of them, including the ones whose trips have not landed
    yet, so that what is being tested is the gate inside `realised_accuracy`
    and not the test's own bookkeeping. In production the join against the trip
    table is just as happy to hand over a row early.
    """
    monitor = Monitor(reference_predictions=reference)
    for i, trip in enumerate(_timeline()):
        if trip.assigned_at > as_of:
            break
        monitor.record_prediction(_prediction(trip, i, broke_at), trip)
        monitor.record_outcome(trip)
    return monitor


def _reference_predictions(n: int = 500) -> list[float]:
    """What this model used to output, which is what a reference window is.

    Taken from the model's own earlier predictions rather than from an assumed
    shape. A reference built from a distribution somebody expected the model to
    have reports drift on the day it is installed, every time.
    """
    return [TRIP_MINUTES + (0.5 if i % 2 else -0.5) for i in range(n)]


# ------------------------------------------------------------- lagged labels

def test_a_broken_model_is_invisible_until_its_labels_arrive():
    """The reason this module exists.

    A model starts doubling every ETA at 09:00. Twenty minutes later the
    accuracy dashboard shows an MAE of half a minute over forty-six trips --
    not slightly off, healthy -- because every trip that has finished was
    predicted before the break. The forty-six trips that have landed are all
    from the good model; the thirty-five predictions that are wrong are still
    in flight.

    By 10:00 the labels have caught up and the MAE is 10.93. Nothing about the
    model changed between those two readings. If this test fails because
    realised accuracy moved earlier, something is scoring predictions whose
    labels do not exist, which means the number on the dashboard during an
    incident is partly invented."""
    at_0920 = _replay(dt.datetime(2026, 9, 1, 9, 20))
    fresh = at_0920.realised_accuracy(dt.datetime(2026, 9, 1, 9, 20))

    assert fresh is not None
    assert fresh.n == 46                    # all predicted before 09:00
    assert fresh.mae == pytest.approx(0.5)  # the healthy model, exactly
    assert fresh.p90_error == pytest.approx(0.5)

    # Meanwhile the damage is real and entirely unscored.
    assert at_0920.labels_pending(dt.datetime(2026, 9, 1, 9, 20)) == 35

    at_1000 = _replay(dt.datetime(2026, 9, 1, 10, 0))
    late = at_1000.realised_accuracy(dt.datetime(2026, 9, 1, 10, 0))

    assert late is not None
    assert late.n == 86
    assert late.mae == pytest.approx(10.93, abs=0.01)
    assert late.mae > 20 * fresh.mae


def test_realised_accuracy_states_how_far_behind_the_present_it_is():
    """An MAE with no lag beside it reads as current, and that misreading is
    the entire failure. Half a minute of error is a fine number to show
    somebody -- as long as it is labelled as describing the model of
    thirty-five minutes ago."""
    as_of = dt.datetime(2026, 9, 1, 9, 20)
    report = _replay(as_of).health_report(as_of)

    assert report.scored_through == dt.datetime(2026, 9, 1, 8, 45)
    assert report.blind_minutes == pytest.approx(35.0)
    assert report.label_lag_minutes == pytest.approx(35.0)
    assert report.labels_pending == 35
    assert report.realised is not None
    assert report.realised.mae == pytest.approx(0.5)


def test_a_label_that_has_not_landed_is_not_scored_early():
    """Filling the gap -- scoring against the promise, carrying the last known
    error forward, assuming the pending trips are fine -- all amount to putting
    the hoped-for answer on the dashboard."""
    monitor = Monitor()
    trip = _timeline(1)[0]
    monitor.record_prediction(_prediction(trip, 0, None), trip)
    monitor.record_outcome(trip)

    one_minute_early = trip.delivered_at - dt.timedelta(minutes=1)
    assert monitor.realised_accuracy(one_minute_early) is None
    assert monitor.labels_pending(one_minute_early) == 1

    assert monitor.realised_accuracy(trip.delivered_at) is not None
    assert monitor.labels_pending(trip.delivered_at) == 0


def test_no_labels_yet_reports_nothing_rather_than_a_perfect_model():
    """An MAE of 0.0 over zero trips renders on a chart as the best model ever
    deployed, and it renders that way for the first half hour after every
    release."""
    monitor = Monitor()
    assert monitor.realised_accuracy(EIGHT_AM) is None

    report = monitor.health_report(EIGHT_AM)
    assert report.realised is None
    assert report.blind_minutes == float("inf")
    assert "realised_mae" not in report.as_row()


def test_a_slow_warehouse_lengthens_the_blind_window():
    """The label is knowable when the trip ends and available when the
    pipeline has landed it. Scoring from the end of the trip flatters the
    dashboard by however long the pipeline takes, which is exactly the part
    nobody measures."""
    trips = _timeline(1)
    prompt, slow = Monitor(), Monitor(label_delay_minutes=25.0)
    for monitor in (prompt, slow):
        monitor.record_prediction(_prediction(trips[0], 0, None), trips[0])
        monitor.record_outcome(trips[0])

    just_landed = trips[0].delivered_at
    assert prompt.realised_accuracy(just_landed) is not None
    assert slow.realised_accuracy(just_landed) is None
    assert slow.realised_accuracy(
        just_landed + dt.timedelta(minutes=25)) is not None


def test_an_outcome_for_a_trip_nobody_predicted_is_ignored():
    """The prediction log and the trip table are joined across two systems and
    will not agree perfectly. A monitor that dies on the mismatch is a monitor
    that is off during the incident."""
    monitor = Monitor()
    stranger = _timeline(1)[0]

    monitor.record_outcome(stranger)        # must not raise
    assert monitor.realised_accuracy(stranger.delivered_at) is None


def test_breach_rate_sits_beside_the_error_because_lateness_is_one_sided():
    """A model can hold a respectable MAE while breaching constantly by being
    symmetrically wrong, and the customer only ever notices one of those two
    directions."""
    as_of = dt.datetime(2026, 9, 1, 10, 0)
    optimistic = Monitor()
    for i, trip in enumerate(_timeline(40)):
        optimistic.record_prediction(
            Prediction(trip_id=trip.trip_id, minutes=TRIP_MINUTES - 4.0,
                       model_version="eta-2.0.0", served_at=trip.assigned_at),
            trip)
        optimistic.record_outcome(trip)

    realised = optimistic.realised_accuracy(as_of)
    assert realised is not None
    assert realised.mae == pytest.approx(4.0)
    assert realised.breach_rate == pytest.approx(1.0)


# -------------------------------------------------------- immediate signals

def test_prediction_drift_catches_the_break_while_it_is_happening():
    """The complement of the lagged-label test, and the practical answer to
    it. The prediction distribution needs no labels at all, so a model that
    starts answering seventy minutes to everything is visible the moment it
    does -- thirty-five minutes before the accuracy chart moves."""
    as_of = dt.datetime(2026, 9, 1, 9, 20)
    monitor = _replay(as_of, reference=_reference_predictions())

    drift = monitor.prediction_drift(as_of, window_minutes=30)
    assert drift is not None
    assert drift > 1.0                      # far past the 0.25 band

    healthy = _replay(as_of, broke_at=None, reference=_reference_predictions())
    quiet = healthy.prediction_drift(as_of, window_minutes=30)
    assert quiet is not None and quiet < 0.25
    assert drift > 10 * quiet


def test_the_fallback_rate_is_visible_without_any_labels():
    """A store that goes cold makes the service serve the flat promise to
    everybody. Accuracy will not show it for over half an hour, and when it
    does it will show it as a modelling problem rather than an outage."""
    monitor = Monitor()
    for i, trip in enumerate(_timeline(20)):
        monitor.record_prediction(Prediction(
            trip_id=trip.trip_id, minutes=19.0, model_version="eta-2.0.0",
            served_at=trip.assigned_at, fallback=i < 5), trip)

    assert monitor.fallback_rate == pytest.approx(0.25)
    assert monitor.health_report(BROKE_AT).fallback_rate == pytest.approx(0.25)


# ------------------------------------------------------------------- drift

def test_psi_is_zero_for_identical_distributions():
    """A drift score that is non-zero on unchanged data makes every threshold
    arbitrary, and the first week of alerts teaches everybody to ignore it."""
    rng = random.Random(7)
    reference = [rng.gauss(35, 6) for _ in range(4000)]

    assert psi(reference, reference) == pytest.approx(0.0, abs=1e-12)

    redrawn = [rng.gauss(35, 6) for _ in range(4000)]
    assert psi(reference, redrawn) == pytest.approx(0.0074, abs=0.002)


def test_psi_is_large_for_a_shifted_distribution():
    """The actual job. A feature whose mean has moved by a standard deviation
    means the model is being asked about a population it was not fitted on, and
    nothing else in the stack will say so until the labels arrive."""
    rng = random.Random(7)
    reference = [rng.gauss(35, 6) for _ in range(4000)]

    nudged = [rng.gauss(36, 6) for _ in range(4000)]
    shifted = [rng.gauss(41, 6) for _ in range(4000)]
    widened = [rng.gauss(35, 12) for _ in range(4000)]

    assert psi(reference, nudged) == pytest.approx(0.038, abs=0.005)
    assert psi(reference, shifted) == pytest.approx(0.90, abs=0.05)
    assert psi(reference, widened) == pytest.approx(0.44, abs=0.05)

    # Ordered the way the bands claim: no shift, moderate, gone.
    assert (psi(reference, reference) < 0.1 < psi(reference, shifted))
    assert psi(reference, nudged) < psi(reference, widened) < psi(reference,
                                                                  shifted)


def test_a_feature_collapsing_to_one_value_is_not_reported_as_no_drift():
    """The failure this catches is a feature pipeline returning its default --
    zero, or the fleet mean -- for every entity. The reference then sits
    entirely on one side of its own lowest cut point, and a binning that sends
    a value above its own quantile reports a serene 0.0 for the most complete
    drift there is."""
    rng = random.Random(11)
    reference = [rng.gauss(35, 6) for _ in range(2000)]

    assert psi(reference, [19.0] * 500) > 10
    assert psi([5.0] * 100, [9.0] * 100) > 10
    assert psi([5.0] * 100, [5.0] * 100) == pytest.approx(0.0)

    with pytest.raises(ArrivalError):
        psi([35.0], [35.0])


def test_drift_is_reported_per_feature_rather_than_as_one_number():
    """One combined drift score tells you to look at twenty features. The
    per-feature score tells you which deploy to roll back."""
    rng = random.Random(5)
    reference = [FeatureVector(f"T-{i}", EIGHT_AM, {
        "distance_m": rng.gauss(7000, 1500),
        "driver_mean_minutes": rng.gauss(30, 5)}) for i in range(2000)]
    current = [FeatureVector(f"U-{i}", BROKE_AT, {
        "distance_m": rng.gauss(7000, 1500),
        "driver_mean_minutes": rng.gauss(38, 5)}) for i in range(2000)]

    scores = feature_drift(reference, current)
    assert scores["distance_m"] < 0.1
    assert scores["driver_mean_minutes"] > 0.25
    assert scores["driver_mean_minutes"] == pytest.approx(2.36, abs=0.1)
    assert scores["distance_m"] == pytest.approx(0.005, abs=0.004)


def test_a_feature_that_vanished_is_reported_as_missing_not_as_drift():
    """A renamed column has a different cause and a different fix from the
    world changing: the model keeps serving, the store keeps answering, and one
    input is quietly absent. Reporting it as drift sends somebody to look at
    traffic patterns instead of at the last deploy."""
    reference = [FeatureVector(f"T-{i}", EIGHT_AM,
                               {"distance_m": 7000.0 + i,
                                "driver_mean_minutes": 30.0 + i})
                 for i in range(50)]
    current = [FeatureVector(f"U-{i}", BROKE_AT, {"distance_m": 7000.0 + i})
               for i in range(50)]

    assert missing_features(reference, current) == ["driver_mean_minutes"]
    assert set(feature_drift(reference, current)) == {"distance_m"}


def test_drift_with_nothing_to_compare_against_refuses_to_answer():
    """A drift score computed from an empty window is 0.0, and 0.0 is the
    value a dashboard shows when everything is fine."""
    with pytest.raises(ArrivalError):
        psi([], [1.0, 2.0])
    with pytest.raises(ArrivalError):
        feature_drift([], [FeatureVector("T-1", EIGHT_AM, {"a": 1.0})])

    assert Monitor().prediction_drift(BROKE_AT) is None
    assert Monitor().feature_drift(BROKE_AT) == {}


def test_the_health_report_puts_the_lagged_and_immediate_signals_together():
    """Read apart, realised accuracy says healthy and prediction drift says
    broken, and whichever dashboard is open wins the argument. They belong in
    one object with the lag stated, so that "healthy as of thirty-five minutes
    ago, and something changed since" is the only available reading."""
    as_of = dt.datetime(2026, 9, 1, 9, 20)
    report = _replay(as_of, reference=_reference_predictions()
                     ).health_report(as_of, window_minutes=30)

    assert report.realised is not None
    assert report.realised.mae == pytest.approx(0.5)     # looks healthy
    assert report.prediction_psi > 1.0                   # is not
    assert report.blind_minutes == pytest.approx(35.0)   # and says so

    row = report.as_row()
    assert row["realised_n"] == 46
    assert row["labels_pending"] == 35
