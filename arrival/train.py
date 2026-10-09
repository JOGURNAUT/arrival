"""Training, and the experiment the whole repository is built to run.

Four models are fitted here and scored on the same held-out trips:

  * the flat promise Dispatch makes today, which is the only number worth
    comparing against -- a model that cannot beat a constant is not a model,
    it is a deployment risk with a dashboard;
  * a ridge regression on point-in-time features;
  * the same regression on windows filtered by `assigned_at`, so trips still
    on the road are counted;
  * the same regression on windows with no edges at all, so a driver's whole
    record is counted, future included.

The last two are not mistakes left in by accident. They are the measurement. A
leaky model looks better offline and is no better in production, and the only
way to say how much better it *looks* -- how large a hole a careless aggregate
can hide in -- is to build it on purpose and print the scores next to each
other. There are two of them because the two leaks are not the same size: one
reads the trip's own duration and buys more apparent accuracy than the honest
model's entire improvement, and the other dilutes itself across a driver's
record and buys a few per cent, which is the kind of gap that gets waved
through. That table is the finding.

No numpy, no sklearn, no pandas. The normal equations for nine features are a
dozen lines of arithmetic, and writing them out means the model's weights can
be read, saved and defended without a dependency that has to be installed
before anyone can see the result.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Sequence

from . import features, registry, source
from .contracts import ArrivalError, FeatureVector, Metrics, Trip

# A linear model extrapolates, and far enough out of the training range it
# extrapolates through zero. A negative or near-zero ETA is not a prediction,
# it is an outage with a number attached, so predictions are floored here.
MINIMUM_MINUTES = 1.0

RIDGE_ALPHA = 1.0
DEFAULT_HOLDOUT = 0.2


# ----------------------------------------------------------------- the split

def chronological_split(trips: Sequence[Trip],
                        holdout: float = DEFAULT_HOLDOUT
                        ) -> tuple[list[Trip], list[Trip]]:
    """Earliest trips to train, latest trips to test.

    Never random. A random split puts Tuesday evening in the test set and
    Tuesday afternoon in the training set, so the model is scored on a day it
    has already seen the shape of -- the same bug the feature layer exists to
    prevent, committed one level up where no leakage assertion can see it. It
    also flatters the model in exactly the way that matters least: production
    is always asked about a time it has never seen.
    """
    if not 0.0 < holdout < 1.0:
        raise ArrivalError(f"holdout {holdout} is not a fraction")
    ordered = sorted(trips, key=lambda t: (t.assigned_at, t.trip_id))
    cut = int(len(ordered) * (1.0 - holdout))
    if cut == 0 or cut == len(ordered):
        raise ArrivalError(f"{len(ordered)} trips cannot be split {holdout}")
    return ordered[:cut], ordered[cut:]


# ------------------------------------------------------------------ the data

@dataclass(frozen=True)
class Dataset:
    """Rows in declared feature order, with their labels beside them."""

    names: list[str]
    rows: list[list[float]]
    labels: list[float]
    promised: list[float]
    trip_ids: list[str]

    def __len__(self) -> int:
        return len(self.rows)


def to_dataset(vectors: Sequence[FeatureVector], trips: Sequence[Trip],
               names: Sequence[str] | None = None) -> Dataset:
    """Pair feature vectors with labels by trip id.

    Paired by id rather than by position: two lists sorted by different keys
    line up for a while and then silently stop, and a label attached to the
    wrong row trains a model that is wrong in a way no metric names.
    """
    order = list(names or features.FEATURE_NAMES)
    by_id = {t.trip_id: t for t in trips}
    rows, labels, promised, trip_ids = [], [], [], []
    for vector in vectors:
        trip = by_id.get(vector.trip_id)
        if trip is None or not trip.is_labelled:
            continue
        rows.append(vector.ordered(order))
        labels.append(float(trip.tat_minutes))
        promised.append(float(trip.promised_minutes))
        trip_ids.append(trip.trip_id)
    if not rows:
        raise ArrivalError("no labelled trips to train or score on")
    return Dataset(order, rows, labels, promised, trip_ids)


# ---------------------------------------------------------------- the models

class BaselineModel:
    """The flat promise, which is what the business does today.

    It takes no features at all. That is the point: every figure the trained
    model reports is only interesting as a difference from this one, and a
    baseline that quietly used features would make that difference mean
    something else.
    """

    def __init__(self, flat_minutes: float, version: str = "unregistered"):
        self.feature_names: list[str] = []
        self.flat_minutes = float(flat_minutes)
        self.version = version

    @classmethod
    def fit(cls, trips: Sequence[Trip]) -> "BaselineModel":
        promises = [float(t.promised_minutes) for t in trips]
        if not promises:
            raise ArrivalError("no trips to read a promise from")
        return cls(_median(promises))

    def predict(self, vector: FeatureVector) -> float:
        return self.flat_minutes

    def payload(self) -> dict:
        return {"kind": "baseline", "feature_names": [],
                "flat_minutes": self.flat_minutes}

    @classmethod
    def from_payload(cls, payload: dict, version: str) -> "BaselineModel":
        return cls(payload["flat_minutes"], version=version)


class RidgeModel:
    """Ridge regression, normal equations, stdlib arithmetic.

    Features are normalised before fitting because `distance_m` runs into the
    thousands while `is_rush_hour` is zero or one; left raw, the ridge penalty
    would fall almost entirely on the small-scaled columns and quietly delete
    the features that cost the least to delete. The normalisation constants are
    part of the model and are saved with the weights -- a model restored
    without them predicts confident nonsense, which is why the registry treats
    them as weights rather than as metadata.
    """

    def __init__(self, feature_names: Sequence[str], weights: Sequence[float],
                 intercept: float, means: Sequence[float],
                 scales: Sequence[float], alpha: float = RIDGE_ALPHA,
                 version: str = "unregistered"):
        self.feature_names = list(feature_names)
        self.weights = [float(w) for w in weights]
        self.intercept = float(intercept)
        self.means = [float(m) for m in means]
        self.scales = [float(s) for s in scales]
        self.alpha = float(alpha)
        self.version = version

    @classmethod
    def fit(cls, data: Dataset, alpha: float = RIDGE_ALPHA) -> "RidgeModel":
        n, k = len(data.rows), len(data.names)
        means = [sum(row[j] for row in data.rows) / n for j in range(k)]
        scales = []
        for j in range(k):
            variance = sum((row[j] - means[j]) ** 2 for row in data.rows) / n
            # A column with no variance -- `is_rush_hour` in a training slice
            # that happens to contain no rush hour -- would divide by zero. A
            # scale of one leaves it centred at zero, so ridge gives it no
            # weight instead of an infinite one.
            scales.append(variance ** 0.5 or 1.0)

        z = [[(row[j] - means[j]) / scales[j] for j in range(k)]
             for row in data.rows]
        y_mean = sum(data.labels) / n
        y = [label - y_mean for label in data.labels]

        # Normal equations with a ridge term on the diagonal. The penalty is
        # not applied to the intercept: shrinking the intercept biases every
        # prediction towards zero minutes, which no amount of regularisation
        # was ever meant to buy.
        gram = [[sum(z[i][a] * z[i][b] for i in range(n)) for b in range(k)]
                for a in range(k)]
        for a in range(k):
            gram[a][a] += alpha
        moment = [sum(z[i][a] * y[i] for i in range(n)) for a in range(k)]

        weights = _solve(gram, moment)
        return cls(data.names, weights, y_mean, means, scales, alpha)

    def predict(self, vector: FeatureVector) -> float:
        row = vector.ordered(self.feature_names)
        total = self.intercept
        for value, mean, scale, weight in zip(row, self.means, self.scales,
                                              self.weights):
            total += weight * (value - mean) / scale
        return max(MINIMUM_MINUTES, total)

    def payload(self) -> dict:
        return {"kind": "ridge", "feature_names": list(self.feature_names),
                "weights": self.weights, "intercept": self.intercept,
                "means": self.means, "scales": self.scales,
                "alpha": self.alpha}

    @classmethod
    def from_payload(cls, payload: dict, version: str) -> "RidgeModel":
        return cls(payload["feature_names"], payload["weights"],
                   payload["intercept"], payload["means"], payload["scales"],
                   payload.get("alpha", RIDGE_ALPHA), version=version)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting.

    Pivoting on the largest remaining row rather than taking rows in order:
    two features that move together leave a near-zero pivot on the diagonal,
    and dividing by it turns small differences in the data into enormous
    differences in the weights.
    """
    n = len(rhs)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ArrivalError(
                f"feature {col} is collinear with another; raise the ridge "
                f"term or drop the duplicate rather than solving this")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            if factor:
                for c in range(col, n + 1):
                    a[row][c] -= factor * a[col][c]

    out = [0.0] * n
    for row in reversed(range(n)):
        total = a[row][n] - sum(a[row][c] * out[c] for c in range(row + 1, n))
        out[row] = total / a[row][row]
    return out


# --------------------------------------------------------------- the scoring

def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, so the figure belongs to a real trip.

    Interpolating between the two worst errors reports a miss that no delivery
    actually had, and this number has to survive being asked which trip it was.
    """
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(q / 100 * len(ordered))))
    return ordered[rank - 1]


def score(model, vectors: Sequence[FeatureVector], data: Dataset) -> Metrics:
    """What the model scored on trips it was not fitted on.

    `breach_rate` is the share of trips that took longer than the model said
    they would -- the promise the model would have made, broken. It is reported
    beside MAE because the two can move apart: a model can cut average error
    while breaking more promises, by shaving its estimates, and only one of
    those two numbers is what a customer experiences.
    """
    by_id = {v.trip_id: v for v in vectors}
    errors, breaches, squared = [], 0, 0.0
    for trip_id, label in zip(data.trip_ids, data.labels):
        prediction = model.predict(by_id[trip_id])
        error = abs(prediction - label)
        errors.append(error)
        squared += error ** 2
        if label > prediction:
            breaches += 1

    n = len(errors)
    return Metrics(n=n,
                   mae=sum(errors) / n,
                   rmse=(squared / n) ** 0.5,
                   p50_error=_median(errors),
                   p90_error=_percentile(errors, 90),
                   breach_rate=breaches / n)


# -------------------------------------------------------------- the pipeline

@dataclass(frozen=True)
class TrainingRun:
    """Everything one `main()` produced, so a test can assert on all of it."""

    model: RidgeModel
    baseline: BaselineModel
    leaky_in_flight: RidgeModel
    leaky_lifetime: RidgeModel
    metrics: dict[str, Metrics]
    training_window: tuple[dt.datetime, dt.datetime]
    n_train: int
    n_test: int

    @property
    def leakage_gap_mae(self) -> float:
        """How much better the leaky model *looks*, in minutes of MAE.

        Positive means the leak flatters it. None of that improvement would
        survive production, because at prediction time the trips it read had
        not finished -- and one of them is the trip being predicted.
        """
        return (self.metrics["ridge"].mae -
                self.metrics["ridge_leaky_in_flight"].mae)

    @property
    def lifetime_leakage_gap_mae(self) -> float:
        """The same figure for the windowless variant, which leaks far less."""
        return (self.metrics["ridge"].mae -
                self.metrics["ridge_leaky_lifetime"].mae)

    @property
    def lift_over_baseline_mae(self) -> float:
        """Minutes of MAE the honest model takes off the flat promise."""
        return self.metrics["baseline"].mae - self.metrics["ridge"].mae


def run(trips: Sequence[Trip], holdout: float = DEFAULT_HOLDOUT,
        alpha: float = RIDGE_ALPHA) -> TrainingRun:
    """Fit and score every model on one chronological split.

    Features are built across the whole table before the split, not separately
    per side. That is safe here precisely because `features.historical` is
    point-in-time: a test-set trip may legitimately see training-set trips that
    had finished before it was assigned, which is exactly what serving will
    see. Rebuilding features per side would instead hide the first hours of the
    test set from their own recent history and under-report the model.
    """
    train_trips, test_trips = chronological_split(trips, holdout)
    train_ids = {t.trip_id for t in train_trips}

    def fit_and_score(vectors):
        """Fit on the early trips, score on the late ones, same feature set."""
        train_vectors = [v for v in vectors if v.trip_id in train_ids]
        test_vectors = [v for v in vectors if v.trip_id not in train_ids]
        train_data = to_dataset(train_vectors, train_trips)
        test_data = to_dataset(test_vectors, test_trips)
        fitted = RidgeModel.fit(train_data, alpha)
        return (fitted, train_data, test_data,
                score(fitted, test_vectors, test_data))

    honest = features.historical(trips)
    ridge, honest_train, honest_test, honest_metrics = fit_and_score(honest)

    # Both leak flavours are fitted and scored the same way, on the same
    # chronological split, so the only thing that differs between the rows of
    # the table is which trips the aggregates were allowed to see.
    in_flight, _, _, in_flight_metrics = fit_and_score(
        features.leaky_in_flight_historical(trips))
    lifetime, _, _, lifetime_metrics = fit_and_score(
        features.leaky_lifetime_historical(trips))

    baseline = BaselineModel.fit(train_trips)
    honest_test_vectors = [v for v in honest if v.trip_id not in train_ids]

    metrics = {
        "baseline": score(baseline, honest_test_vectors, honest_test),
        "ridge": honest_metrics,
        "ridge_leaky_in_flight": in_flight_metrics,
        "ridge_leaky_lifetime": lifetime_metrics,
    }

    return TrainingRun(
        model=ridge, baseline=baseline, leaky_in_flight=in_flight,
        leaky_lifetime=lifetime, metrics=metrics,
        training_window=(train_trips[0].assigned_at,
                         train_trips[-1].assigned_at),
        n_train=len(honest_train), n_test=len(honest_test))


# ------------------------------------------------------------------ the card

_ROWS = (
    ("baseline", "flat promise"),
    ("ridge", "ridge, point-in-time"),
    ("ridge_leaky_in_flight", "ridge, leaky: in-flight"),
    ("ridge_leaky_lifetime", "ridge, leaky: lifetime"),
)


def comparison_table(result: TrainingRun) -> str:
    """The four scores side by side, which is the whole report."""
    header = (f"{'model':<24}{'n':>7}{'mae':>9}{'rmse':>9}"
              f"{'p50':>9}{'p90':>9}{'breach':>9}")
    lines = [header, "-" * len(header)]
    for key, label in _ROWS:
        m = result.metrics[key]
        lines.append(f"{label:<24}{m.n:>7}{m.mae:>9.3f}{m.rmse:>9.3f}"
                     f"{m.p50_error:>9.3f}{m.p90_error:>9.3f}"
                     f"{m.breach_rate:>9.3f}")
    return "\n".join(lines)


def main() -> TrainingRun:
    trips = source.load()
    result = run(trips)

    print(f"trips {len(trips)}  train {result.n_train}  test {result.n_test}")
    print(f"training window {result.training_window[0]} "
          f"-> {result.training_window[1]}")
    print()
    print(comparison_table(result))
    print()
    print(f"the honest model takes {result.lift_over_baseline_mae:.3f} min of "
          f"MAE off the flat promise. That is the improvement that exists.")
    print(f"filtering the window on assigned_at instead of delivered_at looks "
          f"{result.leakage_gap_mae:.3f} min of MAE better again, and is worth "
          f"nothing: those trips had not finished when the estimate was due, "
          f"and one of them is the trip being estimated.")
    print(f"dropping the window altogether looks "
          f"{result.lifetime_leakage_gap_mae:.3f} min better -- small enough "
          f"to be dismissed as noise, and wrong in exactly the same way.")

    version = registry.save(result.model, result.metrics["ridge"],
                            result.training_window)
    print(f"\nsaved {version}")
    return result


if __name__ == "__main__":
    main()
