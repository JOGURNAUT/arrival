"""That the training job produces a defensible artifact, and that the numbers
it prints mean what the table says they mean.

Two of these tests are about the split and the registry rather than the model,
because those are the two places where a mistake is invisible: a split that
leaks makes every score here optimistic, and an artifact that loses its feature
order makes a good model serve nonsense.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from arrival import registry, source, train
from arrival.contracts import ArrivalError, FeatureVector, Metrics

BASE = dt.datetime(2026, 9, 1, 9, 0, 0)


@pytest.fixture(scope="module")
def result() -> train.TrainingRun:
    """One training run, shared. Seeded, so a failure here is reproducible and
    not something that happens on a Tuesday."""
    return train.run(source.generate(n=2000, seed=21, days=10, drivers=8))


# -------------------------------------------------------------- the split

def test_the_split_never_puts_a_later_trip_in_train_than_in_test():
    """A random split hands the model Tuesday evening while scoring it on
    Tuesday afternoon, which is the leakage this whole repository is about,
    committed one level above the feature layer where no assertion can see
    it. Production is always asked about a time it has never seen."""
    trips = source.generate(n=400, seed=4, days=5, drivers=6)
    earlier, later = train.chronological_split(trips, holdout=0.25)

    assert max(t.assigned_at for t in earlier) <= \
        min(t.assigned_at for t in later)
    assert len(earlier) + len(later) == len(trips)
    assert not {t.trip_id for t in earlier} & {t.trip_id for t in later}


def test_a_split_that_would_leave_a_side_empty_is_refused():
    """Scoring on nothing reports an MAE of whatever the last trip happened to
    be, and reports it with the same confidence as a real figure."""
    trips = source.generate(n=20, seed=4, days=1, drivers=2)

    with pytest.raises(ArrivalError):
        train.chronological_split(trips, holdout=0.999)
    with pytest.raises(ArrivalError):
        train.chronological_split(trips, holdout=1.5)
    with pytest.raises(ArrivalError):
        train.chronological_split(trips, holdout=0.0)


# ------------------------------------------------------------- the models

def test_the_trained_model_beats_the_flat_promise_on_mae(result):
    """The only question that decides whether any of this ships. If this
    assertion ever fails, the honest report is that per-trip estimation did
    not beat a constant on this data -- not a looser threshold."""
    baseline = result.metrics["baseline"]
    ridge = result.metrics["ridge"]

    assert ridge.n == baseline.n
    assert ridge.mae < baseline.mae
    assert ridge.p90_error < baseline.p90_error


def test_the_leaky_model_looks_better_offline_than_the_honest_one(result):
    """The headline. The leaky model is not better -- it cannot be, it reads
    trips that had not finished -- and it still scores better on the held-out
    set. The size of that gap is how much a careless window can hide."""
    honest = result.metrics["ridge"]
    leaky = result.metrics["ridge_leaky_in_flight"]

    assert leaky.mae < honest.mae
    assert leaky.p50_error < honest.p50_error
    # How large the gap gets depends on how dense the driver's history is, so
    # the threshold here is only "large enough that a review would wave it
    # through as an improvement". `python -m arrival.train` prints the figure
    # for the data actually to hand, which is the number worth quoting.
    assert result.leakage_gap_mae / honest.mae > 0.02


def test_the_baseline_asks_for_no_features_at_all(result):
    """It has to stay the business's flat promise. A baseline that quietly
    used a feature would make every improvement reported against it mean
    something other than what the table claims."""
    vector = FeatureVector("T-anything", BASE, {})

    assert result.baseline.feature_names == []
    assert result.baseline.predict(vector) == result.baseline.flat_minutes


def test_the_model_refuses_a_vector_that_is_missing_a_feature(result):
    """Serving a model a short vector is the failure that does not crash: the
    remaining values slide into the wrong weights and the answer is confident
    and wrong. `FeatureVector.ordered` raises instead."""
    full = {name: 1.0 for name in result.model.feature_names}
    short = dict(full)
    short.pop(result.model.feature_names[-1])

    assert result.model.predict(FeatureVector("T-1", BASE, full)) > 0

    with pytest.raises(KeyError):
        result.model.predict(FeatureVector("T-1", BASE, short))


def test_the_model_reads_features_by_name_not_by_position(result):
    """A dict that happens to iterate in a different order must not change the
    prediction. This is the bug that survives every type check and every
    smoke test, and shows up only as an error rate nobody can place."""
    forwards = {name: float(i + 1)
                for i, name in enumerate(result.model.feature_names)}
    backwards = dict(reversed(list(forwards.items())))

    assert result.model.predict(FeatureVector("T-1", BASE, forwards)) == \
        result.model.predict(FeatureVector("T-1", BASE, backwards))


def test_the_ridge_recovers_a_relationship_that_is_actually_there():
    """The arithmetic is written out by hand here, so it gets checked against
    a case with a known answer. A solver that is subtly wrong produces weights
    that look like a model and predict like a coin."""
    rows = [[float(x), float(x * x % 7)] for x in range(1, 60)]
    labels = [4.0 + 3.0 * row[0] for row in rows]
    data = train.Dataset(["x", "noise"], rows, labels,
                         [19.0] * len(rows),
                         [f"T-{i}" for i in range(len(rows))])

    model = train.RidgeModel.fit(data, alpha=1e-6)
    predicted = model.predict(FeatureVector("T-new", BASE,
                                            {"x": 10.0, "noise": 2.0}))

    assert predicted == pytest.approx(34.0, abs=0.1)


def test_a_prediction_is_never_zero_or_negative_minutes(result):
    """Linear models extrapolate, and far enough out they extrapolate through
    zero. An ETA of minus four minutes is not a bad estimate, it is an outage
    that reaches the customer as a delivery that is already late."""
    absurd = {name: -1e6 for name in result.model.feature_names}

    assert result.model.predict(FeatureVector("T-1", BASE, absurd)) >= \
        train.MINIMUM_MINUTES


def test_the_metrics_count_the_trips_they_were_measured_on(result):
    """An MAE over forty trips and one over forty thousand are different
    claims. They are the same number on a slide unless `n` travels with it."""
    for metrics in result.metrics.values():
        assert metrics.n == result.n_test
        assert 0.0 <= metrics.breach_rate <= 1.0
        assert metrics.p90_error >= metrics.p50_error


def test_breach_rate_counts_promises_the_model_would_have_broken():
    """MAE and breach rate can move in opposite directions: a model can cut
    average error while breaking more promises, by shaving its estimates. The
    customer only experiences one of those."""
    rows = [[1.0], [1.0]]
    data = train.Dataset(["x"], rows, [10.0, 30.0], [19.0, 19.0],
                         ["T-1", "T-2"])
    vectors = [FeatureVector("T-1", BASE, {"x": 1.0}),
               FeatureVector("T-2", BASE, {"x": 1.0})]

    optimistic = train.BaselineModel(5.0)
    generous = train.BaselineModel(60.0)

    assert train.score(optimistic, vectors, data).breach_rate == 1.0
    assert train.score(generous, vectors, data).breach_rate == 0.0


# ----------------------------------------------------------- the registry

def _metrics() -> Metrics:
    return Metrics(n=10, mae=1.0, rmse=2.0, p50_error=0.5, p90_error=3.0,
                   breach_rate=0.1)


def test_the_registry_round_trips_a_model_and_its_feature_order(result,
                                                                tmp_path):
    """A model restored with its columns in a different order does not fail --
    it answers, and every answer is wrong by an amount nobody can see. The
    order is the artifact as much as the weights are."""
    window = result.training_window
    version = registry.save(result.model, result.metrics["ridge"], window,
                            root=tmp_path)
    restored = registry.load(version, root=tmp_path)

    assert restored.feature_names == result.model.feature_names
    assert restored.version == version

    vector = FeatureVector("T-1", BASE,
                           {name: float(i + 1) for i, name
                            in enumerate(result.model.feature_names)})
    assert restored.predict(vector) == pytest.approx(
        result.model.predict(vector))


def test_the_normalisation_constants_survive_the_round_trip(result, tmp_path):
    """They are weights, not metadata. A model reloaded without them scales
    every feature wrongly and predicts with total confidence."""
    version = registry.save(result.model, result.metrics["ridge"],
                            result.training_window, root=tmp_path)
    restored = registry.load(version, root=tmp_path)

    assert restored.means == result.model.means
    assert restored.scales == result.model.scales
    assert restored.intercept == pytest.approx(result.model.intercept)


def test_the_version_is_the_content_so_the_same_model_is_the_same_version():
    """A counter cannot tell you whether the artifact on the server is the one
    that was evaluated. A content hash can, and it makes a retrain that
    changed nothing a no-op instead of a new deployment."""
    window = (BASE, BASE + dt.timedelta(days=1))
    one = train.BaselineModel(19.0)
    same = train.BaselineModel(19.0)
    different = train.BaselineModel(21.0)

    assert registry.version_of(one.payload(), window) == \
        registry.version_of(same.payload(), window)
    assert registry.version_of(one.payload(), window) != \
        registry.version_of(different.payload(), window)


def test_a_model_fitted_on_a_different_window_is_a_different_version():
    """Same weights from a different fortnight is a different claim about the
    world. Sharing a version with it would mean two rows of metrics
    attributed to one artifact, and no way back to which was which."""
    model = train.BaselineModel(19.0)
    september = (BASE, BASE + dt.timedelta(days=7))
    october = (BASE + dt.timedelta(days=30), BASE + dt.timedelta(days=37))

    assert registry.version_of(model.payload(), september) != \
        registry.version_of(model.payload(), october)


def test_saving_the_same_model_twice_is_not_an_error(tmp_path):
    """Retraining on unchanged data should land on the same artifact rather
    than needing anybody to think about it."""
    window = (BASE, BASE + dt.timedelta(days=1))
    first = registry.save(train.BaselineModel(19.0), _metrics(), window,
                          root=tmp_path)
    second = registry.save(train.BaselineModel(19.0), _metrics(), window,
                           root=tmp_path)

    assert first == second
    assert registry.versions(root=tmp_path) == [first]


def test_a_different_model_cannot_take_an_occupied_version(tmp_path):
    """The one thing a content-addressed registry must never do is overwrite.
    Every prediction already logged against a version would silently be
    attributed to weights that no longer exist."""
    window = (BASE, BASE + dt.timedelta(days=1))
    version = registry.save(train.BaselineModel(19.0), _metrics(), window,
                            root=tmp_path)

    card_path = tmp_path / version / registry.CARD_NAME
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["payload"]["flat_minutes"] = 42.0
    card_path.write_text(json.dumps(card), encoding="utf-8")

    with pytest.raises(ArrivalError):
        registry.save(train.BaselineModel(19.0), _metrics(), window,
                      root=tmp_path)


def test_latest_resolves_the_most_recently_saved_model(tmp_path):
    """Deploying "latest" has to mean the newest model and not the one whose
    hash sorts first, which is what an alphabetical listing would give."""
    window = (BASE, BASE + dt.timedelta(days=1))
    older = registry.save(train.BaselineModel(19.0), _metrics(), window,
                          root=tmp_path)
    newer = registry.save(train.BaselineModel(23.0), _metrics(), window,
                          root=tmp_path)

    assert {older, newer} == set(registry.versions(root=tmp_path))
    assert registry.latest(root=tmp_path) == newer
    assert registry.load(registry.latest(root=tmp_path),
                         root=tmp_path).flat_minutes == 23.0


def test_asking_for_a_model_that_was_never_saved_says_so(tmp_path):
    """Serving would otherwise start with no model and the first symptom
    would be a stack trace in a request handler."""
    with pytest.raises(ArrivalError):
        registry.load("000000000000", root=tmp_path)
    with pytest.raises(ArrivalError):
        registry.latest(root=tmp_path)
