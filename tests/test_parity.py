"""Training and serving must answer the same question.

This is the file the project is for. Everything else here can be rewritten; if
these tests pass, the model that was evaluated is the model that is deployed,
and if they fail, nothing downstream of them is trustworthy no matter what its
own tests say.

The failure these guard against has no symptom. The service returns a number,
the dashboard draws a line, the error is a little worse than the evaluation
promised, and that is indistinguishable from the world having changed. Months
of a model being quietly mediocre is the usual price, and the usual response is
to retrain it -- which does nothing, because the defect was never in the
weights.
"""

from __future__ import annotations

import datetime as dt

import pytest

from arrival import features, source
from arrival.contracts import LeakageError, Trip

BASE = dt.datetime(2026, 9, 1, 9, 0, 0)


def trip(trip_id: str, *, at: float, took: float | None = None,
         store: str = "NORTHGATE", driver: str = "D-0001",
         distance: int = 2200, promised: int = 19) -> Trip:
    assigned = BASE + dt.timedelta(minutes=at)
    if took is None:
        return Trip(trip_id, store, driver, assigned, distance, promised)
    return Trip(trip_id, store, driver, assigned, distance, promised,
                delivered_at=assigned + dt.timedelta(minutes=took),
                tat_minutes=float(took))


# ------------------------------------------------------------------- parity

def test_the_online_path_returns_what_the_offline_path_returned():
    """If these two disagree, every offline number this repository prints is
    about a model that does not exist. The service would be scoring features
    the training set never contained, and no amount of monitoring the
    predictions would say which of the two was wrong."""
    history = [
        trip("T-1", at=0, took=18),
        trip("T-2", at=5, took=24, driver="D-0002"),
        trip("T-3", at=20, took=12, store="RIVERSIDE"),
        trip("T-4", at=40, took=31),
        trip("T-5", at=55, took=9, distance=400),
    ]
    target = trip("T-target", at=90, took=22, distance=3100)

    offline = next(v for v in features.historical(history + [target])
                   if v.trip_id == "T-target")

    store = features.TripFeatureStore()
    for finished in history:
        store.observe(finished)
    online = store.online(target, target.assigned_at)

    assert online.as_of == offline.as_of
    assert online.values == offline.values


def test_parity_holds_for_every_trip_of_a_replayed_day():
    """One trip agreeing is a coincidence. This replays a day the way
    production experiences it -- trips arriving to be estimated, trips coming
    back as they finish -- and insists the two paths never diverge, including
    at the edges where a trip finishes in the same minute as the next one is
    assigned."""
    trips = source.generate(n=600, seed=9, days=3, drivers=5)
    offline = {v.trip_id: v.values for v in features.historical(trips)}

    store = features.TripFeatureStore()
    finished = sorted(trips, key=lambda t: (t.delivered_at, t.trip_id))
    cursor = 0

    for target in sorted(trips, key=lambda t: (t.assigned_at, t.trip_id)):
        as_of = target.assigned_at
        while cursor < len(finished) and finished[cursor].delivered_at <= as_of:
            store.observe(finished[cursor])
            cursor += 1
        assert store.online(target, as_of).values == offline[target.trip_id], \
            f"{target.trip_id} disagreed between the two paths"


def test_the_online_path_is_not_fooled_by_history_arriving_late():
    """Trips do not reach the store in the order they finished -- a retry, a
    backfill, a consumer that fell behind. The aggregate has to depend on the
    trips' timestamps and not on when the store heard about them, or a replay
    produces different features from the live run that preceded it."""
    history = [trip("T-1", at=0, took=18), trip("T-2", at=20, took=24),
               trip("T-3", at=40, took=12)]
    target = trip("T-target", at=70, took=22)

    offline = next(v for v in features.historical(history + [target])
                   if v.trip_id == "T-target")

    store = features.TripFeatureStore()
    for finished in reversed(history):
        store.observe(finished)

    assert store.online(target, target.assigned_at).values == offline.values


def test_the_online_store_withholds_a_trip_delivered_after_the_estimate():
    """A store that has been pushed tomorrow's trips -- a replay, a clock that
    ran ahead -- must still answer as of the instant it was asked about.
    Otherwise the one path that is supposed to be reproducible stops being
    reproducible, and backtests disagree with the live run for reasons nobody
    can reconstruct."""
    store = features.TripFeatureStore()
    store.observe(trip("T-past", at=0, took=10))
    store.observe(trip("T-future", at=200, took=10))

    as_of = BASE + dt.timedelta(minutes=30)
    target = trip("T-target", at=30, took=22)

    assert store.online(target, as_of).values["driver_trips_120m"] == 1.0


# ----------------------------------------------- what parity is guarding from

def test_a_leaky_aggregate_looks_better_than_the_correct_one():
    """Proof that the parity test is worth having. A window that reads forward
    produces a *different* number, and a flatteringly accurate one -- closer
    to the label than the honest feature can be. Without this test, parity
    could pass because both paths were wrong in the same way."""
    target = trip("T-target", at=60, took=20)
    past = trip("T-past", at=0, took=40)            # finished at +40
    peer = trip("T-peer", at=59, took=20)           # finishes at +79, in flight

    trips = [past, peer, target]
    honest = next(v for v in features.historical(trips)
                  if v.trip_id == "T-target")
    leaky = next(v for v in features.leaky_in_flight_historical(trips)
                 if v.trip_id == "T-target")

    honest_mean = honest.values["driver_mean_tat_120m"]
    leaky_mean = leaky.values["driver_mean_tat_120m"]
    label = target.tat_minutes

    assert honest_mean == 40.0
    assert leaky_mean != honest_mean
    # The leaky feature has read the trip's own duration and a peer's that had
    # not finished, so it sits far closer to the answer than anything knowable
    # at assignment time could.
    assert abs(leaky_mean - label) < abs(honest_mean - label)
    assert leaky.values["driver_trips_120m"] > \
        honest.values["driver_trips_120m"]


def test_the_leaky_variant_reads_the_trip_it_is_predicting():
    """Named outright because it is the heart of the demonstration: filtering
    the window on assigned_at makes the trip a member of its own history, so
    its duration becomes one of its own features. Any model will find that,
    and offline it looks like skill."""
    target = trip("T-target", at=0, took=47)
    leaky = next(v for v in features.leaky_in_flight_historical([target])
                 if v.trip_id == "T-target")

    assert leaky.values["driver_trips_120m"] == 1.0
    assert leaky.values["driver_mean_tat_120m"] == 47.0


def test_the_windowless_variant_leaks_too_even_though_it_looks_tame():
    """A lifetime GROUP BY dilutes the leak across a driver's whole record, so
    the offline gap is small enough to argue about. It is wrong for exactly the
    same reason, and a small gap is the more dangerous kind: nobody rejects a
    model over it."""
    target = trip("T-target", at=60, took=20)
    future = trip("T-future", at=400, took=4)       # long after the estimate

    trips = [target, future]
    honest = next(v for v in features.historical(trips)
                  if v.trip_id == "T-target")
    leaky = next(v for v in features.leaky_lifetime_historical(trips)
                 if v.trip_id == "T-target")

    assert honest.values["driver_trips_120m"] == 0.0
    assert leaky.values["driver_trips_120m"] == 2.0
    assert leaky.values["driver_mean_tat_120m"] == pytest.approx(12.0)


# ------------------------------------------------------------- the assertion

def test_leakage_error_when_a_window_is_handed_an_unfinished_trip():
    """The guard that stops a future version of this code from leaking
    silently. A caller that filters wrongly gets an exception instead of a
    plausible number, which is the difference between a failed training run
    and a quarter of believing a false score."""
    target = trip("T-target", at=60, took=20)
    still_out = trip("T-flying", at=59, took=30)    # delivered at +89

    with pytest.raises(LeakageError):
        features.compute_vector(target, target.assigned_at,
                                {("driver", 120): [still_out]})


def test_leakage_error_names_the_trip_and_the_window():
    """An exception that says only "leakage" sends whoever reads it to go
    hunting. The message has to carry the trip, the window and both
    timestamps, because the person reading it is usually mid-deploy."""
    target = trip("T-target", at=60, took=20)
    still_out = trip("T-flying", at=59, took=30)

    with pytest.raises(LeakageError) as raised:
        features.compute_vector(target, target.assigned_at,
                                {("store", 60): [still_out]})

    message = str(raised.value)
    assert "T-target" in message and "T-flying" in message
    assert "store" in message and "60m" in message


def test_an_unlabelled_trip_in_a_window_is_leakage_too():
    """A trip with no delivered_at has not finished either. Treating it as
    merely missing would let it through to the aggregate, where `None` either
    crashes or, worse, is coerced to zero."""
    target = trip("T-target", at=60, took=20)
    unlabelled = trip("T-unknown", at=10, took=None)

    with pytest.raises(LeakageError):
        features.compute_vector(target, target.assigned_at,
                                {("driver", 120): [unlabelled]})


def test_the_honest_paths_never_raise_leakage_on_real_data():
    """The guard has to be quiet when the code is right. A leakage check that
    fires on correct input gets commented out within a week, and then it is
    not a check at all."""
    trips = source.generate(n=500, seed=12, days=3, drivers=4)

    vectors = features.historical(trips)

    store = features.TripFeatureStore()
    store.online_store.extend(t for t in trips
                              if t.delivered_at <= trips[-1].assigned_at)
    last = trips[-1]

    assert len(vectors) == len(trips)
    assert store.online(last, last.assigned_at).trip_id == last.trip_id
