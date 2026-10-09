"""What the feature layer must be true about before anything downstream of it
means anything.

These tests are mostly about one word: *finished*. Every aggregate here is
allowed to read a trip only once that trip has been delivered, and almost
every way of getting this wrong produces a number rather than an error.
"""

from __future__ import annotations

import datetime as dt

import pytest

from arrival import features, source
from arrival.contracts import Trip

BASE = dt.datetime(2026, 9, 1, 9, 0, 0)


def trip(trip_id: str, *, at: float, took: float | None = None,
         store: str = "NORTHGATE", driver: str = "D-0001",
         distance: int = 2200, promised: int = 19) -> Trip:
    """A trip assigned `at` minutes past BASE, which took `took` minutes."""
    assigned = BASE + dt.timedelta(minutes=at)
    if took is None:
        return Trip(trip_id, store, driver, assigned, distance, promised)
    return Trip(trip_id, store, driver, assigned, distance, promised,
                delivered_at=assigned + dt.timedelta(minutes=took),
                tat_minutes=float(took))


def values_for(trips, target_id: str) -> dict[str, float]:
    return next(v.values for v in features.historical(trips)
                if v.trip_id == target_id)


# ----------------------------------------------------------- the declaration

def test_every_declared_feature_comes_back_with_a_value():
    """A spec without a value is a column the model was promised and never
    got; it fails at `ordered` in training, or worse, is quietly dropped and
    the remaining columns shift one place to the left."""
    store = features.TripFeatureStore()
    declared = [spec.name for spec in store.specs()]
    computed = values_for([trip("T-1", at=0, took=20)], "T-1")

    assert declared == features.FEATURE_NAMES
    assert sorted(computed) == sorted(declared)


def test_every_windowed_feature_declares_the_window_it_uses():
    """A window that lives in the code and not in the spec cannot be checked
    against the online store's retention, and the mismatch only shows up as
    serving features that disagree with training ones."""
    for spec in features.TripFeatureStore().specs():
        if spec.entity == "trip":
            assert spec.window_minutes is None
        else:
            assert spec.window_minutes, f"{spec.name} aggregates over nothing"


# ------------------------------------------------------- features of the trip

def test_the_derived_features_read_only_the_assignment():
    """Hour, weekday and distance are the features that are honestly available
    when the estimate is due. If they ever started depending on anything else
    on the trip, the model would be using the outcome to explain itself."""
    friday_rush = trip("T-1", at=0, took=20, distance=1799)
    computed = values_for([friday_rush], "T-1")

    assert computed["distance_m"] == 1799.0
    assert computed["hour_of_day"] == 9.0
    assert computed["day_of_week"] == float(BASE.weekday())
    assert computed["is_rush_hour"] == 0.0
    assert computed["log_distance"] == pytest.approx(7.4956, abs=1e-4)


def test_the_rush_hour_flag_follows_the_hour_of_assignment():
    """Rush hour costs every trip time and is knowable up front. Flagged off
    the wrong timestamp -- delivery rather than assignment -- it becomes a
    feature about when the trip ended, which is the label in disguise."""
    at_noon = trip("T-1", at=3 * 60, took=20)
    assert at_noon.assigned_at.hour == 12
    assert values_for([at_noon], "T-1")["is_rush_hour"] == 1.0


# ------------------------------------------------------ point-in-time windows

def test_a_trip_still_on_the_road_is_not_in_the_aggregate():
    """The whole project in one assertion. A trip assigned before ours and
    delivered after it was in flight when our estimate was due; counting its
    duration means training on a number production will not have."""
    in_flight = trip("T-past", at=0, took=90)       # delivered at +90
    target = trip("T-now", at=30, took=20)          # assigned at +30

    computed = values_for([in_flight, target], "T-now")

    assert computed["driver_trips_120m"] == 0.0
    assert computed["driver_mean_tat_120m"] == float(target.promised_minutes)


def test_a_trip_that_finished_one_minute_ago_is_in_the_aggregate():
    """The counterpart. A window that excludes genuinely finished trips is
    just as wrong as one that includes unfinished ones, and it fails quietly
    as a model that never learned the driver signal was there."""
    finished = trip("T-past", at=0, took=29)        # delivered at +29
    target = trip("T-now", at=30, took=20)

    computed = values_for([finished, target], "T-now")

    assert computed["driver_trips_120m"] == 1.0
    assert computed["driver_mean_tat_120m"] == 29.0


def test_the_window_drops_a_trip_once_it_falls_out_the_back():
    """A trailing window that never forgets is a lifetime average wearing a
    window's name, and it stops tracking the thing it was added to track:
    tonight's conditions, not this driver's year."""
    old = trip("T-old", at=0, took=10)              # delivered at +10
    recent = trip("T-recent", at=100, took=10)      # delivered at +110
    target = trip("T-now", at=131, took=20)         # window opens at +11

    computed = values_for([old, recent, target], "T-now")

    assert computed["driver_trips_120m"] == 1.0
    assert computed["driver_mean_tat_120m"] == 10.0


def test_an_unlabelled_trip_never_enters_an_aggregate():
    """A trip whose delivered event went missing has no duration. Counting it
    as anything teaches the model that broken telemetry means a fast trip."""
    lost = trip("T-lost", at=0, took=None)
    target = trip("T-now", at=30, took=20)

    computed = values_for([lost, target], "T-now")

    assert computed["driver_trips_120m"] == 0.0


def test_the_forward_sweep_agrees_with_the_obvious_slow_filter():
    """The sweep is the fast version of a filter per row. Written once as a
    deque and a cursor, it is also the version that can be subtly wrong -- an
    eviction off by one trip, a bucket shared between two windows -- and the
    only honest check is the stupid implementation, kept here on purpose."""
    trips = source.generate(n=400, seed=5, days=4, drivers=6)
    swept = {v.trip_id: v.values for v in features.historical(trips)}

    for target in trips:
        as_of = target.assigned_at
        window = as_of - dt.timedelta(minutes=120)
        peers = [t for t in trips
                 if t.driver_id == target.driver_id
                 and t.delivered_at is not None
                 and window < t.delivered_at <= as_of]
        expected = (sum(t.tat_minutes for t in peers) / len(peers) if peers
                    else float(target.promised_minutes))

        assert swept[target.trip_id]["driver_trips_120m"] == len(peers)
        assert swept[target.trip_id]["driver_mean_tat_120m"] == \
            pytest.approx(expected)


def test_the_sweep_scales_past_the_point_a_quadratic_one_would():
    """Ten thousand trips is a small table and a large number of pairs. If
    this ever goes quadratic, the training job stops being rerunnable, and a
    pipeline nobody reruns is a pipeline nobody fixes."""
    trips = source.generate(n=10_000, seed=6, days=20, drivers=40)
    assert len(features.historical(trips)) == len(trips)


# ------------------------------------------------------------- the cold start

def test_a_driver_with_no_finished_trips_gets_the_promise_not_zero():
    """A new driver's mean is unknown, not zero. Zero reads as a driver who
    delivers instantly, and a linear model will happily predict exactly that
    -- for the drivers whose first shift it is, which is when a wild estimate
    costs the most."""
    target = trip("T-now", at=0, took=20, promised=23)
    computed = values_for([target], "T-now")

    assert computed["driver_mean_tat_120m"] == 23.0
    assert computed["driver_trips_120m"] == 0.0


def test_the_cold_start_is_visible_in_the_count_beside_it():
    """The fallback is a guess, and the model is only allowed to treat it as
    one because the count says how many trips the mean came from. Drop the
    count and a fabricated mean is indistinguishable from a measured one."""
    one_finished = trip("T-past", at=0, took=14)
    target = trip("T-now", at=20, took=20)

    cold = values_for([target], "T-now")
    warm = values_for([one_finished, target], "T-now")

    assert cold["driver_trips_120m"] == 0.0
    assert warm["driver_trips_120m"] == 1.0
    assert warm["driver_mean_tat_120m"] == 14.0


def test_a_store_with_no_history_gets_the_declared_pack_midpoint():
    """Handling time has no business promise to fall back on, so the fallback
    is a declared constant. Declared, so that a reader can find out what the
    number means instead of inferring it from a zero."""
    computed = values_for([trip("T-now", at=0, took=20)], "T-now")
    assert computed["store_pack_p50_60m"] == features.COLD_START_PACK_MINUTES


def test_the_median_pack_time_ignores_one_order_that_sat_forever():
    """A single stuck order is not a slow kitchen. A mean would let it set the
    store's handling time for the next hour and push every estimate out of
    that store up, which is how one bad trip becomes a hundred bad ETAs."""
    normal = [trip(f"T-{i}", at=i, took=10, distance=0) for i in range(5)]
    stuck = trip("T-stuck", at=0, took=59, distance=0)
    target = trip("T-now", at=60, took=20, distance=0)

    computed = values_for(normal + [stuck, target], "T-now")

    assert computed["store_pack_p50_60m"] == 10.0
    # The mean over the same six trips is pulled almost twice as high, which is
    # the number the store would have been judged by.
    assert computed["store_mean_tat_120m"] == pytest.approx(109 / 6)


def test_handling_time_is_what_is_left_after_the_ride():
    """The warehouse never records when the rider left the store, so handling
    time is a residual of duration and distance. Said out loud here because a
    reader who assumes it is measured will trust it further than they should."""
    far = trip("T-past", at=0, took=30,
               distance=int(10 * features.NOMINAL_SPEED_M_PER_MIN))
    target = trip("T-now", at=40, took=20)

    computed = values_for([far, target], "T-now")

    assert computed["store_pack_p50_60m"] == pytest.approx(20.0)


# ------------------------------------------------------------- the online side

def test_the_online_store_refuses_a_trip_that_has_not_finished():
    """An in-flight trip has no duration. Accepting one would put a `None`
    into an aggregate, and the serving path would crash on the trip after it
    rather than on the push that caused it."""
    with pytest.raises(ValueError):
        features.OnlineStore().push(trip("T-flying", at=0, took=None))


def test_the_online_store_will_not_be_built_too_small_to_answer():
    """Retention shorter than the widest declared window means the online
    path quietly answers a narrower question than training asked. Refused at
    construction, because by serving time there is nothing left to notice."""
    with pytest.raises(ValueError):
        features.OnlineStore(retain_minutes=5)


def test_the_online_store_forgets_only_what_no_window_can_reach():
    """Eviction keeps memory flat. Evicting too eagerly would make the online
    aggregate disagree with the offline one, which is the one failure this
    layer is not allowed to have."""
    store = features.OnlineStore(retain_minutes=120)
    store.extend([trip(f"T-{i}", at=i * 30, took=5) for i in range(20)])

    as_of = BASE + dt.timedelta(minutes=20 * 30)
    held = store.window("driver", "D-0001", 120, as_of)

    assert store.trips_held() < 20
    assert len(held) == 4
