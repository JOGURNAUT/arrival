"""Feature definitions, and the two ways of asking for them.

Every feature in this project is declared once, in `_DERIVED` or `_AGGREGATES`
below, beside the function that computes it. Training and serving then differ
only in how the history is *fetched*: the arithmetic that turns history into a
number is one function, called by both. The reason for that discipline is
mundane and expensive -- a training aggregate written in SQL and a serving
lookup written in Python agree on the day they are written and drift apart a
month later, and nothing fails loudly when they do. The model simply turns out
worse than its evaluation promised, with no line of code to point at.

The other half of the discipline is time. An aggregate here may only read trips
that had already *finished* by the instant it is computed for:

    delivered_at <= as_of

Not `assigned_at <= as_of`. A trip assigned at 12:00 and delivered at 12:35 was
still on the road at 12:10; its duration was not a fact yet, and a feature that
uses it is reading the answer off the back of the paper. That bug does not look
like a bug -- it looks like a good model -- so `compute_vector` raises
`LeakageError` rather than trusting its caller to have filtered honestly.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .contracts import FeatureSpec, FeatureVector, LeakageError, Trip

# The hours Dispatch's flat promise is breached most. Knowable from
# `assigned_at` alone, which is what makes it a fair feature where the driver's
# next trip is not.
RUSH_HOURS = frozenset({12, 13, 19, 20, 21})

# A metres-per-minute divisor used to separate handling time from riding time.
# Declared, not fitted: the warehouse records when a trip was assigned and when
# it was delivered, never when the rider actually left the store, so handling
# time can only be a residual. A wrong divisor tilts `store_pack_p50_60m` with
# distance; it stays useful because every store is compared through the same
# wrong divisor.
NOMINAL_SPEED_M_PER_MIN = 220.0

# Handling time has no business promise to fall back on, so a store with no
# finished trips in the window gets a declared midpoint of the handling times
# Dispatch observes. Declared and documented, never silently zero.
COLD_START_PACK_MINUTES = 8.0


def _cold_start_minutes(trip: Trip) -> float:
    """What a driver or store with nothing in the window is worth.

    The flat promise is the honest answer: it is what the business already
    assumes about a delivery it knows nothing else about, and the count feature
    sits beside it at zero so the model can learn how far to trust a mean built
    from no trips. Emitting 0.0 would read as "this driver finishes instantly",
    and a linear model fits that happily -- hardest of all on new drivers, who
    are the people the estimate matters most for.
    """
    return float(trip.promised_minutes)


# ------------------------------------------------------- declared definitions

@dataclass(frozen=True)
class _Derived:
    """A feature that needs nothing but the trip in front of it."""

    spec: FeatureSpec
    of: Callable[[Trip], float]


@dataclass(frozen=True)
class _Aggregate:
    """A feature that needs an entity's recent finished trips.

    `of` is handed only trips already cleared as finished before the instant
    being computed for, so it never has to reason about time -- which is the
    point. Time is handled in one place, arithmetic in another.
    """

    spec: FeatureSpec
    of: Callable[[list[Trip], Trip], float]

    @property
    def entity(self) -> str:
        return self.spec.entity

    @property
    def window_minutes(self) -> int:
        return int(self.spec.window_minutes or 0)


def _implied_pack(trip: Trip) -> float:
    """The minutes of a trip that were not riding.

    Left unclamped deliberately. A fast rider produces a negative residual, and
    flooring it at zero would file that rider's speed under the store's kitchen
    and make a quick store look slow.
    """
    return float(trip.tat_minutes) - trip.distance_m / NOMINAL_SPEED_M_PER_MIN


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _mean_tat(finished: list[Trip], trip: Trip) -> float:
    if not finished:
        return _cold_start_minutes(trip)
    return sum(float(t.tat_minutes) for t in finished) / len(finished)


def _count(finished: list[Trip], trip: Trip) -> float:
    return float(len(finished))


def _p50_pack(finished: list[Trip], trip: Trip) -> float:
    if not finished:
        return COLD_START_PACK_MINUTES
    # Sorted inside the reducer so the answer cannot depend on the order the
    # history happened to arrive in. The offline sweep and the online store
    # hold the same trips; parity has to survive them holding them in
    # different orders.
    return _median([_implied_pack(t) for t in finished])


_DERIVED: list[_Derived] = [
    _Derived(
        FeatureSpec("distance_m", "trip",
                    "Metres to the drop, known the instant the trip is "
                    "assigned."),
        lambda t: float(t.distance_m),
    ),
    _Derived(
        FeatureSpec("log_distance", "trip",
                    "log1p of distance. Duration rises with distance, but the "
                    "thin tail of very far drops would otherwise set the slope "
                    "for every short one."),
        lambda t: math.log1p(float(t.distance_m)),
    ),
    _Derived(
        FeatureSpec("hour_of_day", "trip",
                    "Hour of assignment."),
        lambda t: float(t.assigned_at.hour),
    ),
    _Derived(
        FeatureSpec("is_rush_hour", "trip",
                    "Assignment fell in a rush hour, when every trip on the "
                    "road costs more than its distance says."),
        lambda t: 1.0 if t.assigned_at.hour in RUSH_HOURS else 0.0,
    ),
    _Derived(
        FeatureSpec("day_of_week", "trip",
                    "Monday is 0. A Saturday evening is not a Tuesday "
                    "evening."),
        lambda t: float(t.assigned_at.weekday()),
    ),
]

_AGGREGATES: list[_Aggregate] = [
    _Aggregate(
        FeatureSpec("store_pack_p50_60m", "store",
                    "Median handling time at this store over the trailing "
                    "hour. Median rather than mean: one trip that sat for an "
                    "hour is a stuck order, not a slow kitchen.",
                    window_minutes=60),
        _p50_pack,
    ),
    _Aggregate(
        FeatureSpec("driver_mean_tat_120m", "driver",
                    "This driver's mean finished duration over the trailing "
                    "two hours. Drivers hold their pace, so this carries real "
                    "signal -- and leaks spectacularly if the window is ever "
                    "allowed to look forward.",
                    window_minutes=120),
        _mean_tat,
    ),
    _Aggregate(
        FeatureSpec("driver_trips_120m", "driver",
                    "How many finished trips that mean was built from. It "
                    "travels beside the mean so that a mean of one trip and a "
                    "mean of twenty are not the same number to the model.",
                    window_minutes=120),
        _count,
    ),
    _Aggregate(
        FeatureSpec("store_mean_tat_120m", "store",
                    "Mean finished duration out of this store over the "
                    "trailing two hours: queue depth, weather and short "
                    "staffing, without having to measure any of them.",
                    window_minutes=120),
        _mean_tat,
    ),
]

#: Declared order. Models are trained and served against this list, never
#: against whatever order a dict happened to iterate in.
FEATURE_NAMES: list[str] = ([d.spec.name for d in _DERIVED] +
                            [a.spec.name for a in _AGGREGATES])

#: The (entity, window) pairs a retrieval has to fetch. One bucket per pair,
#: shared by every aggregate declared on it, so two features over the same
#: window can never disagree about which trips were in it.
WINDOW_GROUPS: list[tuple[str, int]] = sorted(
    {(a.entity, a.window_minutes) for a in _AGGREGATES})

Windows = dict[tuple[str, int], list[Trip]]


def specs() -> list[FeatureSpec]:
    return [d.spec for d in _DERIVED] + [a.spec for a in _AGGREGATES]


def entity_key(entity: str, trip: Trip) -> str:
    if entity == "driver":
        return trip.driver_id
    if entity == "store":
        return trip.store_id
    if entity == "trip":
        return trip.trip_id
    raise ValueError(f"no such entity: {entity}")


def is_finished(trip: Trip) -> bool:
    """Whether a trip can contribute to an aggregate at all.

    An unlabelled trip is not a fast trip. Letting one through would teach the
    model that trips whose delivered event went missing were quick.
    """
    return trip.delivered_at is not None and trip.tat_minutes is not None


# --------------------------------------------------- the one computation path

def _derived_values(trip: Trip) -> dict[str, float]:
    return {d.spec.name: float(d.of(trip)) for d in _DERIVED}


def _values(trip: Trip, windows: Windows) -> dict[str, float]:
    """Produce every declared value from history already fetched.

    Private because it does not check time. `compute_vector` is the entry
    point; the only other callers are the deliberately leaky variants, which
    have to route around the check to exist -- and having to is the evidence
    that the check carries weight.
    """
    values = _derived_values(trip)
    for agg in _AGGREGATES:
        finished = windows.get((agg.entity, agg.window_minutes), [])
        values[agg.spec.name] = float(agg.of(finished, trip))
    return values


def compute_vector(trip: Trip, as_of: dt.datetime,
                   windows: Windows) -> FeatureVector:
    """Features for one trip at one instant.

    The leakage check lives here rather than in each caller because a caller
    that forgets it returns a plausible number instead of an error, and the
    only symptom is an evaluation score production never reproduces.
    """
    for (entity, window_minutes), finished in windows.items():
        for other in finished:
            if other.delivered_at is None or other.delivered_at > as_of:
                raise LeakageError(
                    f"{trip.trip_id}: the {entity}/{window_minutes}m window "
                    f"was handed {other.trip_id}, delivered "
                    f"{other.delivered_at}, after as_of {as_of}")
    return FeatureVector(trip_id=trip.trip_id, as_of=as_of,
                         values=_values(trip, windows))


# ---------------------------------------------------------- the offline sweep

def historical(trips: Sequence[Trip]) -> list[FeatureVector]:
    """Point-in-time features for a whole training set.

    One forward sweep. Trips are visited in assignment order while a second
    cursor walks the same trips in *delivery* order, so a trip enters an
    entity's history exactly when it finished and leaves when it drops out of
    the window. The obvious alternative -- filter the whole table once per row
    -- is quadratic, and whoever hits that wall is tempted to widen the filter
    until it is fast, which is how a window quietly starts including the
    future.
    """
    order = sorted(trips, key=lambda t: (t.assigned_at, t.trip_id))
    finished = sorted((t for t in trips if is_finished(t)),
                      key=lambda t: (t.delivered_at, t.trip_id))

    buckets: dict[tuple[str, int], dict[str, deque[Trip]]] = {
        group: {} for group in WINDOW_GROUPS}
    cursor = 0
    vectors: list[FeatureVector] = []

    for trip in order:
        as_of = trip.assigned_at

        while cursor < len(finished) and finished[cursor].delivered_at <= as_of:
            done = finished[cursor]
            cursor += 1
            for (entity, _window), by_key in buckets.items():
                by_key.setdefault(entity_key(entity, done),
                                  deque()).append(done)

        windows: Windows = {}
        for (entity, window_minutes), by_key in buckets.items():
            queue = by_key.get(entity_key(entity, trip))
            if not queue:
                windows[(entity, window_minutes)] = []
                continue
            cutoff = as_of - dt.timedelta(minutes=window_minutes)
            while queue and queue[0].delivered_at <= cutoff:
                queue.popleft()
            windows[(entity, window_minutes)] = list(queue)

        vectors.append(compute_vector(trip, as_of, windows))

    return vectors


def _by_entity(trips: Sequence[Trip],
               sort_key: Callable[[Trip], dt.datetime]
               ) -> dict[tuple[str, str], list[Trip]]:
    by_entity: dict[tuple[str, str], list[Trip]] = {}
    for trip in trips:
        if not is_finished(trip):
            continue
        for entity, _window in WINDOW_GROUPS:
            by_entity.setdefault((entity, entity_key(entity, trip)),
                                 []).append(trip)
    for bucket in by_entity.values():
        bucket.sort(key=sort_key)
    return by_entity


def leaky_in_flight_historical(trips: Sequence[Trip]) -> list[FeatureVector]:
    """The same features, computed wrongly, on purpose: the window filtered on
    `assigned_at`.

    This is the bug as it is actually written. The window looks right -- two
    hours, keyed on the driver, nothing about the future in the SQL -- and the
    filter is on the wrong column. Every trip that was *assigned* in the last
    two hours counts, including the ones still on the road, whose durations
    were not facts yet. Including, since `assigned_at <= T` is true of it too,
    the trip being predicted: its own duration walks into its own feature, and
    the model learns to read the label.

    It is kept here, beside the correct sweep, because the gap between the two
    offline scores is the only honest way to say what point-in-time
    correctness is worth. Nothing but the training comparison may call it, and
    it has to bypass `compute_vector` to exist at all.
    """
    by_entity = _by_entity(trips, lambda t: t.assigned_at)

    vectors: list[FeatureVector] = []
    for trip in sorted(trips, key=lambda t: (t.assigned_at, t.trip_id)):
        as_of = trip.assigned_at
        windows: Windows = {}
        for entity, window_minutes in WINDOW_GROUPS:
            bucket = by_entity.get((entity, entity_key(entity, trip)), [])
            cutoff = as_of - dt.timedelta(minutes=window_minutes)
            lo = bisect.bisect_right(bucket, cutoff,
                                     key=lambda t: t.assigned_at)
            hi = bisect.bisect_right(bucket, as_of,
                                     key=lambda t: t.assigned_at)
            windows[(entity, window_minutes)] = bucket[lo:hi]
        vectors.append(FeatureVector(trip_id=trip.trip_id, as_of=as_of,
                                     values=_values(trip, windows)))
    return vectors


def leaky_lifetime_historical(trips: Sequence[Trip]) -> list[FeatureVector]:
    """Wrong a second way: no window at all.

    Every aggregate reads the entity's entire history, past and future -- the
    GROUP BY someone writes when the window and the direction of time both
    slip their mind. The leak is real but diluted across a driver's whole
    record, so it flatters the model far less than the in-flight version does.
    That contrast is worth printing: the size of a leak is not the size of the
    mistake that caused it, and a gap small enough to dismiss as noise can
    come from a query that is wrong in principle.
    """
    by_entity = _by_entity(trips, lambda t: t.delivered_at)
    # A window with no edges is the same list of trips for every trip of the
    # same driver, so each reducer is called once per driver instead of once
    # per trip. The reducers are still the only arithmetic; what is cached is
    # the answer, not a second copy of the definition. Without this the
    # variant is quadratic, and the cost of the bug is not what is being
    # demonstrated here.
    cached: dict[tuple[str, str], float] = {}

    vectors: list[FeatureVector] = []
    for trip in sorted(trips, key=lambda t: (t.assigned_at, t.trip_id)):
        values = _derived_values(trip)
        for agg in _AGGREGATES:
            key = entity_key(agg.entity, trip)
            history = by_entity.get((agg.entity, key), [])
            if not history:
                values[agg.spec.name] = float(agg.of([], trip))
                continue
            seen = cached.get((agg.spec.name, key))
            if seen is None:
                seen = float(agg.of(history, trip))
                cached[(agg.spec.name, key)] = seen
            values[agg.spec.name] = seen
        vectors.append(FeatureVector(trip_id=trip.trip_id,
                                     as_of=trip.assigned_at, values=values))
    return vectors


# ------------------------------------------------------------ online lookup

class OnlineStore:
    """Recent finished trips, held in memory for the serving path.

    The serving layer pushes a trip the moment it is delivered and asks for
    features the moment the next one is assigned. Retention has to cover the
    widest declared window with room to spare: a store that has already
    forgotten trips the offline sweep would have counted is answering a
    quietly different question from the one the model was trained on, and the
    parity test is the only thing between that and production.
    """

    def __init__(self, retain_minutes: int = 24 * 60) -> None:
        widest = max((w for _entity, w in WINDOW_GROUPS), default=0)
        if retain_minutes < widest:
            raise ValueError(f"retention of {retain_minutes}m cannot answer a "
                             f"{widest}m window")
        self.retain_minutes = retain_minutes
        self._by_entity: dict[tuple[str, str], list[Trip]] = {}
        self._newest: dt.datetime | None = None

    def push(self, trip: Trip) -> None:
        """Take one finished trip into the history."""
        if not is_finished(trip):
            raise ValueError(f"{trip.trip_id} has not finished: an in-flight "
                             f"trip has no duration to aggregate")
        for entity, _window in WINDOW_GROUPS:
            bucket = self._by_entity.setdefault(
                (entity, entity_key(entity, trip)), [])
            bisect.insort(bucket, trip,
                          key=lambda t: (t.delivered_at, t.trip_id))
        if self._newest is None or trip.delivered_at > self._newest:
            self._newest = trip.delivered_at
        self._evict()

    def extend(self, trips: Iterable[Trip]) -> None:
        for trip in trips:
            self.push(trip)

    def window(self, entity: str, key: str, window_minutes: int,
               as_of: dt.datetime) -> list[Trip]:
        """Finished trips in `(as_of - window, as_of]` for one entity.

        The upper bound is what keeps this the same question training asked. A
        store that has been handed a trip delivered after `as_of` -- a replay,
        a backfill, a clock that ran ahead -- must not pass it on.
        """
        bucket = self._by_entity.get((entity, key), [])
        cutoff = as_of - dt.timedelta(minutes=window_minutes)
        lo = bisect.bisect_right(bucket, cutoff, key=lambda t: t.delivered_at)
        hi = bisect.bisect_right(bucket, as_of, key=lambda t: t.delivered_at)
        return bucket[lo:hi]

    def windows_for(self, trip: Trip, as_of: dt.datetime) -> Windows:
        return {
            (entity, window_minutes): self.window(
                entity, entity_key(entity, trip), window_minutes, as_of)
            for entity, window_minutes in WINDOW_GROUPS
        }

    def _evict(self) -> None:
        """Forget history no declared window can reach any more."""
        if self._newest is None:
            return
        horizon = self._newest - dt.timedelta(minutes=self.retain_minutes)
        for bucket in self._by_entity.values():
            drop = bisect.bisect_right(bucket, horizon,
                                       key=lambda t: t.delivered_at)
            if drop:
                del bucket[:drop]

    def trips_held(self) -> int:
        """Distinct finished trips in the store.

        Counted through one entity's buckets because every trip is filed under
        each entity it has a key for, and summing all of them would report the
        same trip several times.
        """
        if not WINDOW_GROUPS:
            return 0
        entity = WINDOW_GROUPS[0][0]
        return sum(len(bucket) for (kind, _key), bucket
                   in self._by_entity.items() if kind == entity)

    def __len__(self) -> int:
        return self.trips_held()


class TripFeatureStore:
    """The `FeatureStore` both retrieval paths go through.

    `historical` and `online` share their declarations and their arithmetic;
    all they disagree about is where the recent trips came from. If this class
    ever grows a second copy of an aggregate, the parity test is what will say
    so, and it will say so by failing.
    """

    def __init__(self, online_store: OnlineStore | None = None) -> None:
        self.online_store = (online_store if online_store is not None
                             else OnlineStore())

    def specs(self) -> list[FeatureSpec]:
        return specs()

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def historical(self, trips: Sequence[Trip]) -> list[FeatureVector]:
        return historical(trips)

    def online(self, trip: Trip, as_of: dt.datetime) -> FeatureVector:
        return compute_vector(trip, as_of,
                              self.online_store.windows_for(trip, as_of))

    def observe(self, trip: Trip) -> None:
        """Take a finished trip back from the serving layer."""
        self.online_store.push(trip)
