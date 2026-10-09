"""What the serving layer must never do, written as failures.

Every docstring here says what reaches production if the test goes red. The
stubs are deliberately crude: they implement the `Model` and `FeatureStore`
Protocols and nothing else, so these tests pass with the feature package
half-written or absent, which is the state it is in while this is being
reviewed.
"""

from __future__ import annotations

import datetime as dt
import importlib
import sys

import pytest

from arrival.contracts import ArrivalError, FeatureSpec, FeatureVector, Trip
from arrival.serve import (LatencySummary, PredictionService, build_app,
                           percentile, trip_from_payload)

NOON = dt.datetime(2026, 9, 1, 12, 0, 0)


# ------------------------------------------------------------------- stubs

class StubModel:
    """Minutes from a single feature, so the number is predictable."""

    feature_names = ["distance_m"]
    version = "stub-1.4.0"

    def __init__(self, answer: float | None = None) -> None:
        self.answer = answer

    def predict(self, vector: FeatureVector) -> float:
        if self.answer is not None:
            return self.answer
        return 5.0 + vector.ordered(self.feature_names)[0] / 1000.0


class StubStore:
    """An online store that always answers, and remembers what it is told."""

    def __init__(self) -> None:
        self.observed: list[Trip] = []

    def specs(self) -> list[FeatureSpec]:
        return [FeatureSpec("distance_m", "trip", "metres to the drop")]

    def historical(self, trips):
        return [self.online(t, t.assigned_at) for t in trips]

    def online(self, trip: Trip, as_of: dt.datetime) -> FeatureVector:
        return FeatureVector(trip_id=trip.trip_id, as_of=as_of,
                             values={"distance_m": float(trip.distance_m)})

    def observe(self, trip: Trip) -> None:
        self.observed.append(trip)


class ColdStore(StubStore):
    """An entity the store has never seen -- a driver's first shift."""

    def online(self, trip: Trip, as_of: dt.datetime) -> FeatureVector:
        raise KeyError(f"no online features for {trip.driver_id}")


class HalfColdStore(StubStore):
    """Cold for some drivers, warm for others."""

    def online(self, trip: Trip, as_of: dt.datetime) -> FeatureVector:
        if trip.driver_id.endswith("9"):
            raise ArrivalError(f"entity {trip.driver_id} unknown")
        return super().online(trip, as_of)


class WriteOnlyNowhereStore(StubStore):
    """A retrieval-only store: conforms to the Protocol, takes no writes."""

    observe = None          # type: ignore[assignment]


class ScriptedClock:
    """A clock that makes each request take exactly as long as the script says.

    Latency tested against a real clock either asserts nothing or asserts that
    the machine running CI is fast, and the second one fails on a Friday for
    reasons nobody can reproduce.
    """

    def __init__(self, durations_ms) -> None:
        self._durations = iter(durations_ms)
        self._now = 0.0
        self._inside = False

    def __call__(self) -> float:
        if not self._inside:
            self._inside = True
            return self._now
        self._inside = False
        self._now += next(self._durations) / 1000.0
        return self._now


def a_trip(i: int = 1, driver: str = "D-0001", distance: int = 7000) -> Trip:
    return Trip(trip_id=f"T-{i:06d}", store_id="NORTHGATE", driver_id=driver,
                assigned_at=NOON + dt.timedelta(minutes=i),
                distance_m=distance,
                promised_minutes=19)


# ------------------------------------------------------------ percentiles

def test_percentiles_land_on_the_ranks_they_claim():
    """If percentiles are off by a rank, every latency SLO in the service is
    measuring a different number than the one it is named after, and the
    discrepancy is invisible because the figure still looks plausible."""
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 90) == 90
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100
    assert percentile(values, 0) == 1


def test_a_percentile_reports_a_latency_some_request_actually_had():
    """Interpolated percentiles invent values. When the p99 alarm fires at
    03:00 the first question is which request was that slow, and a number
    halfway between two requests has no answer to it."""
    observed = [10.0, 10.0, 10.0, 2000.0]
    assert percentile(observed, 99) in observed


def test_latency_percentiles_are_computed_correctly_including_the_tail():
    """The tail is the whole reason this is not a mean. A service where one
    request in twenty takes two seconds has already timed out somebody
    upstream, while its mean sits at a tenth of a second and under every
    alarm -- so an alarm on the mean never fires."""
    fast = [10.0] * 950
    slow = [2000.0] * 50
    summary = LatencySummary.of(fast + slow)

    assert summary.n == 1000
    assert summary.p50_ms == 10.0          # what a request usually costs
    assert summary.p99_ms == 2000.0        # what one in a hundred costs
    assert summary.mean_ms == pytest.approx(109.5)

    # The mean is between the two and describes neither.
    assert summary.p50_ms < summary.mean_ms < summary.p99_ms


def test_the_service_measures_the_latency_of_each_request_it_serves():
    """Percentiles over the wrong population are worse than none. If the
    service only timed successful requests, the slow failures -- the ones that
    cause the incident -- would be the exact population excluded from the
    graph."""
    clock = ScriptedClock([10.0] * 95 + [1800.0] * 5)
    service = PredictionService(StubModel(), HalfColdStore(), clock=clock)

    for i in range(100):
        # Every twentieth trip has a cold driver and falls back, and its
        # latency must still be recorded.
        driver = "D-0009" if i % 20 == 0 else "D-0001"
        service.predict(a_trip(i, driver=driver))

    latency = service.latency()
    assert latency.n == 100
    assert latency.p50_ms == pytest.approx(10.0)
    assert latency.p99_ms == pytest.approx(1800.0)


def test_latency_with_no_traffic_reports_n_of_zero_rather_than_a_fake_p99():
    """A freshly started service reporting p99 of 0ms looks like the fastest
    service in the fleet, and the dashboard stays wrong until enough traffic
    arrives to drown the lie."""
    summary = PredictionService(StubModel(), StubStore()).latency()
    assert summary.n == 0
    assert summary.p99_ms == 0.0

    with pytest.raises(ArrivalError):
        percentile([], 99)


# -------------------------------------------------------------- prediction

def test_a_prediction_comes_back_with_the_model_version_that_made_it():
    """A response that does not name its model cannot be reconciled with a
    deploy timeline afterwards, so a bad rollout cannot be dated."""
    service = PredictionService(StubModel(), StubStore())
    p = service.predict(a_trip(1, distance=9000))

    assert p.minutes == pytest.approx(14.0)
    assert p.model_version == "stub-1.4.0"
    assert p.fallback is False


def test_as_of_defaults_to_the_assignment_time_not_the_wall_clock():
    """Defaulting to now means a replay of last month's trips retrieves this
    morning's features. The backtest it produces looks excellent and describes
    nothing that could ever happen in production."""
    trip = a_trip(5)
    service = PredictionService(StubModel(), StubStore())

    assert service.predict(trip).served_at == trip.assigned_at
    assert service.predict(trip, NOON).served_at == NOON


# ---------------------------------------------------------------- fallback

def test_the_fallback_path_returns_an_answer_and_is_flagged():
    """A cold-start driver must not become a 500 -- the caller needs a number
    to show the customer. But an unflagged fallback is the worse failure: the
    service reports healthy while serving the flat promise to everyone, and
    the model it is supposedly running has stopped being consulted."""
    trip = a_trip(1)
    service = PredictionService(StubModel(), ColdStore())
    p = service.predict(trip)

    assert p.minutes == float(trip.promised_minutes)
    assert p.fallback is True
    assert "KeyError" in p.fallback_reason
    assert p.model_version == "stub-1.4.0"


def test_the_fallback_rate_is_measurable_because_fallbacks_are_counted():
    """Without a rate, a store that goes cold for a tenth of traffic is
    indistinguishable from a working service: latency improves, errors stay at
    zero, and nothing anywhere says the model stopped mattering."""
    service = PredictionService(StubModel(), HalfColdStore())
    for i in range(100):
        cold = i % 10 == 0
        service.predict(a_trip(i, driver="D-0009" if cold else "D-0001"))

    assert service.health().fallbacks == 10
    assert service.fallback_rate == pytest.approx(0.10)


def test_a_model_that_fails_on_a_vector_falls_back_rather_than_raising():
    """Model failure and feature failure are different incidents, and both
    arrive as a stack trace in the same handler. Only one of them is fixed by
    restarting the store, so the reason is carried on the response."""
    class BrokenModel(StubModel):
        def predict(self, vector):
            raise ValueError("feature order changed under us")

    p = PredictionService(BrokenModel(), StubStore()).predict(a_trip(1))
    assert p.fallback is True
    assert p.fallback_reason.startswith("model: ValueError")


@pytest.mark.parametrize("answer", [float("nan"), float("inf"), 0.0, -4.0])
def test_a_nonsense_number_falls_back_instead_of_being_served(answer):
    """A NaN ETA propagates through every average downstream and blanks a
    dashboard hours later, far from the model that produced it. A negative ETA
    is arithmetic, not an estimate. Both are model failures and neither should
    be handed to a customer."""
    p = PredictionService(StubModel(answer), StubStore()).predict(a_trip(1))
    assert p.fallback is True
    assert p.minutes == 19.0


# ----------------------------------------------------------------- health

def test_health_reports_the_loaded_model_version():
    """After an incident the first question is which model was serving. A
    service that cannot answer it turns a ten-minute investigation into an
    afternoon of reading deploy logs."""
    health = PredictionService(StubModel(), StubStore()).health()
    assert health.ok is True
    assert health.model_version == "stub-1.4.0"
    assert health.feature_names == ["distance_m"]


def test_health_carries_the_fallback_rate_beside_the_version():
    """A health check that only says `ok` is true of a service serving the
    flat promise to every caller, which is the failure that most needs to be
    visible from outside."""
    service = PredictionService(StubModel(), ColdStore())
    service.predict(a_trip(1))
    row = service.health().as_row()

    assert row["model_version"] == "stub-1.4.0"
    assert row["fallback_rate"] == pytest.approx(1.0)
    assert row["requests"] == 1


# ---------------------------------------------------------------- observe

def test_observing_a_finished_trip_pushes_it_back_into_the_online_store():
    """Without the write-back the online store is frozen at deploy time. A
    driver's recent-trip average stops moving, the serving features drift away
    from the ones the model was fitted on, and every other signal stays
    green."""
    store = StubStore()
    service = PredictionService(StubModel(), store)
    finished = Trip(trip_id="T-9", store_id="NORTHGATE", driver_id="D-0001",
                    assigned_at=NOON, distance_m=7000, promised_minutes=19,
                    delivered_at=NOON + dt.timedelta(minutes=31),
                    tat_minutes=31.0)

    service.observe(finished)
    assert store.observed == [finished]


def test_observing_a_trip_with_no_duration_is_refused():
    """Feeding an unfinished trip into an aggregate over durations adds no
    information and dilutes the aggregate with a guess, which then shows up as
    a feature that disagrees with the one training computed."""
    service = PredictionService(StubModel(), StubStore())
    with pytest.raises(ArrivalError, match="no duration"):
        service.observe(a_trip(1))


def test_trips_a_store_cannot_accept_are_kept_rather_than_dropped():
    """A store with no write path is a real deployment state. Dropping the
    completions silently makes a stale online store look like a healthy one,
    and the staleness is only discovered through the model getting worse."""
    service = PredictionService(StubModel(), WriteOnlyNowhereStore())
    finished = Trip(trip_id="T-9", store_id="NORTHGATE", driver_id="D-0001",
                    assigned_at=NOON, distance_m=7000, promised_minutes=19,
                    delivered_at=NOON + dt.timedelta(minutes=31),
                    tat_minutes=31.0)

    service.observe(finished)
    assert [t.trip_id for t in service.unobserved] == ["T-9"]


# ------------------------------------------------------------ the HTTP skin

def test_a_malformed_request_is_rejected_before_it_reaches_the_model():
    """A missing field that reaches the model becomes a fallback, and client
    bugs then accumulate in our fallback rate -- hiding the store failures the
    rate exists to surface."""
    body = {"trip_id": "T-1", "store_id": "NORTHGATE", "driver_id": "D-1",
            "assigned_at": "2026-09-01T12:00:00", "distance_m": 7000,
            "promised_minutes": 19}
    assert trip_from_payload(body).distance_m == 7000

    with pytest.raises(ArrivalError, match="missing distance_m"):
        trip_from_payload({k: v for k, v in body.items() if k != "distance_m"})

    with pytest.raises(ArrivalError, match="unreadable"):
        trip_from_payload({**body, "distance_m": "roughly seven kilometres"})


def test_the_service_works_with_no_web_framework_installed():
    """The repository promises to run from a clone with nothing but pytest. If
    the serving logic acquires an import of FastAPI at module scope, the test
    suite silently starts requiring a web server to check arithmetic, and the
    promise is broken for everybody who clones it next."""
    class Blocked:
        """An import hook that behaves like the package simply is not there."""

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"fastapi", "uvicorn", "starlette",
                                      "pydantic"}:
                raise ModuleNotFoundError(f"No module named {name!r}",
                                          name=name)
            return None

    saved = {k: v for k, v in sys.modules.items()
             if k.startswith(("arrival.serve", "fastapi", "pydantic",
                              "starlette", "uvicorn"))}
    sys.meta_path.insert(0, Blocked())
    try:
        for name in list(saved):
            del sys.modules[name]
        serve = importlib.import_module("arrival.serve")

        service = serve.PredictionService(StubModel(), StubStore())
        assert service.predict(a_trip(1)).minutes == pytest.approx(12.0)
        assert service.health().model_version == "stub-1.4.0"

        with pytest.raises(ArrivalError, match="fastapi is not installed"):
            serve.build_app(service)
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved)


def test_the_http_layer_is_a_wrapper_and_nothing_more():
    """Logic that leaks into the route handlers is logic the offline tests
    cannot reach, so it is the part that breaks in production."""
    fastapi = pytest.importorskip("fastapi")
    app = build_app(PredictionService(StubModel(), StubStore()))

    assert isinstance(app, fastapi.FastAPI)
    routes = {r.path for r in app.routes}
    assert {"/health", "/predict", "/observe"} <= routes
