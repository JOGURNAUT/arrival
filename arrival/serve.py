"""Serving one prediction at a time, and admitting it when that fails.

Training is allowed to be slow and allowed to crash. Serving is allowed to be
neither, and the difference changes what the code has to look like:

  * A request must come back with a number. If the feature store cannot produce
    a vector -- a driver on their first shift, an entity the online store has
    never seen, a store that was down -- the answer is the flat promise, not an
    exception. But a fallback that is not labelled as one is worse than an
    exception, because the dashboard keeps reporting a model that is no longer
    being consulted. So every fallback is counted and carried on the response.

  * Latency is a distribution, never a number. See `LatencySummary`.

  * The service must be able to say which model it is running. After an
    incident the first question is "what was deployed", and a service that
    cannot answer it turns a ten-minute investigation into an afternoon of
    reading deploy logs.

The logic lives in `PredictionService`, which is plain Python and has no web
framework in it. `build_app` is a thin wrapper that exposes the same three
operations over HTTP when FastAPI happens to be installed. That split is not
tidiness: it means the serving behaviour is tested by the test suite that runs
from a clone with nothing but pytest.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

from .contracts import (ArrivalError, FeatureStore, Model, Prediction,
                        Trip)

# Methods an online store might offer for taking a completed trip back. The
# FeatureStore Protocol does not require any of them, because writing is not
# part of the retrieval contract, so the service asks rather than assumes.
# ------------------------------------------------------------- percentiles

def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile by nearest rank, where q is 0-100.

    Nearest rank rather than interpolation: interpolating between the two
    slowest requests reports a latency that no request actually had, and the
    number this feeds is meant to be defensible when somebody asks which
    request was that slow.
    """
    if not values:
        raise ArrivalError("no observations to take a percentile of")
    if not 0 <= q <= 100:
        raise ArrivalError(f"percentile {q} is outside 0-100")

    ordered = sorted(values)
    rank = math.ceil(q / 100 * len(ordered))
    return float(ordered[max(0, min(len(ordered) - 1, rank - 1))])


@dataclass(frozen=True)
class LatencySummary:
    """Latency as the shape it actually has.

    The mean is the wrong number for a service. A handler that answers in 10ms
    for ninety-five requests out of a hundred and in 2s for the other five has
    a mean near 110ms -- a figure no single user ever experienced, sitting
    comfortably under any plausible alarm, while one request in twenty is slow
    enough that the caller upstream has already given up. The mean is pulled
    around by the tail and then reports neither it nor the typical case.

    p50 is what a request usually costs. p99 is what the angriest one per
    hundred costs, and it is the one that shows up as a timeout somewhere else.
    `n` sits beside them because a p99 over eleven requests is not a p99.
    """

    n: int
    p50_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float

    @classmethod
    def of(cls, samples: Sequence[float]) -> "LatencySummary":
        if not samples:
            return cls(n=0, p50_ms=0.0, p99_ms=0.0, max_ms=0.0, mean_ms=0.0)
        return cls(
            n=len(samples),
            p50_ms=percentile(samples, 50),
            p99_ms=percentile(samples, 99),
            max_ms=max(samples),
            mean_ms=sum(samples) / len(samples),
        )

    def as_row(self) -> dict[str, float]:
        return {"n": self.n, "p50_ms": self.p50_ms, "p99_ms": self.p99_ms,
                "max_ms": self.max_ms, "mean_ms": self.mean_ms}


@dataclass(frozen=True)
class ServiceHealth:
    """What a health endpoint has to say to be worth calling.

    `ok` alone answers nothing. The model version is what an incident starts
    from, and the fallback rate is the one number that distinguishes a service
    that is up from a service that is up and no longer using its model.
    """

    ok: bool
    model_version: str
    feature_names: list[str]
    requests: int
    fallbacks: int
    fallback_rate: float
    latency: LatencySummary

    def as_row(self) -> dict[str, object]:
        return {"ok": self.ok, "model_version": self.model_version,
                "features": len(self.feature_names),
                "requests": self.requests, "fallbacks": self.fallbacks,
                "fallback_rate": self.fallback_rate, **self.latency.as_row()}


# ------------------------------------------------------------- the service

class PredictionService:
    """A model, a feature store, and the discipline between them.

    Takes the `Model` and `FeatureStore` Protocols, never concrete classes, so
    the baseline and the trained model are swapped without this file changing
    and without it being able to tell which it has.
    """

    def __init__(self, model: Model, store: FeatureStore, *,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.model = model
        self.store = store
        self._clock = clock            # injectable, so latency is testable
        self._latencies: list[float] = []
        self._requests = 0
        self._fallbacks = 0
        self._unobserved: list[Trip] = []

    # -- prediction ---------------------------------------------------------

    def predict(self, trip: Trip, as_of: datetime | None = None, *,
                variant: str = "control") -> Prediction:
        """An ETA for one trip, as of one instant.

        `as_of` defaults to the trip's own `assigned_at` rather than to the
        wall clock. Defaulting to now would mean a replay of last week's trips
        quietly retrieved this morning's features, and the resulting numbers
        would look like a successful backtest.
        """
        moment = as_of or trip.assigned_at
        started = self._clock()
        try:
            minutes, fallback, reason = self._estimate(trip, moment)
        finally:
            # Recorded in the `finally` because a request that blew up slowly
            # is exactly the one the latency graph needs to contain.
            self._latencies.append((self._clock() - started) * 1000.0)

        self._requests += 1
        if fallback:
            self._fallbacks += 1

        return Prediction(
            trip_id=trip.trip_id,
            minutes=minutes,
            model_version=self.model.version,
            variant=variant,
            served_at=moment,
            fallback=fallback,
            fallback_reason=reason,
        )

    def _estimate(self, trip: Trip,
                  as_of: datetime) -> tuple[float, bool, str]:
        """The model's number, or the promise and the reason why.

        Every failure between here and a float is caught, including the ones
        not foreseen, because the alternative is a 500 for a cold-start driver.
        Breadth is the point: a feature store raises `KeyError` for a missing
        feature, `ArrivalError` for a missing entity, and whatever its storage
        layer raises when storage is unhappy, and none of those should reach
        the caller.
        """
        try:
            vector = self.store.online(trip, as_of)
        except Exception as exc:                       # noqa: BLE001
            return float(trip.promised_minutes), True, _why("features", exc)

        try:
            minutes = float(self.model.predict(vector))
        except Exception as exc:                       # noqa: BLE001
            return float(trip.promised_minutes), True, _why("model", exc)

        # A NaN propagates silently through averages and comparisons and turns
        # a dashboard blank hours later; a negative ETA is arithmetic, not an
        # estimate. Both are failures of the model, so both fall back.
        if not math.isfinite(minutes) or minutes <= 0:
            return (float(trip.promised_minutes), True,
                    f"model returned {minutes}")

        return minutes, False, ""

    # -- the completion loop ------------------------------------------------

    def observe(self, trip: Trip) -> None:
        """Push a finished trip back so the next prediction can see it.

        Without this the online store is frozen at deploy time: a driver's
        recent-trip average stops moving, and the serving features drift away
        from the ones training was built on while every other signal stays
        green.

        An unlabelled trip is refused. Feeding a trip whose duration is not yet
        known into an aggregate over durations does not add information, it
        dilutes the aggregate with a guess.
        """
        if not trip.is_labelled:
            raise ArrivalError(
                f"{trip.trip_id} has no duration yet; nothing to observe")

        sink = getattr(self.store, "observe", None)
        if callable(sink):
            sink(trip)
            return

        # A store with no write path is a real deployment state -- a read
        # replica, a store still being backfilled -- not an error. The trips are
        # kept and counted, because dropping them silently is what makes a stale
        # online store look like a healthy one.
        self._unobserved.append(trip)

    @property
    def unobserved(self) -> list[Trip]:
        """Completed trips the online store could not be told about."""
        return list(self._unobserved)

    # -- introspection ------------------------------------------------------

    def health(self) -> ServiceHealth:
        """The `/health` answer, including which model is loaded."""
        return ServiceHealth(
            ok=True,
            model_version=self.model.version,
            feature_names=list(getattr(self.model, "feature_names", [])),
            requests=self._requests,
            fallbacks=self._fallbacks,
            fallback_rate=self.fallback_rate,
            latency=self.latency(),
        )

    def latency(self) -> LatencySummary:
        return LatencySummary.of(self._latencies)

    @property
    def fallback_rate(self) -> float:
        if not self._requests:
            return 0.0
        return self._fallbacks / self._requests

    @property
    def served_predictions(self) -> int:
        return self._requests


def _why(stage: str, exc: Exception) -> str:
    """A reason short enough for a response field and specific enough to grep.

    The exception type is kept because "features: KeyError" and
    "features: TimeoutError" are a schema problem and an outage respectively,
    and a fallback rate that does not separate them cannot be acted on.
    """
    detail = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{stage}: {type(exc).__name__}" + (f": {detail}" if detail else "")


# ------------------------------------------------------------ the HTTP skin

def trip_from_payload(payload: dict) -> Trip:
    """Build a `Trip` out of a request body.

    A plain function rather than a request model so that the parsing -- which
    is where a serving layer usually breaks first, on a field that arrives as a
    string -- is tested without a web framework installed.
    """
    try:
        return Trip(
            trip_id=str(payload["trip_id"]),
            store_id=str(payload["store_id"]),
            driver_id=str(payload["driver_id"]),
            assigned_at=_as_datetime(payload["assigned_at"]),
            distance_m=int(payload["distance_m"]),
            promised_minutes=int(payload["promised_minutes"]),
        )
    except KeyError as exc:
        raise ArrivalError(f"request is missing {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ArrivalError(f"request field is unreadable: {exc}") from exc


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", ""))


def build_app(service: PredictionService):
    """Wrap a service in FastAPI, if FastAPI is here.

    Imported inside the function so that this module -- and therefore the
    serving logic and its tests -- stays importable in an environment with
    nothing but pytest, which is the environment the repository promises to run
    in.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:                         # pragma: no cover
        raise ArrivalError(
            "fastapi is not installed; PredictionService works without it"
        ) from exc

    app = FastAPI(title="arrival", version=service.model.version)

    @app.get("/health")
    def health() -> dict:
        return service.health().as_row()

    @app.post("/predict")
    def predict(payload: dict) -> dict:
        try:
            trip = trip_from_payload(payload)
        except ArrivalError as exc:
            # A malformed request is the caller's fault and gets a 400. The
            # fallback path is for when *our* side cannot answer; using it here
            # would bury client bugs in our fallback rate.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        p = service.predict(trip, variant=str(payload.get("variant",
                                                          "control")))
        return {"trip_id": p.trip_id, "minutes": p.minutes,
                "model_version": p.model_version, "variant": p.variant,
                "fallback": p.fallback, "fallback_reason": p.fallback_reason}

    @app.post("/observe")
    def observe(payload: dict) -> dict:
        trip = trip_from_payload(payload)
        tat = payload.get("tat_minutes")
        if tat is None:
            raise HTTPException(status_code=400,
                                detail="observe needs tat_minutes")
        delivered = payload.get("delivered_at")
        service.observe(Trip(
            trip_id=trip.trip_id, store_id=trip.store_id,
            driver_id=trip.driver_id, assigned_at=trip.assigned_at,
            distance_m=trip.distance_m,
            promised_minutes=trip.promised_minutes,
            delivered_at=_as_datetime(delivered) if delivered else None,
            tat_minutes=float(tat),
        ))
        return {"observed": trip.trip_id}

    return app
