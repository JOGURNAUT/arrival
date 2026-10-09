"""Where trips come from: Dispatch's warehouse, or a seeded stand-in.

Arrival sits on top of Dispatch rather than beside it. Dispatch is the pipeline
that lands trip telemetry and measures how wrong a flat delivery promise is;
this predicts the number that promise should have been. Its gold table
`fct_trip` is already the training table -- one row per trip, with the time the
trip was assigned, the distance, the promise that was made, and what actually
happened.

Two properties of that table are what make this project possible at all:

  * `assigned_at` is the prediction time. A model would have been asked at that
    instant and could not have known anything after it.
  * `tat_minutes` is only known once the trip ends, typically half an hour
    later. The label arrives late, which is a fact about the world and the
    reason the monitoring layer cannot simply compare predictions to truth.

When Dispatch's warehouse is not present, a seeded generator produces trips with
the same shape, so this repository runs from a clone on its own. The generator
is not a convenience: the structure it puts in -- stores with different pack
times, drivers with persistent speed, distance mattering -- is what any model
here is supposed to recover, and a test asserts it is there.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import random
import sqlite3

from .contracts import ArrivalError, Trip

DISPATCH_WAREHOUSE = pathlib.Path.home() / "dispatch" / "data" / "warehouse.db"

# Minutes of handling at the store before a rider can leave, by store. Real and
# structural, the same way kitchen time is: a bigger store packs slower. The
# promise Dispatch measures is flat across both, which is what makes a
# per-trip estimate worth having.
STORE_PACK_MINUTES = {"NORTHGATE": 11.0, "RIVERSIDE": 6.0}

# Metres a rider covers per minute, on average. Divides distance into time.
BASE_SPEED_M_PER_MIN = 220.0


def from_dispatch(path: pathlib.Path | str | None = None) -> list[Trip]:
    """Read completed trips out of Dispatch's gold layer.

    Only `completeness = 'complete'` rows carry a trustworthy duration -- a trip
    whose stream is missing its delivered event has no label, and including it
    would teach the model that those trips were fast.
    """
    db = pathlib.Path(path or DISPATCH_WAREHOUSE)
    if not db.exists():
        raise ArrivalError(f"no Dispatch warehouse at {db}")

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("""
            SELECT trip_id, store_id, driver_id, assigned_at, distance_m,
                   promised_minutes, delivered_at, tat_minutes
            FROM fct_trip
            WHERE completeness = 'complete'
              AND tat_minutes IS NOT NULL
              AND distance_m  IS NOT NULL
            ORDER BY assigned_at
        """).fetchall()
    finally:
        conn.close()

    return [
        Trip(trip_id=r[0], store_id=r[1], driver_id=r[2],
             assigned_at=_ts(r[3]), distance_m=int(r[4]),
             promised_minutes=int(r[5]),
             delivered_at=_ts(r[6]) if r[6] else None,
             tat_minutes=float(r[7]))
        for r in rows
    ]


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(str(value).replace("Z", ""))


def generate(n: int = 12_000, seed: int = 11, days: int = 30,
             drivers: int = 60) -> list[Trip]:
    """Trips with the same shape as Dispatch's, seeded.

    Each driver gets a persistent speed multiplier. That is the single most
    important thing in here: it means a driver's past trips genuinely predict
    their next one, so a feature keyed on driver has real signal -- and a
    point-in-time bug that leaks their *future* trips will show up as a model
    that is suspiciously good offline.
    """
    rng = random.Random(seed)
    start = dt.datetime(2026, 9, 1, 8, 0, 0)

    driver_speed = {f"D-{i:04d}": rng.lognormvariate(0.0, 0.18)
                    for i in range(1, drivers + 1)}
    driver_ids = list(driver_speed)
    stores = list(STORE_PACK_MINUTES)

    trips: list[Trip] = []
    for i in range(1, n + 1):
        store = rng.choice(stores)
        driver = rng.choice(driver_ids)
        assigned = start + dt.timedelta(
            minutes=rng.uniform(0, days * 14 * 60))
        distance = int(rng.lognormvariate(7.6, 0.55))

        pack = STORE_PACK_MINUTES[store] * rng.lognormvariate(0.0, 0.2)
        ride = (distance / (BASE_SPEED_M_PER_MIN * driver_speed[driver])) \
            * rng.lognormvariate(0.0, 0.17)
        # Rush hour costs everybody time, and it is knowable at prediction time
        # from assigned_at alone -- which is what makes hour-of-day a fair
        # feature and the driver's next trip an unfair one.
        rush = 1.25 if assigned.hour in (12, 13, 19, 20, 21) else 1.0
        minutes = max(4.0, (pack + ride) * rush)

        trips.append(Trip(
            trip_id=f"T-{i:06d}", store_id=store, driver_id=driver,
            assigned_at=assigned, distance_m=distance,
            promised_minutes=19,          # the flat promise Dispatch measures
            delivered_at=assigned + dt.timedelta(minutes=minutes),
            tat_minutes=round(minutes, 3),
        ))

    trips.sort(key=lambda t: t.assigned_at)
    return trips


def load(path: pathlib.Path | str | None = None, **kwargs) -> list[Trip]:
    """Dispatch's warehouse when it is there, the generator when it is not."""
    try:
        return from_dispatch(path)
    except ArrivalError:
        return generate(**kwargs)
