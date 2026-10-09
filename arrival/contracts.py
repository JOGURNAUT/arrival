"""The boundary every other module codes against.

This file exists so that the feature layer, the serving layer and the monitoring
layer can be written against fixed shapes instead of against each other. Nothing
here does work; it declares what the work must look like.

The one idea worth understanding before reading further:

    A feature has a value only at a moment in time.

"The driver's average trip time" is not a number. It is a number *as of* some
instant, and it is a different number a minute later. Training asks for it as of
when each historical trip was assigned; serving asks for it as of now. If those
two questions are answered by two different pieces of code, they drift, and the
model is scored offline on numbers it will never see in production.

So every retrieval in this project carries an `as_of`, and the online path is
the same definition asked a different question -- never a reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence


# --------------------------------------------------------------- the raw row

@dataclass(frozen=True)
class Trip:
    """One completed trip, as Dispatch's `fct_trip` records it.

    `assigned_at` is the prediction time: the instant a model would have been
    asked for an ETA. Anything that happened after it is unavailable to a
    prediction about it, which is the whole constraint this project is built
    around.

    `tat_minutes` is the label and is only known once `delivered_at` has
    happened -- typically 30-40 minutes after the prediction. That delay is not
    an inconvenience; it is why monitoring has to be built differently.
    """

    trip_id: str
    store_id: str
    driver_id: str
    assigned_at: datetime
    distance_m: int
    promised_minutes: int
    delivered_at: datetime | None = None
    tat_minutes: float | None = None

    @property
    def is_labelled(self) -> bool:
        return self.tat_minutes is not None


# ----------------------------------------------------------------- features

@dataclass(frozen=True)
class FeatureSpec:
    """One feature, declared once.

    `window_minutes` is how far back the aggregate looks. `entity` is what it is
    keyed on, which decides what the online store is asked for.

    A feature declared here but computed somewhere else is the bug this whole
    design exists to prevent, so the computation lives beside the declaration.
    """

    name: str
    entity: str                      # "driver" | "store" | "trip"
    description: str
    window_minutes: int | None = None
    dtype: str = "float"


@dataclass(frozen=True)
class FeatureVector:
    """Features for one entity at one instant.

    `as_of` is carried with the values rather than remembered by the caller,
    because a vector that has lost its timestamp cannot be checked against
    anything and silently becomes whatever the reader assumes it is.
    """

    trip_id: str
    as_of: datetime
    values: dict[str, float] = field(default_factory=dict)

    def ordered(self, names: Sequence[str]) -> list[float]:
        """Values in a declared order -- never dict order.

        A model trained on columns in one order and served them in another
        produces confident nonsense, and no test that checks types will see it.
        """
        missing = [n for n in names if n not in self.values]
        if missing:
            raise KeyError(f"{self.trip_id}: missing features {missing}")
        return [float(self.values[n]) for n in names]


class FeatureStore(Protocol):
    """Two questions, one set of definitions.

    `historical` answers "what was true as of each of these moments", for
    training. `online` answers "what is true now", for serving. A conforming
    implementation must answer both from the same declarations, because the
    parity test asserts they agree -- and that test is the point of the project.
    """

    def specs(self) -> list[FeatureSpec]: ...

    def historical(self, trips: Sequence[Trip]) -> list[FeatureVector]:
        """Point-in-time correct: each trip gets features as of its own
        `assigned_at`, computed only from trips that had already finished."""

    def online(self, trip: Trip, as_of: datetime) -> FeatureVector:
        """The same definitions, asked for one entity at one instant."""


# ------------------------------------------------------------------- models

@dataclass(frozen=True)
class Prediction:
    trip_id: str
    minutes: float
    model_version: str
    variant: str = "control"
    served_at: datetime | None = None


class Model(Protocol):
    """Deliberately small. A model that can be fitted and asked for a number is
    everything the serving layer needs, and keeping the interface this narrow is
    what lets the baseline and the trained model be swapped without the API
    knowing which it has."""

    feature_names: list[str]
    version: str

    def predict(self, vector: FeatureVector) -> float: ...


@dataclass(frozen=True)
class Metrics:
    """What a model scored, and on how much.

    `n` sits beside every figure on purpose. An MAE over 40 trips and one over
    40,000 are different claims, and a card that reports only the number makes
    them look identical.
    """

    n: int
    mae: float
    rmse: float
    p50_error: float
    p90_error: float
    breach_rate: float

    def as_row(self) -> dict[str, float]:
        return {"n": self.n, "mae": self.mae, "rmse": self.rmse,
                "p50_error": self.p50_error, "p90_error": self.p90_error,
                "breach_rate": self.breach_rate}


# ------------------------------------------------------------------- errors

class ArrivalError(RuntimeError):
    """Base for every failure this project raises deliberately."""


class LeakageError(ArrivalError):
    """Raised when a feature would be built from data that did not exist yet.

    This is an exception rather than a warning because the result of ignoring it
    is a model that looks excellent in evaluation and is ordinary in production,
    with nothing in between to say which number was the lie.
    """
