"""Comparing two models in production without fooling yourself.

An A/B test on an ETA model is easy to run and easy to run invalidly. Three
things go wrong, in rising order of how often they are caught:

  1. Assignment by per-request randomness. The same driver gets the control
     model at 09:00 and the treatment at 09:04, so no entity is ever purely in
     one arm, the arms are no longer independent samples of behaviour, and the
     comparison measures nothing. Assignment here is a hash of a stable entity
     id: the same id lands in the same arm forever, with no state to store.

  2. No guardrail. A treatment arm that falls back to the flat promise for a
     third of its requests will often post a respectable average error, because
     the promise is not a terrible estimate. It is still broken. Fallback rate
     is watched separately from accuracy for that reason.

  3. The one that is almost never caught: declaring a winner from an experiment
     that could not have resolved the difference it claims. With a few hundred
     trips per arm and per-trip errors spread over several minutes, the
     smallest effect this design could reliably detect is on the order of a
     minute. A reported 2% improvement -- ten seconds -- is then noise with a
     decimal point on it, and shipping it is a coin flip described as a
     decision. `minimum_detectable_effect` makes that number explicit, and
     `ExperimentResult` carries it beside the effect so the two are read
     together.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .contracts import ArrivalError

CONTROL = "control"
TREATMENT = "treatment"

# The usual pair: a 5% false-positive rate and an 80% chance of seeing an
# effect that is really there. They are stated as defaults rather than buried
# as literals because an MDE quoted without them is not a quantity.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


# ------------------------------------------------------------- assignment

def bucket(entity_id: str, salt: str = "") -> float:
    """A stable number in [0, 1) for an entity.

    SHA-256 rather than the built-in `hash`, which is randomised per process
    for strings: with `hash` the arms would silently reshuffle on every deploy
    and restart, so a driver would cross arms mid-experiment and every
    before-and-after comparison would be measuring the reshuffle.

    The salt separates concurrent experiments. Without it two experiments
    launched the same week share their split exactly, their populations
    correlate, and neither one's result is about its own change.
    """
    digest = hashlib.sha256(f"{salt}:{entity_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign(entity_id: str, split: float = 0.5, salt: str = "") -> str:
    """Which arm an entity belongs to, deterministically and forever.

    `split` is the share going to treatment.
    """
    if not 0.0 <= split <= 1.0:
        raise ArrivalError(f"split {split} is not a share")
    return TREATMENT if bucket(entity_id, salt) < split else CONTROL


# ------------------------------------------------------------- power maths

def _z(p: float) -> float:
    """The standard normal quantile, from the standard library."""
    return statistics.NormalDist().inv_cdf(p)


def minimum_detectable_effect(
    std: float, n_control: int, n_treatment: int, *,
    alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER,
) -> float:
    """The smallest true difference in means this experiment could have found.

    Two-sided test of two independent means:

        MDE = (z[1 - alpha/2] + z[power]) * std * sqrt(1/n_c + 1/n_t)

    Read it the right way round. This is not "the effect we measured" and not
    "the effect we need". It is the resolution of the instrument. An observed
    difference smaller than the MDE is consistent with there being no
    difference at all, however neat the point estimate looks, and the honest
    report of such an experiment is "inconclusive" rather than a percentage.

    The arms shrink the number only as the square root of their size, which is
    why doubling traffic buys so much less certainty than people expect: four
    times the trips to halve the effect you can see.
    """
    if n_control < 2 or n_treatment < 2:
        raise ArrivalError("an arm with fewer than two observations has no "
                           "variance and therefore no detectable effect")
    if std < 0:
        raise ArrivalError("standard deviation cannot be negative")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ArrivalError("alpha and power are probabilities")

    z_sum = _z(1 - alpha / 2) + _z(power)
    return z_sum * std * math.sqrt(1 / n_control + 1 / n_treatment)


def sample_size_per_arm(effect: float, std: float, *,
                        alpha: float = DEFAULT_ALPHA,
                        power: float = DEFAULT_POWER) -> int:
    """Trips per arm needed before an effect of `effect` is detectable.

    The question to ask before launching rather than after: if the change is
    worth 30 seconds a trip, this says how long the experiment has to run, and
    sometimes the answer is longer than the quarter.
    """
    if effect <= 0:
        raise ArrivalError("an effect to detect must be positive")
    if std <= 0:
        return 2
    z_sum = _z(1 - alpha / 2) + _z(power)
    return int(math.ceil(2 * (z_sum ** 2) * (std ** 2) / (effect ** 2)))


# ----------------------------------------------------------------- arms

@dataclass
class Arm:
    """One arm's observations.

    The outcome is per-trip absolute error in minutes -- lower is better -- and
    the fallback flag is carried separately because a fallback is not a kind of
    accuracy, it is the absence of a model.
    """

    name: str
    errors: list[float] = field(default_factory=list)
    fallbacks: int = 0

    def record(self, error: float, fallback: bool = False) -> None:
        self.errors.append(float(error))
        if fallback:
            self.fallbacks += 1

    @property
    def n(self) -> int:
        return len(self.errors)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.errors) if self.errors else float("nan")

    @property
    def std(self) -> float:
        # Sample standard deviation: one arm is a sample of the traffic that
        # arm would have seen, not the whole of it.
        return statistics.stdev(self.errors) if self.n > 1 else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.n if self.n else 0.0


@dataclass(frozen=True)
class Guardrail:
    """The conditions under which the experiment stops regardless of its means.

    An experiment is a change to production with a measurement attached, and
    the measurement takes days. These are the things that must not be allowed
    to run for days.
    """

    max_fallback_rate: float = 0.05
    max_error_ratio: float = 1.10      # treatment mean error vs control's

    def breaches(self, control: Arm, treatment: Arm) -> list[str]:
        reasons: list[str] = []
        if treatment.n == 0:
            return reasons

        if treatment.fallback_rate > self.max_fallback_rate:
            reasons.append(
                f"treatment fallback rate {treatment.fallback_rate:.1%} "
                f"exceeds {self.max_fallback_rate:.1%}")

        if control.n and control.mean > 0:
            ratio = treatment.mean / control.mean
            if ratio > self.max_error_ratio:
                reasons.append(
                    f"treatment error is {ratio:.2f}x control, over "
                    f"{self.max_error_ratio:.2f}x")
        return reasons


@dataclass(frozen=True)
class ExperimentResult:
    """What the experiment can and cannot say, in one object.

    `effect` is control mean error minus treatment mean error, so a positive
    effect means the treatment is better. `mde` is what the experiment was
    capable of resolving, and `conclusive` is the comparison of the two that
    everybody skips.
    """

    name: str
    split: float
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    effect: float
    mde: float
    pooled_std: float
    conclusive: bool
    should_stop: bool
    guardrail_breaches: tuple[str, ...] = ()

    @property
    def verdict(self) -> str:
        if self.should_stop:
            return "stop: " + "; ".join(self.guardrail_breaches)
        if not self.conclusive:
            return (f"inconclusive: effect {self.effect:+.2f} min is inside "
                    f"the {self.mde:.2f} min this test could detect")
        better = "treatment" if self.effect > 0 else "control"
        return (f"{better} wins by {abs(self.effect):.2f} min "
                f"(detectable from {self.mde:.2f} min)")

    def as_row(self) -> dict[str, object]:
        return {"name": self.name, "n_control": self.n_control,
                "n_treatment": self.n_treatment,
                "mean_control": self.mean_control,
                "mean_treatment": self.mean_treatment,
                "effect": self.effect, "mde": self.mde,
                "conclusive": self.conclusive,
                "should_stop": self.should_stop}


# ------------------------------------------------------------ the experiment

class Experiment:
    """Two arms, a deterministic split, and a result that reports its own
    power.

    Entities are assigned by hash, so this object holds no assignment state and
    a restarted process agrees with the one it replaced.
    """

    def __init__(self, name: str, *, split: float = 0.5,
                 salt: str | None = None,
                 guardrail: Guardrail | None = None) -> None:
        if not 0.0 < split < 1.0:
            raise ArrivalError(
                "a split of 0 or 1 is a deploy, not an experiment")
        self.name = name
        self.split = split
        self.salt = name if salt is None else salt
        self.guardrail = guardrail or Guardrail()
        self.arms = {CONTROL: Arm(CONTROL), TREATMENT: Arm(TREATMENT)}

    def arm_for(self, entity_id: str) -> str:
        """The arm for an entity. Call it as often as you like; it agrees."""
        return assign(entity_id, self.split, self.salt)

    def record(self, entity_id: str, error: float,
               fallback: bool = False) -> str:
        """Log one trip's outcome against whichever arm its entity is in.

        The arm is derived from the entity here rather than passed in, so a
        caller cannot accidentally file an outcome under the wrong arm -- which
        is the quietest way to destroy an experiment, because the totals still
        look plausible.
        """
        arm = self.arm_for(entity_id)
        self.arms[arm].record(error, fallback)
        return arm

    def split_observed(self, entity_ids: Iterable[str]) -> float:
        """The share of these ids that land in treatment.

        Worth checking against the configured split before trusting a result:
        a hash that buckets unevenly on real ids, which are rarely uniform,
        gives arms of unequal size and an MDE worse than the one assumed.
        """
        ids = list(entity_ids)
        if not ids:
            raise ArrivalError("no ids to measure the split of")
        return sum(self.arm_for(i) == TREATMENT for i in ids) / len(ids)

    def analyse(self, *, alpha: float = DEFAULT_ALPHA,
                power: float = DEFAULT_POWER) -> ExperimentResult:
        control, treatment = self.arms[CONTROL], self.arms[TREATMENT]
        breaches = tuple(self.guardrail.breaches(control, treatment))

        if control.n < 2 or treatment.n < 2:
            # Too small to say anything, which is a result and not an error.
            return ExperimentResult(
                name=self.name, split=self.split,
                n_control=control.n, n_treatment=treatment.n,
                mean_control=control.mean, mean_treatment=treatment.mean,
                effect=float("nan"), mde=float("inf"), pooled_std=0.0,
                conclusive=False, should_stop=bool(breaches),
                guardrail_breaches=breaches)

        pooled = pooled_std(control.errors, treatment.errors)
        mde = minimum_detectable_effect(pooled, control.n, treatment.n,
                                        alpha=alpha, power=power)
        effect = control.mean - treatment.mean

        return ExperimentResult(
            name=self.name, split=self.split,
            n_control=control.n, n_treatment=treatment.n,
            mean_control=control.mean, mean_treatment=treatment.mean,
            effect=effect, mde=mde, pooled_std=pooled,
            conclusive=abs(effect) >= mde,
            should_stop=bool(breaches),
            guardrail_breaches=breaches)


def pooled_std(a: Sequence[float], b: Sequence[float]) -> float:
    """The standard deviation the two arms share.

    Pooled rather than one arm's, because the MDE is a property of the
    experiment and not of whichever arm happened to be noisier.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        raise ArrivalError("pooling needs two observations in each arm")
    var_a, var_b = statistics.variance(a), statistics.variance(b)
    return math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b)
                     / (n_a + n_b - 2))
