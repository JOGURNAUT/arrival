"""The ways an A/B test on an ETA model quietly stops meaning anything.

These tests are mostly about arithmetic, and the arithmetic is what decides
whether a model ships. The one that matters most is
`test_the_mde_is_larger_than_a_tiny_claimed_improvement`: it is the difference
between a result and a number.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys

import pytest

from arrival.contracts import ArrivalError
from arrival.experiment import (CONTROL, TREATMENT, Experiment, Guardrail,
                                assign, bucket, minimum_detectable_effect,
                                pooled_std, sample_size_per_arm)

DRIVERS = [f"D-{i:05d}" for i in range(20_000)]


# ------------------------------------------------------------- assignment

def test_the_same_entity_always_lands_in_the_same_arm():
    """Per-request randomness is the classic way to run an invalid experiment.
    The same driver gets control at 09:00 and treatment at 09:04, so no entity
    is ever purely in one arm, the arms stop being independent samples, and the
    difference between their means is measuring nothing at all."""
    experiment = Experiment("eta-v2", split=0.5)

    for entity in DRIVERS[:500]:
        arms = {experiment.arm_for(entity) for _ in range(20)}
        assert len(arms) == 1


def test_the_arm_survives_a_restart_of_the_process():
    """Python randomises string hashing per process. A split built on the
    built-in `hash` reshuffles every arm on every deploy, so drivers cross arms
    mid-experiment and the before-and-after comparison measures the reshuffle
    rather than the model."""
    probe = (
        "from arrival.experiment import Experiment;"
        "e = Experiment('eta-v2', split=0.37);"
        "print(''.join(e.arm_for(f'D-{i:05d}')[0] for i in range(200)))"
    )
    runs = []
    for seed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", probe], env=env,
                             capture_output=True, text=True, check=True,
                             cwd=os.path.dirname(os.path.dirname(__file__)))
        runs.append(out.stdout.strip())

    assert len(set(runs)) == 1
    assert set(runs[0]) == {"c", "t"}


def test_different_entities_spread_across_both_arms():
    """A hash that buckets everything one way is not an experiment, and the
    failure looks like a successful launch: one arm simply has no traffic and
    its metrics never move."""
    experiment = Experiment("eta-v2", split=0.5)
    arms = [experiment.arm_for(d) for d in DRIVERS[:1000]]

    assert set(arms) == {CONTROL, TREATMENT}
    assert 400 < arms.count(TREATMENT) < 600


@pytest.mark.parametrize("split", [0.1, 0.3, 0.5, 0.9])
def test_the_configured_split_is_roughly_honoured(split):
    """Arms that are not the size the configuration claims have a worse MDE
    than the one assumed when the experiment was sized, so it runs for the
    planned fortnight and resolves less than planned."""
    observed = Experiment("eta-v2", split=split).split_observed(DRIVERS)
    assert observed == pytest.approx(split, abs=0.015)


def test_two_experiments_do_not_share_a_split():
    """Unsalted, two experiments launched the same week put exactly the same
    drivers in treatment. Their populations correlate completely and neither
    result is about its own change."""
    a, b = Experiment("eta-v2"), Experiment("pack-time-v1")
    disagreements = sum(a.arm_for(d) != b.arm_for(d) for d in DRIVERS[:2000])

    assert 800 < disagreements < 1200


def test_a_bucket_is_a_share_and_a_split_of_one_is_a_deploy():
    """A split of 0 or 1 gives an empty arm, and every downstream statistic is
    then computed against nothing while appearing to run normally."""
    assert all(0.0 <= bucket(d) < 1.0 for d in DRIVERS[:1000])

    with pytest.raises(ArrivalError):
        Experiment("all-in", split=1.0)
    with pytest.raises(ArrivalError):
        assign("D-00001", split=1.4)


def test_an_outcome_is_filed_under_the_arm_its_entity_is_in():
    """Filing an outcome under the wrong arm is the quietest way to destroy an
    experiment: the totals stay plausible, the means move towards each other,
    and the test reports a smaller effect than is really there."""
    experiment = Experiment("eta-v2", split=0.5)
    for driver in DRIVERS[:200]:
        assert experiment.record(driver, 5.0) == experiment.arm_for(driver)

    assert (experiment.arms[CONTROL].n
            + experiment.arms[TREATMENT].n) == 200


# -------------------------------------------------------------- guardrails

def _arm_of(experiment, name, error, n, fallbacks=0):
    arm = experiment.arms[name]
    for i in range(n):
        arm.record(error, fallback=i < fallbacks)
    return arm


def test_the_guardrail_fires_when_the_treatment_arm_keeps_falling_back():
    """A treatment arm falling back to the flat promise for a third of its
    requests often posts a respectable average error, because the promise is
    not a terrible estimate. It is still broken, and accuracy alone will let it
    run for the full fortnight."""
    experiment = Experiment("eta-v2", guardrail=Guardrail(
        max_fallback_rate=0.05, max_error_ratio=1.10))
    _arm_of(experiment, CONTROL, 8.0, 500)
    _arm_of(experiment, TREATMENT, 8.0, 500, fallbacks=150)

    result = experiment.analyse()
    assert result.should_stop is True
    assert "fallback rate 30.0%" in result.verdict


def test_the_guardrail_fires_when_the_treatment_arm_is_simply_worse():
    """An experiment is a change to production with a measurement attached,
    and the measurement takes days. A treatment that is 40% less accurate must
    not be allowed to run for days while the maths settles."""
    experiment = Experiment("eta-v2")
    _arm_of(experiment, CONTROL, 8.0, 500)
    _arm_of(experiment, TREATMENT, 11.2, 500)

    result = experiment.analyse()
    assert result.should_stop is True
    assert "1.40x control" in result.guardrail_breaches[0]


def test_the_guardrail_stays_quiet_on_a_healthy_treatment_arm():
    """A guardrail that fires on noise is turned off within a week, and then
    the real degradation runs unattended."""
    experiment = Experiment("eta-v2")
    _arm_of(experiment, CONTROL, 8.0, 500)
    _arm_of(experiment, TREATMENT, 7.6, 500, fallbacks=5)

    result = experiment.analyse()
    assert result.should_stop is False
    assert result.guardrail_breaches == ()


# --------------------------------------------------------------- the maths

def test_the_mde_matches_the_closed_form_it_claims():
    """Everything downstream of this number is a ship-or-not decision. If the
    formula is wrong the experiment reports a resolution it does not have, and
    the mistake is undetectable from the output."""
    # (z[0.975] + z[0.80]) * std * sqrt(1/n_c + 1/n_t)
    expected = 2.801585218 * 6.0 * (2 / 400) ** 0.5
    assert minimum_detectable_effect(6.0, 400, 400) == pytest.approx(expected)
    assert minimum_detectable_effect(6.0, 400, 400) == pytest.approx(1.1886,
                                                                     abs=1e-4)


def test_the_mde_is_larger_than_a_tiny_claimed_improvement():
    """This is the test that makes the maths honest.

    Four hundred trips an arm, per-trip errors spread over five and a half
    minutes, and a treatment that is genuinely 2% better: sixteen seconds off
    an eight minute average error. The smallest effect this experiment could
    reliably resolve is 1.10 minutes, roughly seven times the effect being
    claimed.

    Watch what the arms then report. The measured difference comes out at
    +0.31 minutes -- a clean, plausible, quotable number, and nearly double
    the improvement that was actually built in. That is what noise looks like
    at this sample size, and a launch review handed +0.31 with no MDE beside
    it has no way to tell it apart from a result.

    If this test goes red because the MDE shrank, somebody has made the
    experiment look more powerful than it is, and the next 2% claim ships."""
    rng = random.Random(20260901)
    experiment = Experiment("eta-v2", split=0.5)
    for driver in DRIVERS[:800]:
        better = experiment.arm_for(driver) == TREATMENT
        experiment.record(driver, max(0.0, rng.gauss(7.84 if better else 8.0,
                                                    6.0)))

    result = experiment.analyse()
    assert result.n_control > 350 and result.n_treatment > 350
    assert result.mde == pytest.approx(1.10, abs=0.02)
    assert result.effect == pytest.approx(0.31, abs=0.02)
    assert abs(result.effect) < result.mde
    assert result.conclusive is False
    assert "inconclusive" in result.verdict

    # And the honest version of the question: how much traffic would it take.
    assert sample_size_per_arm(0.16, 6.0) > 20_000


def test_an_effect_big_enough_to_see_is_reported_as_conclusive():
    """An MDE that calls everything inconclusive is as useless as no MDE at
    all -- it would block a real four-minute improvement from ever shipping."""
    rng = random.Random(7)
    experiment = Experiment("eta-v2", split=0.5,
                            guardrail=Guardrail(max_error_ratio=10.0))
    for driver in DRIVERS[:800]:
        better = experiment.arm_for(driver) == TREATMENT
        experiment.record(driver, max(0.0, rng.gauss(4.0 if better else 8.0,
                                                    6.0)))

    result = experiment.analyse()
    assert result.effect > result.mde
    assert result.conclusive is True
    assert "treatment wins" in result.verdict


def test_the_mde_only_halves_when_the_arms_quadruple():
    """The square root is why "just run it longer" is usually not an answer.
    Four times the traffic to resolve half the effect, which for a daily trip
    volume is the difference between a week and a month -- and it is better to
    know that before launching than after."""
    small = minimum_detectable_effect(6.0, 400, 400)
    large = minimum_detectable_effect(6.0, 1600, 1600)

    assert large == pytest.approx(small / 2, rel=1e-9)
    assert (small, large) == pytest.approx((1.1886, 0.5943), abs=1e-4)


def test_sample_size_grows_as_the_square_of_the_shrinking_effect():
    """Quoted the other way round, this is the number that stops an experiment
    being planned for an effect it can never see."""
    assert sample_size_per_arm(1.0, 6.0) == 566
    assert sample_size_per_arm(0.5, 6.0) == pytest.approx(566 * 4, rel=0.01)

    with pytest.raises(ArrivalError):
        sample_size_per_arm(0.0, 6.0)


def test_an_experiment_too_small_to_resolve_anything_says_so():
    """Three trips an arm will happily produce two means and a difference. The
    result has to refuse to dress that up as an effect."""
    experiment = Experiment("eta-v2")
    _arm_of(experiment, CONTROL, 8.0, 1)
    _arm_of(experiment, TREATMENT, 2.0, 1)

    result = experiment.analyse()
    assert result.mde == float("inf")
    assert result.conclusive is False

    with pytest.raises(ArrivalError):
        pooled_std([1.0], [2.0, 3.0])


def test_the_pooled_deviation_is_shared_rather_than_one_arms():
    """The MDE is a property of the experiment. Taking the quieter arm's
    deviation reports a resolution the experiment does not have, and taking the
    noisier one's throws away power that was really there."""
    quiet = [10.0, 10.5, 9.5, 10.0, 10.2]
    noisy = [2.0, 18.0, 5.0, 15.0, 10.0]
    pooled = pooled_std(quiet, noisy)

    assert min(0.4, 6.7) < pooled < max(0.4, 6.7)


def test_the_result_carries_per_arm_n_beside_every_mean():
    """A mean without its n makes a comparison over forty trips look identical
    to one over forty thousand."""
    experiment = Experiment("eta-v2")
    _arm_of(experiment, CONTROL, 8.0, 300)
    _arm_of(experiment, TREATMENT, 7.0, 220)
    row = experiment.analyse().as_row()

    assert row["n_control"] == 300
    assert row["n_treatment"] == 220
    assert row["mean_control"] == pytest.approx(8.0)
    assert row["mean_treatment"] == pytest.approx(7.0)
    assert "mde" in row
