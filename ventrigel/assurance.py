"""
Probability that the trial actually works, as distinct from its nominal power.

Power answers a conditional question: *if* the true effect equals the value I
assumed, how often will the trial reach significance? Sizing a trial for 80%
power at a point estimate and then describing it as an 80% trial silently
promotes that assumption to a fact.

Assurance answers the unconditional question. It integrates power over the
uncertainty in the effect itself:

    assurance(n) = E_theta[ power(n, theta) ]

with the expectation taken over the sampling distribution of the stratum's true
mean and variance -- the same distribution :mod:`ventrigel.sensitivity` already
uses for the bootstrap. It is sometimes called probability of success, and it is
the quantity a sponsor deciding whether to fund the trial actually cares about.

The gap matters enormously here. The enriched VentriGel design is sized from six
to eight patients, so the effect is barely pinned down, and assurance falls far
below nominal power. It also *plateaus*: no sample size drives assurance above
roughly 90%, because a few percent of the plausible effects point the wrong way
and no amount of enrollment fixes an effect that is not there. That ceiling is
the honest expression of what a fifteen-patient single-arm trial can promise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .power import power_at, power_at_vec
from .recovery import subgroup_effects
from .sensitivity import DEFAULT_SHRINKAGE, _draw_stratum
from .trial_data import ENDPOINTS


@dataclass(frozen=True)
class AssuranceResult:
    endpoint: str
    n_total: float
    shrinkage: float
    control: float
    assurance: float
    #: Power computed at the point estimate, for direct comparison.
    nominal_power: float
    #: Share of draws in which the effect points away from benefit. This is the
    #: irreducible part: no sample size overcomes it.
    wrong_sign_fraction: float
    n_draws: int


def _draws(
    endpoint: str,
    n_draws: int,
    seed: int,
    shrinkage: float,
    control: float,
    timepoint: str,
    stratum: str,
    control_se: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (effect, sd) pairs for one stratum, benefit-signed.

    ``control_se`` propagates the uncertainty in the control-arm anchor itself.
    Leaving it at zero asserts that the comparator is known exactly, which is
    never true: FOCUS-CCTRN's "zero" comes from 28 patients and carries a
    standard error of roughly 4 mL, the same order as the 7.6 mL effect being
    measured against it. Ignoring that understates the required sample size by
    about a factor of two at conventional targets, so it is a default of
    convenience only and callers are expected to supply the real value from
    :func:`ventrigel.literature.anchored_control`.
    """
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)[stratum]
    rng = np.random.default_rng(seed)
    sign = -1.0 if ep.lower_is_better else 1.0
    effects = np.empty(n_draws)
    sds = np.empty(n_draws)
    for i in range(n_draws):
        mu, sd = _draw_stratum(eff["mean"], eff["sd"], eff["n"], rng)
        c = control if control_se <= 0 else float(rng.normal(control, control_se))
        effects[i] = sign * (mu - c) * shrinkage
        sds[i] = sd
    return effects, sds


def _draws_mixture(
    endpoint: str,
    e: float,
    n_draws: int,
    seed: int,
    shrinkage: float,
    control_early: float,
    control_late: float,
    control_early_se: float,
    control_late_se: float,
    timepoint: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw (effect, sd) for a population enriched to level ``e``.

    The single-stratum :func:`_draws` is correct only for a fully enriched
    design. Anything short of that enrolls both strata, and its effect and
    variance follow the same mixture arithmetic the rest of the model uses, so
    assurance for a partially enriched trial has to be built the same way.
    Reusing the late-only draws for a partially enriched design reports the
    success probability of a trial that is not being described.
    """
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)
    rng = np.random.default_rng(seed)
    sign = -1.0 if ep.lower_is_better else 1.0
    w_e, w_l = 1.0 - e, e

    effects = np.empty(n_draws)
    sds = np.empty(n_draws)
    for i in range(n_draws):
        mu_e, sd_e = _draw_stratum(eff["early"]["mean"], eff["early"]["sd"], eff["early"]["n"], rng)
        mu_l, sd_l = _draw_stratum(eff["late"]["mean"], eff["late"]["sd"], eff["late"]["n"], rng)
        c_e = control_early if control_early_se <= 0 else float(rng.normal(control_early, control_early_se))
        c_l = control_late if control_late_se <= 0 else float(rng.normal(control_late, control_late_se))

        raw = w_e * (mu_e - c_e) + w_l * (mu_l - c_l)
        effects[i] = sign * raw * shrinkage

        grand = w_e * mu_e + w_l * mu_l
        var = (
            w_e * sd_e**2 + w_l * sd_l**2
            + w_e * (mu_e - grand) ** 2 + w_l * (mu_l - grand) ** 2
        )
        sds[i] = math.sqrt(max(var, 1e-12))
    return effects, sds


def assurance_at_enrichment(
    endpoint: str,
    n_total: float,
    e: float,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control_early: float = 0.0,
    control_late: float = 0.0,
    control_early_se: float = 0.0,
    control_late_se: float = 0.0,
    alpha: float = 0.05,
    n_draws: int = 20000,
    seed: int = 20190712,
    timepoint: str = "6mo",
) -> AssuranceResult:
    """Assurance for a trial enriched to level ``e``, not necessarily 1.0."""
    effects, sds = _draws_mixture(
        endpoint, e, n_draws, seed, shrinkage, control_early, control_late,
        control_early_se, control_late_se, timepoint,
    )
    a = float(power_at_vec(n_total / 2.0, effects, sds, alpha).mean())
    return AssuranceResult(
        endpoint=endpoint,
        n_total=float(n_total),
        shrinkage=shrinkage,
        control=control_late,
        assurance=a,
        nominal_power=power_at(
            n_total / 2.0, float(np.median(effects)), float(np.median(sds)), alpha
        ),
        wrong_sign_fraction=float(np.mean(effects <= 0)),
        n_draws=n_draws,
    )


def assurance_curve_at_enrichment(
    endpoint: str,
    n_values: np.ndarray,
    e: float,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control_early: float = 0.0,
    control_late: float = 0.0,
    control_early_se: float = 0.0,
    control_late_se: float = 0.0,
    alpha: float = 0.05,
    n_draws: int = 20000,
    seed: int = 20190712,
    timepoint: str = "6mo",
) -> tuple[np.ndarray, float]:
    """``(assurance per n, ceiling)`` for a trial enriched to level ``e``."""
    effects, sds = _draws_mixture(
        endpoint, e, n_draws, seed, shrinkage, control_early, control_late,
        control_early_se, control_late_se, timepoint,
    )
    vals = np.array(
        [float(power_at_vec(float(n) / 2.0, effects, sds, alpha).mean()) for n in n_values]
    )
    return vals, float(np.mean(effects > 0))


def assurance(
    endpoint: str,
    n_total: float,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control: float = 0.0,
    alpha: float = 0.05,
    n_draws: int = 20000,
    seed: int = 20190712,
    timepoint: str = "6mo",
    stratum: str = "late",
    control_se: float = 0.0,
) -> AssuranceResult:
    """Probability of a significant result for a fully enriched design."""
    effects, sds = _draws(
        endpoint, n_draws, seed, shrinkage, control, timepoint, stratum, control_se
    )
    n_arm = n_total / 2.0
    total = float(power_at_vec(n_arm, effects, sds, alpha).sum())

    ep = ENDPOINTS[endpoint]
    base = subgroup_effects(endpoint, timepoint)[stratum]
    sign = -1.0 if ep.lower_is_better else 1.0
    point_effect = sign * (base["mean"] - control) * shrinkage

    return AssuranceResult(
        endpoint=endpoint,
        n_total=float(n_total),
        shrinkage=shrinkage,
        control=control,
        assurance=total / n_draws,
        nominal_power=power_at(n_arm, point_effect, base["sd"], alpha),
        wrong_sign_fraction=float(np.mean(effects <= 0)),
        n_draws=n_draws,
    )


def assurance_curve(
    endpoint: str,
    n_values: np.ndarray | None = None,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control: float = 0.0,
    alpha: float = 0.05,
    n_draws: int = 8000,
    seed: int = 20190712,
    timepoint: str = "6mo",
    control_se: float = 0.0,
) -> list[AssuranceResult]:
    """Assurance across a range of trial sizes.

    Uses common random numbers: one set of effect draws is evaluated at every
    sample size rather than redrawing per point. Independent draws would leave
    Monte Carlo noise of a percentage point or two at each N, which shows up as
    a visibly jagged and locally non-monotone curve even though assurance is
    monotone in N. Sharing the draws removes that noise entirely, and it is
    also the more honest comparison, since the only thing changing across the
    curve is then the sample size.
    """
    ns = (
        np.asarray(n_values)
        if n_values is not None
        else np.unique(np.round(np.logspace(math.log10(20), math.log10(2000), 30)).astype(int))
    )
    effects, sds = _draws(
        endpoint, n_draws, seed, shrinkage, control, timepoint, "late", control_se
    )

    ep = ENDPOINTS[endpoint]
    base = subgroup_effects(endpoint, timepoint)["late"]
    sign = -1.0 if ep.lower_is_better else 1.0
    point_effect = sign * (base["mean"] - control) * shrinkage
    wrong = float(np.mean(effects <= 0))

    out: list[AssuranceResult] = []
    for n in ns:
        n_arm = float(n) / 2.0
        a = float(power_at_vec(n_arm, effects, sds, alpha).mean())
        out.append(
            AssuranceResult(
                endpoint=endpoint,
                n_total=float(n),
                shrinkage=shrinkage,
                control=control,
                assurance=a,
                nominal_power=power_at(n_arm, point_effect, base["sd"], alpha),
                wrong_sign_fraction=wrong,
                n_draws=n_draws,
            )
        )
    return out


def n_for_assurance(
    endpoint: str,
    target: float = 0.80,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control: float = 0.0,
    alpha: float = 0.05,
    n_draws: int = 6000,
    seed: int = 20190712,
    timepoint: str = "6mo",
    n_max: float = 100000.0,
    control_se: float = 0.0,
) -> float:
    """Smallest total enrollment reaching a given assurance.

    Returns infinity when the target lies above the achievable ceiling, which
    is the common case for demanding targets: assurance is bounded above by
    one minus the wrong-sign fraction, so an 85% target is unreachable when 3%
    of plausible effects point the wrong way and the rest are modest.
    """
    # Pre-draw once so every evaluation sees the same sample and the function
    # is deterministic and monotone, which the solver requires.
    effects, sds = _draws(
        endpoint, n_draws, seed, shrinkage, control, timepoint, "late", control_se
    )

    def f(n_total: float) -> float:
        return float(power_at_vec(n_total / 2.0, effects, sds, alpha).mean())

    if f(n_max) < target:
        return float("inf")
    lo, hi = 4.0, n_max
    for _ in range(60):
        mid = math.sqrt(lo * hi)  # geometric bisection: n spans orders of magnitude
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.001:
            break
    return float(math.ceil(hi / 2.0) * 2)


@dataclass(frozen=True)
class ProgrammeOutcome:
    """Probability that the whole programme succeeds, not just the trial.

    Assurance is conditional on the subgroup effect being real. That condition
    is doing a great deal of work here, because the effect rests on one
    nominally significant interaction out of nine that survives no multiplicity
    correction. A sponsor is not choosing whether to run a trial given that the
    effect exists; they are choosing whether to run it at all.
    """

    n_total: float
    prior_effect_real: float
    conditional_assurance: float
    unconditional: float


def programme_success(
    endpoint: str,
    n_total: float,
    prior_effect_real: float,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control: float = 0.0,
    control_se: float = 0.0,
    alpha: float = 0.05,
    n_draws: int = 20000,
    seed: int = 20190712,
    timepoint: str = "6mo",
) -> ProgrammeOutcome:
    """Unconditional probability of success: prior times assurance.

    ``prior_effect_real`` is the probability that the early-versus-late
    interaction reflects something real rather than the best of nine
    unadjusted comparisons in fifteen patients. No attempt is made to derive
    it. It is a judgement, it is the single most consequential number in the
    analysis, and the honest thing is to make the reader supply it and see the
    consequence rather than to bury a value inside a calculation.

    For orientation: a nominal p of 0.034 on one of nine tests, with no
    correction surviving, is weak evidence. Balanced baselines and the failure
    of the regression-to-the-mean explanation pull the other way. Somewhere
    between 0.3 and 0.6 is defensible; below 0.3 the programme is hard to
    justify at any sample size.
    """
    a = assurance(
        endpoint, n_total, shrinkage, control, alpha, n_draws, seed, timepoint,
        "late", control_se,
    )
    return ProgrammeOutcome(
        n_total=float(n_total),
        prior_effect_real=float(prior_effect_real),
        conditional_assurance=a.assurance,
        unconditional=float(prior_effect_real) * a.assurance,
    )


def programme_grid(
    endpoint: str,
    n_values: np.ndarray,
    priors: np.ndarray,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control: float = 0.0,
    control_se: float = 0.0,
    alpha: float = 0.05,
    n_draws: int = 20000,
    seed: int = 20190712,
    timepoint: str = "6mo",
) -> np.ndarray:
    """Unconditional success probability over (sample size x prior).

    Returns an array indexed ``[i, j]`` for ``n_values[i]`` and ``priors[j]``.
    Because the prior enters multiplicatively, the grid makes the ceiling
    obvious: no sample size can lift the programme above the prior itself.
    """
    curve = assurance_curve(
        endpoint, np.asarray(n_values), shrinkage, control, alpha, n_draws, seed,
        timepoint, control_se,
    )
    a = np.array([c.assurance for c in curve])
    return np.outer(a, np.asarray(priors, dtype=float))


def assurance_ceiling(
    endpoint: str,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control: float = 0.0,
    n_draws: int = 20000,
    seed: int = 20190712,
    timepoint: str = "6mo",
    control_se: float = 0.0,
) -> float:
    """The best assurance any sample size can reach.

    Equal to the fraction of draws whose effect points toward benefit, since
    those and only those can be detected given unlimited enrollment.
    """
    effects, _ = _draws(
        endpoint, n_draws, seed, shrinkage, control, timepoint, "late", control_se
    )
    return float(np.mean(effects > 0))
