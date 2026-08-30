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
from scipy import optimize, stats

from .recovery import subgroup_effects
from .sensitivity import DEFAULT_SHRINKAGE, _draw_stratum
from .trial_data import ENDPOINTS


def _power_normal(ncp: float, alpha: float) -> float:
    """Normal approximation to two-sample power, used as a smooth fallback."""
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp))


def power_at_vec(
    n_per_arm: float, effects: np.ndarray, sds: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Vectorized :func:`power_at` over an array of drawn effects.

    The scalar form calls into ``scipy.stats.nct`` once per draw, which makes a
    twelve-thousand-draw assurance curve take minutes. Evaluating the whole
    draw set in one call per sample size takes milliseconds, which matters
    because the curve is recomputed interactively in the web application.
    Results are identical to the scalar path, including the normal-approximation
    fallback where the exact computation returns NaN.
    """
    effects = np.asarray(effects, dtype=float)
    sds = np.asarray(sds, dtype=float)
    out = np.zeros(effects.shape, dtype=float)
    if n_per_arm <= 1.5:
        return out

    valid = (effects > 0) & (sds > 0) & np.isfinite(effects) & np.isfinite(sds)
    if not valid.any():
        return out

    df = 2 * n_per_arm - 2
    ncp = np.zeros_like(effects)
    ncp[valid] = effects[valid] / (sds[valid] * math.sqrt(2.0 / n_per_arm))

    crit = float(stats.t.ppf(1 - alpha / 2, df))
    with np.errstate(all="ignore"):
        p = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
        z = float(stats.norm.ppf(1 - alpha / 2))
        fallback = stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp)
    p = np.where(np.isfinite(p), p, fallback)
    out[valid] = np.clip(p[valid], 0.0, 1.0)
    return out


def power_at(n_per_arm: float, effect: float, sd: float, alpha: float = 0.05) -> float:
    """Power of a two-arm design, evaluated against the noncentral t.

    ``scipy.stats.nct`` returns NaN for large noncentrality, which the
    integration below reaches routinely once a draw lands on a big effect. An
    earlier version resolved those failures to 0 or 1 by thresholding the
    noncentrality, which made the assurance curve non-monotone: as n grew, a
    draw could jump from a valid power of 0.9 straight to 0. Falling back to
    the normal approximation instead keeps the function smooth and monotone,
    and the approximation is accurate precisely where the exact computation
    fails, since large noncentrality means large df and a power near one.
    """
    if n_per_arm <= 1.5 or effect <= 0 or sd <= 0:
        return 0.0
    df = 2 * n_per_arm - 2
    ncp = effect / (sd * math.sqrt(2.0 / n_per_arm))
    if not math.isfinite(ncp):
        return 0.0
    try:
        crit = stats.t.ppf(1 - alpha / 2, df)
        p = float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))
    except (ValueError, FloatingPointError):
        return _power_normal(ncp, alpha)
    return p if math.isfinite(p) else _power_normal(ncp, alpha)


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
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (effect, sd) pairs for one stratum, benefit-signed."""
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)[stratum]
    rng = np.random.default_rng(seed)
    sign = -1.0 if ep.lower_is_better else 1.0
    effects = np.empty(n_draws)
    sds = np.empty(n_draws)
    for i in range(n_draws):
        mu, sd = _draw_stratum(eff["mean"], eff["sd"], eff["n"], rng)
        effects[i] = sign * (mu - control) * shrinkage
        sds[i] = sd
    return effects, sds


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
) -> AssuranceResult:
    """Probability of a significant result for a fully enriched design."""
    effects, sds = _draws(endpoint, n_draws, seed, shrinkage, control, timepoint, stratum)
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
    effects, sds = _draws(endpoint, n_draws, seed, shrinkage, control, timepoint, "late")

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
) -> float:
    """Smallest total enrollment reaching a given assurance.

    Returns infinity when the target lies above the achievable ceiling, which
    is the common case for demanding targets: assurance is bounded above by
    one minus the wrong-sign fraction, so an 85% target is unreachable when 3%
    of plausible effects point the wrong way and the rest are modest.
    """
    # Pre-draw once so every evaluation sees the same sample and the function
    # is deterministic and monotone, which the solver requires.
    effects, sds = _draws(endpoint, n_draws, seed, shrinkage, control, timepoint, "late")

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


def assurance_ceiling(
    endpoint: str,
    shrinkage: float = DEFAULT_SHRINKAGE,
    control: float = 0.0,
    n_draws: int = 20000,
    seed: int = 20190712,
    timepoint: str = "6mo",
) -> float:
    """The best assurance any sample size can reach.

    Equal to the fraction of draws whose effect points toward benefit, since
    those and only those can be detected given unlimited enrollment.
    """
    effects, _ = _draws(endpoint, n_draws, seed, shrinkage, control, timepoint, "late")
    return float(np.mean(effects > 0))
