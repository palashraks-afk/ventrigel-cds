"""
Sample size for a two-arm Phase II trial, as a function of how strictly the
population is enriched for the responding subgroup.

The model
--------

The eligible population is a mixture of two strata. In the VentriGel case they
are the trial's own prespecified subgroups: patients treated less than 12
months post-MI ("early") and more than 12 months post-MI ("late"). Write
``pi`` for the fraction of the *eligible pool* that is late, and ``e`` for the
fraction of *enrolled* patients that is late. Setting ``e = pi`` is an
unselected trial; setting ``e = 1`` enrolls only late patients; anything in
between is partial enrichment.

For a treatment-arm change ``d_g`` in stratum ``g`` and a control-arm change
``c_g`` in the same stratum, the effect a two-arm trial estimates at enrichment
level ``e`` is

    Delta(e) = sum_g w_g(e) * (d_g - c_g)          w_late(e) = e

and the variance of the change score in the enrolled population follows the law
of total variance, exactly as in :mod:`ventrigel.recovery`:

    sigma^2(e) = sum_g w_g(e) * sigma_g^2                       (within)
               + sum_g w_g(e) * (d_g - Delta_treat(e))^2        (between)

Enrichment therefore helps twice over: it raises ``|Delta|`` by dropping strata
that dilute the effect, and it lowers ``sigma`` by removing the between-stratum
heterogeneity. Because required n scales as ``sigma^2 / Delta^2``, the two
effects compound.

The control arm is a parameter, not a finding
---------------------------------------------

This is the load-bearing caveat of the whole analysis and it is placed here
rather than buried in a limitations paragraph.

The VentriGel Phase I was single-arm. It reports what happened to treated
patients, not what would have happened untreated. The observed early-vs-late
difference is therefore consistent with two very different stories:

1.  **Effect modification.** The therapy genuinely works better in late
    patients, and the early group's adverse LVESV change reflects a real
    absence of benefit.

2.  **Differential natural history.** Untreated early post-MI patients dilate
    on their own over six months while untreated late patients are relatively
    stable. The early group's +9.3 mL could then be ordinary post-infarct
    remodeling that the therapy failed to arrest, or even partially arrested.

These are not distinguishable from single-arm data. They have different
implications: under (1) enrichment is a large real win, under (2) part of the
apparent win is an artifact of comparing each stratum against its own baseline
rather than against a control.

The model handles this by making ``c_early`` and ``c_late`` explicit,
stratum-specific inputs with no default that flatters the conclusion, and by
having :mod:`ventrigel.sensitivity` sweep them over the full plausible range.
Every headline number in this project is reported with the control assumption
that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .recovery import subgroup_effects
from .trial_data import ENDPOINTS

STRATA = ("early", "late")


# --------------------------------------------------------------------------
# Population model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stratum:
    """One subgroup of the eligible population."""

    name: str
    #: Treatment-arm change from baseline, in the endpoint's native units and
    #: native sign (not benefit-oriented).
    treatment_change: float
    #: Standard deviation of that change, recovered from the published SEM.
    sd: float
    #: Assumed control-arm change from baseline over the same interval.
    control_change: float = 0.0
    #: Multiplier applied to the *effect* to discount for the winner's curse.
    #: 1.0 takes the published estimate at face value.
    shrinkage: float = 1.0

    @property
    def effect(self) -> float:
        """Treatment minus control, discounted, in native units and sign.

        The discount multiplies the treatment-versus-control difference, not
        the raw treatment-arm change. Those are different operations and only
        the first is meaningful: the winner's curse inflates the estimated
        *effect* of selecting this subgroup, and says nothing about how much
        of the raw change was natural history. Applying the discount to the
        raw change instead lets the assumed control arm leak into the
        discount, so that discounting a large effect against a large control
        drift could paradoxically increase the estimated effect.
        """
        return (self.treatment_change - self.control_change) * self.shrinkage


@dataclass(frozen=True)
class EnrichedPopulation:
    """The population a trial actually enrolls at enrichment level ``e``."""

    early: Stratum
    late: Stratum
    #: Fraction of enrolled patients drawn from the late stratum.
    e: float
    #: True when a decrease in the endpoint is the clinically good direction.
    lower_is_better: bool

    @property
    def weights(self) -> tuple[float, float]:
        return (1.0 - self.e, self.e)

    @property
    def effect(self) -> float:
        """Treatment-versus-control difference, signed so positive = benefit."""
        w_e, w_l = self.weights
        raw = w_e * self.early.effect + w_l * self.late.effect
        return -raw if self.lower_is_better else raw

    @property
    def sd(self) -> float:
        """SD of the change score in the enrolled population."""
        w = np.array(self.weights)
        means = np.array([self.early.treatment_change, self.late.treatment_change])
        sds = np.array([self.early.sd, self.late.sd])
        grand = float(np.sum(w * means))
        var = float(np.sum(w * sds**2) + np.sum(w * (means - grand) ** 2))
        return math.sqrt(var)

    @property
    def sd_within(self) -> float:
        w = np.array(self.weights)
        sds = np.array([self.early.sd, self.late.sd])
        return math.sqrt(float(np.sum(w * sds**2)))

    @property
    def sd_between(self) -> float:
        w = np.array(self.weights)
        means = np.array([self.early.treatment_change, self.late.treatment_change])
        grand = float(np.sum(w * means))
        return math.sqrt(float(np.sum(w * (means - grand) ** 2)))

    @property
    def standardized_effect(self) -> float:
        """Cohen's d for the two-arm comparison."""
        return self.effect / self.sd if self.sd else float("nan")


def build_population(
    endpoint: str,
    e: float,
    control_early: float = 0.0,
    control_late: float = 0.0,
    timepoint: str = "6mo",
    shrinkage: float = 1.0,
) -> EnrichedPopulation:
    """Assemble the population model for one endpoint from the published data."""
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)
    return EnrichedPopulation(
        early=Stratum(
            "early", eff["early"]["mean"], eff["early"]["sd"], control_early, shrinkage
        ),
        late=Stratum(
            "late", eff["late"]["mean"], eff["late"]["sd"], control_late, shrinkage
        ),
        e=float(e),
        lower_is_better=ep.lower_is_better,
    )


# --------------------------------------------------------------------------
# Sample size
# --------------------------------------------------------------------------


def n_per_arm_normal(effect: float, sd: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Closed-form per-arm n for a two-sample comparison of means.

    Uses the normal approximation. Returned unrounded so callers can see how
    close a design sits to a boundary.
    """
    if effect == 0 or not math.isfinite(effect):
        return float("inf")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return 2.0 * (sd**2) * (z_a + z_b) ** 2 / (effect**2)


def n_per_arm_exact(
    effect: float, sd: float, alpha: float = 0.05, power: float = 0.80, cap: int = 10**7
) -> float:
    """Per-arm n solved against the noncentral t distribution.

    The normal approximation understates n at small sample sizes, which is
    exactly the regime an enriched Phase II lives in, so the reported figures
    use this instead.
    """
    if effect == 0 or not math.isfinite(effect):
        return float("inf")

    def two_sided_power(n: float) -> float:
        """Power to detect an effect of this *magnitude* in either direction.

        Distinct from the module-level :func:`power_at`, which returns zero for
        an effect pointing away from benefit. Sizing legitimately uses the
        magnitude, because a trial is powered before its direction is known;
        assurance legitimately does not, because a trial cannot succeed on an
        effect that is not there. Keeping the two separate and named
        differently avoids the two meanings being silently interchanged.
        """
        return power_at(n, abs(effect), sd, alpha)

    guess = n_per_arm_normal(effect, sd, alpha, power)
    if not math.isfinite(guess) or guess > cap:
        return float("inf")

    # Bracket the solution, skipping over any NaN region.
    lo = 2.0
    hi = max(guess * 2.0, lo + 10.0)
    for _ in range(60):
        p_hi = two_sided_power(hi)
        if math.isfinite(p_hi) and p_hi >= power:
            break
        hi *= 2.0
        if hi > cap:
            return float("inf")
    else:
        return float("inf")

    # Walk lo upward until it is a valid point below the target, so brentq
    # never evaluates inside a NaN region.
    p_lo = two_sided_power(lo)
    while not math.isfinite(p_lo) or p_lo >= power:
        lo = lo + max(1.0, 0.05 * (hi - lo))
        if lo >= hi:
            return float(hi)
        p_lo = two_sided_power(lo)

    try:
        root = optimize.brentq(lambda n: two_sided_power(n) - power, lo, hi, xtol=1e-4)
    except (ValueError, RuntimeError):
        return float(hi)
    return float(root)


def _power_normal(ncp: float, alpha: float) -> float:
    """Normal approximation to two-sample power, used as a smooth fallback."""
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp))


def power_at(n_per_arm: float, effect: float, sd: float, alpha: float = 0.05) -> float:
    """Power of a two-arm design, evaluated against the noncentral t.

    ``scipy.stats.nct`` returns NaN for large noncentrality, which the assurance
    integration reaches routinely once a draw lands on a big effect. Resolving
    those failures by thresholding the noncentrality makes the assurance curve
    non-monotone: as n grows a draw can jump from a valid 0.9 straight to 0.
    Falling back to the normal approximation keeps the function smooth, and the
    approximation is accurate precisely where the exact computation fails, since
    large noncentrality means large df and a power near one.

    Effects that do not favour treatment return 0 rather than their absolute
    value's power, because a trial cannot succeed on an effect pointing the
    wrong way.
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


def power_at_vec(
    n_per_arm: float, effects: np.ndarray, sds: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Vectorized :func:`power_at` over an array of drawn effects.

    The scalar form calls into scipy once per draw, which makes a
    twenty-thousand-draw assurance curve take minutes. One vectorized call per
    sample size takes milliseconds, which matters because the curve is
    recomputed interactively in the web application. Results are identical to
    the scalar path, fallback included.
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


#: Retained name for the power function. There is deliberately only one
#: implementation: an earlier version had a second copy in the assurance module
#: with different NaN handling, which is exactly the kind of duplication that
#: lets two parts of an analysis quietly disagree at the extremes.
achieved_power = power_at


# --------------------------------------------------------------------------
# Recruitment burden
# --------------------------------------------------------------------------


def n_screened(n_total: float, e: float, pi: float) -> float:
    """Patients who must be assessed for eligibility to enroll ``n_total``.

    Enrichment is not free. If the eligible pool is only ``pi`` late but the
    trial wants a fraction ``e`` late, every enrolled late patient costs
    ``1 / pi`` screens, and early patients found along the way are turned away
    once that stratum's quota is filled. The binding constraint is whichever
    stratum runs out first.
    """
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must lie strictly between 0 and 1")
    need_late = e * n_total
    need_early = (1.0 - e) * n_total
    return float(max(need_late / pi, need_early / (1.0 - pi)))


@dataclass(frozen=True)
class Design:
    """A costed trial design at one enrichment level."""

    endpoint: str
    e: float
    pi: float
    effect: float
    sd: float
    sd_within: float
    sd_between: float
    standardized_effect: float
    n_per_arm: float
    n_total: float
    n_screened: float
    screens_per_enrolled: float

    @property
    def feasible(self) -> bool:
        return math.isfinite(self.n_total)

    @property
    def favors_treatment(self) -> bool:
        """Whether the modelled effect points toward benefit at all.

        Sample size depends on the square of the effect, so a design whose
        effect points the wrong way still returns a finite n. That n is the
        size needed to detect a difference of that magnitude *in either
        direction*, which is not the same thing as a trial that could show the
        therapy works. Callers must check this flag before describing an n as
        the cost of demonstrating benefit.
        """
        return self.effect > 0


def design(
    endpoint: str,
    e: float,
    pi: float,
    control_early: float = 0.0,
    control_late: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.80,
    dropout: float = 0.0,
    timepoint: str = "6mo",
    shrinkage: float = 1.0,
) -> Design:
    """Size one trial design.

    ``dropout`` inflates enrolment to preserve the number of patients with an
    evaluable primary endpoint; the VentriGel Phase I lost 1 of 15 to CMR
    follow-up, so a nonzero value is the realistic default in callers.
    """
    pop = build_population(endpoint, e, control_early, control_late, timepoint, shrinkage)
    n_arm = n_per_arm_exact(pop.effect, pop.sd, alpha, power)
    if math.isfinite(n_arm):
        n_arm = math.ceil(n_arm / max(1e-9, (1.0 - dropout)))
        n_total = 2.0 * n_arm
        screened = n_screened(n_total, e, pi)
    else:
        n_total = float("inf")
        screened = float("inf")
    return Design(
        endpoint=endpoint,
        e=float(e),
        pi=float(pi),
        effect=pop.effect,
        sd=pop.sd,
        sd_within=pop.sd_within,
        sd_between=pop.sd_between,
        standardized_effect=pop.standardized_effect,
        n_per_arm=float(n_arm),
        n_total=float(n_total),
        n_screened=float(screened),
        screens_per_enrolled=float(screened / n_total) if math.isfinite(n_total) else float("inf"),
    )


def enrichment_curve(
    endpoint: str,
    pi: float,
    control_early: float = 0.0,
    control_late: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.80,
    dropout: float = 0.0,
    n_points: int = 101,
    timepoint: str = "6mo",
    shrinkage: float = 1.0,
) -> list[Design]:
    """Sweep enrichment from the unselected pool composition up to full."""
    grid = np.linspace(pi, 1.0, n_points)
    return [
        design(
            endpoint,
            float(e),
            pi,
            control_early,
            control_late,
            alpha,
            power,
            dropout,
            timepoint,
            shrinkage,
        )
        for e in grid
    ]


# --------------------------------------------------------------------------
# Confirming the subgroup claim itself
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionDesign:
    """A 2x2 trial powered to establish effect modification, not just effect.

    The enriched design above enrolls only late patients. It can confirm that
    the therapy works *in* that stratum; it can never establish that timing
    matters, because it contains no early patients to compare against. But
    "treat late, not early" is the actual claim, and it is the claim a sponsor
    would be acting on. Confirming it needs both strata and both arms.
    """

    endpoint: str
    #: Treatment-versus-control effect in each stratum, benefit-signed.
    effect_early: float
    effect_late: float
    #: The interaction contrast: how much better the late stratum does.
    contrast: float
    sd_pooled: float
    n_per_cell: float
    n_total: float
    #: Total enrollment of the enriched two-arm design, for comparison.
    n_enriched_reference: float
    ratio_to_enriched: float

    @property
    def feasible(self) -> bool:
        return math.isfinite(self.n_total)


def interaction_design(
    endpoint: str,
    control_early: float = 0.0,
    control_late: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.80,
    dropout: float = 0.0,
    shrinkage: float = 1.0,
    timepoint: str = "6mo",
) -> InteractionDesign:
    """Size a 2x2 trial powered on the interaction contrast.

    With four equally sized cells the interaction estimate

        (y_LT - y_LC) - (y_ET - y_EC)

    has variance ``4 * sigma^2 / n_cell``, twice that of a simple two-arm
    comparison at the same per-group n. Required enrollment therefore carries a
    factor of two relative to a main-effect design of the same effect size --
    the familiar result that interactions are expensive.

    That penalty is partly offset here, because the contrast being detected is
    larger than the late-stratum effect alone. Whether the offset wins depends
    entirely on the control assumptions: against no control drift the contrast
    is the full 16.9 mL and the interaction design is barely more expensive,
    while against anchored controls much of the early stratum's apparent harm
    is natural history, the contrast roughly halves, and the design becomes
    materially larger.
    """
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)
    sign = -1.0 if ep.lower_is_better else 1.0

    e_early = sign * (eff["early"]["mean"] - control_early) * shrinkage
    e_late = sign * (eff["late"]["mean"] - control_late) * shrinkage
    contrast = e_late - e_early

    # Pooled within-stratum SD, weighted by the n each stratum contributed.
    n_e, n_l = eff["early"]["n"], eff["late"]["n"]
    var_pooled = (
        (n_e - 1) * eff["early"]["sd"] ** 2 + (n_l - 1) * eff["late"]["sd"] ** 2
    ) / (n_e + n_l - 2)
    sd_pooled = math.sqrt(var_pooled)

    ref = design(
        endpoint, 1.0, 0.5, control_early, control_late, alpha, power, dropout,
        timepoint, shrinkage,
    )

    if contrast <= 0:
        return InteractionDesign(
            endpoint, e_early, e_late, contrast, sd_pooled,
            math.inf, math.inf, ref.n_total, float("nan"),
        )

    # Solve as a two-arm problem on the contrast, then double for the extra
    # variance the four-cell structure carries.
    n_equiv = n_per_arm_exact(contrast, sd_pooled, alpha, power)
    if not math.isfinite(n_equiv):
        return InteractionDesign(
            endpoint, e_early, e_late, contrast, sd_pooled,
            math.inf, math.inf, ref.n_total, float("nan"),
        )
    n_cell = math.ceil(2.0 * n_equiv / max(1e-9, 1.0 - dropout))
    n_total = 4.0 * n_cell
    return InteractionDesign(
        endpoint=endpoint,
        effect_early=e_early,
        effect_late=e_late,
        contrast=contrast,
        sd_pooled=sd_pooled,
        n_per_cell=float(n_cell),
        n_total=float(n_total),
        n_enriched_reference=ref.n_total,
        ratio_to_enriched=(n_total / ref.n_total) if ref.feasible and ref.n_total else float("nan"),
    )


def dilution_factor(endpoint: str, pi: float, timepoint: str = "6mo") -> float:
    """How many times larger an unselected trial must be than an enriched one.

    Reported with zero control drift, which is the assumption under which the
    single-arm data speak most directly. Infinite when the unselected effect
    is zero, which is close to the truth for LVESV and is the point.
    """
    unsel = design(endpoint, pi, pi, timepoint=timepoint)
    enr = design(endpoint, 1.0, pi, timepoint=timepoint)
    if not enr.feasible or enr.n_total == 0:
        return float("nan")
    return unsel.n_total / enr.n_total
