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

    @property
    def effect(self) -> float:
        """Treatment minus control, in native units and sign."""
        return self.treatment_change - self.control_change


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
) -> EnrichedPopulation:
    """Assemble the population model for one endpoint from the published data."""
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)
    return EnrichedPopulation(
        early=Stratum("early", eff["early"]["mean"], eff["early"]["sd"], control_early),
        late=Stratum("late", eff["late"]["mean"], eff["late"]["sd"], control_late),
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

    def power_at(n: float) -> float:
        # The noncentral t can return NaN for extreme noncentrality, which the
        # bootstrap reaches routinely. Treat a non-finite result as "no
        # information" rather than letting it poison the root-finder.
        if n <= 1.5:
            return 0.0
        df = 2 * n - 2
        ncp = abs(effect) / (sd * math.sqrt(2.0 / n))
        if not math.isfinite(ncp):
            return 0.0
        try:
            crit = stats.t.ppf(1 - alpha / 2, df)
            p = float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))
        except (ValueError, FloatingPointError):
            return float("nan")
        return p if math.isfinite(p) else float("nan")

    guess = n_per_arm_normal(effect, sd, alpha, power)
    if not math.isfinite(guess) or guess > cap:
        return float("inf")

    # Bracket the solution, skipping over any NaN region.
    lo = 2.0
    hi = max(guess * 2.0, lo + 10.0)
    for _ in range(60):
        p_hi = power_at(hi)
        if math.isfinite(p_hi) and p_hi >= power:
            break
        hi *= 2.0
        if hi > cap:
            return float("inf")
    else:
        return float("inf")

    # Walk lo upward until it is a valid point below the target, so brentq
    # never evaluates inside a NaN region.
    p_lo = power_at(lo)
    while not math.isfinite(p_lo) or p_lo >= power:
        lo = lo + max(1.0, 0.05 * (hi - lo))
        if lo >= hi:
            return float(hi)
        p_lo = power_at(lo)

    try:
        root = optimize.brentq(lambda n: power_at(n) - power, lo, hi, xtol=1e-4)
    except (ValueError, RuntimeError):
        return float(hi)
    return float(root)


def achieved_power(
    n_per_arm: float, effect: float, sd: float, alpha: float = 0.05
) -> float:
    """Power of a given design, for reading the model in the other direction."""
    if n_per_arm <= 1 or effect == 0 or not math.isfinite(effect):
        return 0.0
    df = 2 * n_per_arm - 2
    ncp = abs(effect) / (sd * math.sqrt(2.0 / n_per_arm))
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


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
) -> Design:
    """Size one trial design.

    ``dropout`` inflates enrolment to preserve the number of patients with an
    evaluable primary endpoint; the VentriGel Phase I lost 1 of 15 to CMR
    follow-up, so a nonzero value is the realistic default in callers.
    """
    pop = build_population(endpoint, e, control_early, control_late, timepoint)
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
) -> list[Design]:
    """Sweep enrichment from the unselected pool composition up to full."""
    grid = np.linspace(pi, 1.0, n_points)
    return [
        design(endpoint, float(e), pi, control_early, control_late, alpha, power, dropout, timepoint)
        for e in grid
    ]


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
