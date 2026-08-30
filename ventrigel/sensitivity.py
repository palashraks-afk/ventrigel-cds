"""
How far the conclusions survive their assumptions.

Simulation appears in this project only here, and only for the purpose it is
actually valid for: propagating uncertainty that is already present in the
published data, and mapping the range of assumptions over which a conclusion
holds. It is never used to invent observations or to manufacture a training
set. The distinction matters, because the two uses look similar in code and
are opposite in epistemics. Generating patients and then discovering a pattern
you built into the generator tells you nothing. Perturbing published estimates
within their own confidence structure and watching whether a conclusion breaks
tells you how much to trust the conclusion.

Three sources of doubt are handled separately, because they behave differently.

**Estimation uncertainty.** The subgroup means and SDs come from six to eight
patients. A mean of -7.6 mL with SEM 3.2 is not the truth; it is a draw from a
distribution around the truth. :func:`bootstrap_designs` propagates that with
a parametric bootstrap, sampling each stratum's true mean and variance from
their sampling distributions and re-solving the whole design each time. This
converts a point estimate of "48 patients" into an interval.

**Structural assumptions.** The control-arm drift is unmeasured, the real-world
prevalence of late patients is unknown, and Phase I effect estimates are
systematically optimistic. :func:`sweep_assumptions` grids over these rather
than sampling, because a reader should be able to look up the answer for their
own assumptions rather than accept an average over someone else's prior.

**Winner's curse.** The late subgroup was identified post hoc, in the same data
that produced its effect estimate. That guarantees the estimate is biased
upward: the subgroup was chosen partly because its noise happened to point the
right way. :func:`shrinkage_curve` reports how much of the enrichment advantage
survives as the effect is deflated toward zero, and this shrinkage is applied
by default in the headline results rather than being left as a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .power import Design, EnrichedPopulation, Stratum, n_per_arm_exact, n_screened
from .recovery import subgroup_effects
from .trial_data import ENDPOINTS

#: Default deflation applied to Phase I subgroup effects before they are used
#: to size a Phase II. A post hoc subgroup in a 15-patient single-arm trial is
#: the textbook setting for the winner's curse; empirical work on how effect
#: sizes shrink from early-phase to confirmatory trials supports discounting on
#: the order of 20-40%. 0.75 is the value used in the headline results and is
#: swept from 0.4 to 1.0 in the reported sensitivity analysis.
DEFAULT_SHRINKAGE = 0.75


# --------------------------------------------------------------------------
# Parametric bootstrap over estimation uncertainty
# --------------------------------------------------------------------------


def _draw_stratum(mean: float, sd: float, n: int, rng: np.random.Generator) -> tuple[float, float]:
    """Draw one plausible (true mean, true SD) for a stratum.

    Standard normal-model sampling distributions, which is what the published
    paired t-tests already assume:

        mu    ~ mean + (sd / sqrt(n)) * t_{n-1}
        sigma ~ sqrt((n - 1) * sd^2 / chi2_{n-1})
    """
    mu = mean + (sd / math.sqrt(n)) * rng.standard_t(n - 1)
    chi2 = rng.chisquare(n - 1)
    sigma = math.sqrt((n - 1) * sd**2 / chi2) if chi2 > 0 else sd
    return float(mu), float(sigma)


@dataclass(frozen=True)
class BootstrapResult:
    endpoint: str
    e: float
    pi: float
    n_draws: int
    n_feasible: int
    #: Fraction of draws where the design was solvable at all. A low value is
    #: itself a finding: it means the effect is not reliably nonzero.
    feasible_fraction: float
    n_total_median: float
    n_total_q10: float
    n_total_q90: float
    effect_median: float
    effect_q10: float
    effect_q90: float
    samples: np.ndarray


def bootstrap_designs(
    endpoint: str,
    e: float,
    pi: float,
    control_early: float = 0.0,
    control_late: float = 0.0,
    shrinkage: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    dropout: float = 0.0,
    n_draws: int = 4000,
    seed: int = 20190712,  # the trial's publication date, for a memorable seed
    timepoint: str = "6mo",
    cap: float = 1e6,
) -> BootstrapResult:
    """Re-solve the design across draws from the published sampling distributions."""
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)
    rng = np.random.default_rng(seed)

    n_totals: list[float] = []
    effects: list[float] = []
    for _ in range(n_draws):
        mu_e, sd_e = _draw_stratum(eff["early"]["mean"], eff["early"]["sd"], eff["early"]["n"], rng)
        mu_l, sd_l = _draw_stratum(eff["late"]["mean"], eff["late"]["sd"], eff["late"]["n"], rng)
        pop = EnrichedPopulation(
            early=Stratum("early", mu_e * shrinkage, sd_e, control_early),
            late=Stratum("late", mu_l * shrinkage, sd_l, control_late),
            e=e,
            lower_is_better=ep.lower_is_better,
        )
        effects.append(pop.effect)
        # A design is counted as infeasible when the drawn effect points the
        # wrong way or is too small to power at any realistic size. Discarding
        # these would bias the interval optimistically, so they are retained
        # as infinities and handled by the feasible fraction.
        if pop.effect <= 0:
            n_totals.append(math.inf)
            continue
        n_arm = n_per_arm_exact(pop.effect, pop.sd, alpha, power)
        if math.isfinite(n_arm):
            n_arm = math.ceil(n_arm / max(1e-9, 1.0 - dropout))
            n_totals.append(min(2.0 * n_arm, cap))
        else:
            n_totals.append(math.inf)

    arr = np.array(n_totals, dtype=float)
    eff_arr = np.array(effects, dtype=float)
    finite = arr[np.isfinite(arr)]
    n_feasible = int(finite.size)

    def q(a: np.ndarray, p: float) -> float:
        return float(np.quantile(a, p)) if a.size else float("nan")

    return BootstrapResult(
        endpoint=endpoint,
        e=e,
        pi=pi,
        n_draws=n_draws,
        n_feasible=n_feasible,
        feasible_fraction=n_feasible / n_draws,
        n_total_median=q(finite, 0.5),
        n_total_q10=q(finite, 0.10),
        n_total_q90=q(finite, 0.90),
        effect_median=q(eff_arr, 0.5),
        effect_q10=q(eff_arr, 0.10),
        effect_q90=q(eff_arr, 0.90),
        samples=arr,
    )


# --------------------------------------------------------------------------
# Grids over structural assumptions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AssumptionGrid:
    endpoint: str
    control_early_values: np.ndarray
    control_late_values: np.ndarray
    #: n_total[i, j] for control_early[i], control_late[j].
    n_total: np.ndarray
    #: Ratio of unselected to enriched n at each grid point.
    advantage: np.ndarray


def sweep_assumptions(
    endpoint: str,
    pi: float,
    control_early_values: np.ndarray | None = None,
    control_late_values: np.ndarray | None = None,
    shrinkage: float = DEFAULT_SHRINKAGE,
    alpha: float = 0.05,
    power: float = 0.80,
    dropout: float = 0.0,
    timepoint: str = "6mo",
    cap: float = 1e6,
) -> AssumptionGrid:
    """Grid the two control-arm drift parameters, the dominant unknowns.

    For LVESV the interesting region is ``control_early > 0`` (untreated early
    patients keep dilating) with ``control_late`` near zero (untreated late
    patients are relatively stable). If the enrichment advantage evaporates in
    that corner, the conclusion is an artifact of the single-arm design and the
    paper has to say so.
    """
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)
    ce = control_early_values if control_early_values is not None else np.linspace(0, 12, 13)
    cl = control_late_values if control_late_values is not None else np.linspace(-4, 4, 9)

    n_grid = np.full((ce.size, cl.size), np.inf)
    adv_grid = np.full((ce.size, cl.size), np.nan)

    for i, c_e in enumerate(ce):
        for j, c_l in enumerate(cl):
            def solve(e_level: float) -> float:
                pop = EnrichedPopulation(
                    early=Stratum("early", eff["early"]["mean"] * shrinkage, eff["early"]["sd"], c_e),
                    late=Stratum("late", eff["late"]["mean"] * shrinkage, eff["late"]["sd"], c_l),
                    e=e_level,
                    lower_is_better=ep.lower_is_better,
                )
                if pop.effect <= 0:
                    return math.inf
                n_arm = n_per_arm_exact(pop.effect, pop.sd, alpha, power)
                if not math.isfinite(n_arm):
                    return math.inf
                n_arm = math.ceil(n_arm / max(1e-9, 1.0 - dropout))
                return min(2.0 * n_arm, cap)

            n_enr = solve(1.0)
            n_uns = solve(pi)
            n_grid[i, j] = n_enr
            if math.isfinite(n_enr) and n_enr > 0:
                adv_grid[i, j] = (n_uns / n_enr) if math.isfinite(n_uns) else np.inf

    return AssumptionGrid(endpoint, ce, cl, n_grid, adv_grid)


@dataclass(frozen=True)
class ShrinkagePoint:
    shrinkage: float
    effect_enriched: float
    n_enriched: float
    n_unselected: float
    advantage: float


def shrinkage_curve(
    endpoint: str,
    pi: float,
    control_early: float = 0.0,
    control_late: float = 0.0,
    factors: np.ndarray | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    dropout: float = 0.0,
    timepoint: str = "6mo",
    cap: float = 1e6,
) -> list[ShrinkagePoint]:
    """How the enrichment advantage decays as Phase I effects are discounted."""
    ep = ENDPOINTS[endpoint]
    eff = subgroup_effects(endpoint, timepoint)
    fs = factors if factors is not None else np.linspace(0.4, 1.0, 13)
    out: list[ShrinkagePoint] = []
    for f in fs:
        def solve(e_level: float) -> tuple[float, float]:
            pop = EnrichedPopulation(
                early=Stratum("early", eff["early"]["mean"] * f, eff["early"]["sd"], control_early),
                late=Stratum("late", eff["late"]["mean"] * f, eff["late"]["sd"], control_late),
                e=e_level,
                lower_is_better=ep.lower_is_better,
            )
            if pop.effect <= 0:
                return pop.effect, math.inf
            n_arm = n_per_arm_exact(pop.effect, pop.sd, alpha, power)
            if not math.isfinite(n_arm):
                return pop.effect, math.inf
            n_arm = math.ceil(n_arm / max(1e-9, 1.0 - dropout))
            return pop.effect, min(2.0 * n_arm, cap)

        eff_enr, n_enr = solve(1.0)
        _, n_uns = solve(pi)
        adv = (n_uns / n_enr) if math.isfinite(n_enr) and n_enr > 0 else float("nan")
        out.append(ShrinkagePoint(float(f), eff_enr, n_enr, n_uns, adv))
    return out


def prevalence_curve(
    endpoint: str,
    pi_values: np.ndarray | None = None,
    shrinkage: float = DEFAULT_SHRINKAGE,
    **kw,
) -> list[tuple[float, float, float, float]]:
    """Enrichment advantage and screening burden as the eligible pool changes.

    Returns ``(pi, n_unselected, n_enriched, screens_per_enrolled)``.
    """
    from .power import design as _design

    pis = pi_values if pi_values is not None else np.linspace(0.15, 0.85, 15)
    out = []
    for pi in pis:
        u = _design(endpoint, float(pi), float(pi), **kw)
        e = _design(endpoint, 1.0, float(pi), **kw)
        out.append(
            (float(pi), u.n_total, e.n_total, n_screened(e.n_total, 1.0, float(pi)) / e.n_total)
        )
    return out
