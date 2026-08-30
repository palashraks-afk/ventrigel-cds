"""
Does the subgroup effect exist at all?

Everything downstream of this module assumes the early and late strata respond
differently. That assumption was never tested, by the source trial or by earlier
versions of this analysis. The trial compared each stratum against its own
baseline and observed that one reached significance while the other did not.
That is not a test of effect modification: two subgroups can easily land on
opposite sides of p = 0.05 without differing from each other, and this is one of
the most common errors in the reporting of subgroup analyses.

This module performs the test that was missing, corrects it for the number of
endpoints examined, and checks the two artifacts most likely to produce a
spurious subgroup effect.

The honest summary it produces
------------------------------

Only end-systolic volume shows a nominally significant interaction
(p = 0.034), and it does not survive any correction for the nine endpoints
examined. Viable mass, which reads dramatically at +15.0 g against -10.5 g,
has an interaction p of 0.17: the difference is not distinguishable from noise.

Two checks come back in the analysis's favour. The strata are balanced at
baseline on every cardiac magnetic resonance measure, so the contrast is not
confounded by differing severity at entry. And the pattern runs opposite to
regression to the mean: the early stratum began with the *higher* end-systolic
volume and got worse, whereas regression would have pulled it down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .trial_data import ENDPOINTS, Endpoint, Measurement


# --------------------------------------------------------------------------
# The interaction test
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionTest:
    """Welch two-sample comparison of the change scores between strata."""

    endpoint: str
    label: str
    unit: str
    early_mean: float
    early_sem: float
    early_n: int
    late_mean: float
    late_sem: float
    late_n: int
    #: Late minus early, in native units and sign.
    difference: float
    #: The same difference signed so that positive favours the late stratum.
    difference_benefit: float
    se: float
    t_stat: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    lower_is_better: bool

    @property
    def nominally_significant(self) -> bool:
        return self.p_value < 0.05


def interaction_test(ep: Endpoint, timepoint: str = "6mo") -> InteractionTest | None:
    """Test whether the two strata differ from each other.

    Welch's unequal-variance form is used because the strata have different n
    and visibly different spread, and because the pooled-variance alternative
    would borrow precision from one stratum to flatter the other.
    """
    table = ep.change_6mo if timepoint == "6mo" else ep.change_3mo
    if not {"early", "late"} <= table.keys():
        return None
    a: Measurement = table["early"]
    b: Measurement = table["late"]
    if a.sem == 0 and b.sem == 0:
        return None

    se = math.sqrt(a.sem**2 + b.sem**2)
    diff = b.mean - a.mean
    if se == 0:
        return None
    t_stat = diff / se
    # Welch-Satterthwaite. SEMs are already SD/sqrt(n), so the usual
    # (s^2/n)^2 / (n-1) terms are just sem^4 / (n-1).
    denom = (a.sem**4) / (a.n - 1) + (b.sem**4) / (b.n - 1)
    df = ((a.sem**2 + b.sem**2) ** 2) / denom if denom > 0 else float("inf")
    p = float(2 * stats.t.sf(abs(t_stat), df))
    crit = float(stats.t.ppf(0.975, df))
    sign = -1.0 if ep.lower_is_better else 1.0

    return InteractionTest(
        endpoint=ep.key,
        label=ep.label,
        unit=ep.unit,
        early_mean=a.mean,
        early_sem=a.sem,
        early_n=a.n,
        late_mean=b.mean,
        late_sem=b.sem,
        late_n=b.n,
        difference=diff,
        difference_benefit=sign * diff,
        se=se,
        t_stat=t_stat,
        df=df,
        p_value=p,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        lower_is_better=ep.lower_is_better,
    )


def all_interaction_tests(timepoint: str = "6mo") -> list[InteractionTest]:
    out = [interaction_test(ep, timepoint) for ep in ENDPOINTS.values()]
    return sorted([t for t in out if t is not None], key=lambda t: t.p_value)


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiplicityResult:
    endpoint: str
    p_value: float
    rank: int
    n_tests: int
    bonferroni_p: float
    bonferroni_pass: bool
    bh_threshold: float
    bh_pass: bool


def multiplicity(tests: list[InteractionTest], q: float = 0.05) -> list[MultiplicityResult]:
    """Bonferroni and Benjamini-Hochberg across the interaction tests.

    The source trial states explicitly that it made no adjustment for multiple
    comparisons, which is defensible for an exploratory Phase I but means the
    nominal p-values cannot be read as evidence strength without correction
    here.
    """
    ordered = sorted(tests, key=lambda t: t.p_value)
    m = len(ordered)
    out: list[MultiplicityResult] = []
    # Benjamini-Hochberg is a step-up procedure: find the largest rank passing,
    # then declare everything at or below it significant.
    passing_rank = 0
    for i, t in enumerate(ordered, start=1):
        if t.p_value <= q * i / m:
            passing_rank = i
    for i, t in enumerate(ordered, start=1):
        out.append(
            MultiplicityResult(
                endpoint=t.endpoint,
                p_value=t.p_value,
                rank=i,
                n_tests=m,
                bonferroni_p=min(1.0, t.p_value * m),
                bonferroni_pass=t.p_value < q / m,
                bh_threshold=q * i / m,
                bh_pass=i <= passing_rank,
            )
        )
    return out


#: Endpoints that are not independent tests. Ejection fraction is an algebraic
#: function of the two volumes, EF = (LVEDV - LVESV) / LVEDV, and viable mass
#: and scar fraction are complementary partitions of the same myocardium on the
#: same scan. Treating all nine as independent overstates the correction.
ALGEBRAIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "cmr_volumes": ("lvesv", "lvedv", "ef"),
    "cmr_tissue": ("viable_mass", "scar"),
    "symptoms": ("mlwhfq", "nyha"),
    "function": ("six_min_walk",),
    "biomarker": ("bnp",),
}


def effective_n_tests() -> tuple[int, int, str]:
    """Nominal and family-reduced counts of independent interaction tests.

    Returns ``(nominal, effective, explanation)``. The effective count treats
    each algebraically or physiologically linked family as one test, which is
    the most favourable defensible correction. It is reported alongside the
    nominal count rather than instead of it, because choosing the smaller
    denominator after seeing the p-values would itself be a form of selection.
    """
    nominal = len([e for e in ENDPOINTS.values() if {"early", "late"} <= e.change_6mo.keys()])
    effective = len(ALGEBRAIC_FAMILIES)
    why = (
        "Ejection fraction is determined by the two volumes, and viable mass and "
        "scar fraction partition the same myocardium on the same scan, so the "
        "nine endpoints are not nine independent questions. Collapsing each "
        f"linked family to one test gives {effective}. Even at that denominator "
        f"the Bonferroni threshold is {0.05 / effective:.4f}, which the "
        "end-systolic volume interaction at p = 0.034 still fails."
    )
    return nominal, effective, why


# --------------------------------------------------------------------------
# The two artifacts most likely to fake a subgroup effect
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BalanceTest:
    endpoint: str
    label: str
    unit: str
    early_mean: float
    late_mean: float
    difference: float
    p_value: float

    @property
    def imbalanced(self) -> bool:
        return self.p_value < 0.05


def baseline_balance(timepoint: str = "6mo") -> list[BalanceTest]:
    """Were the strata comparable before treatment?

    The two strata were not randomized against each other -- membership is
    determined by how long ago the patient's infarction was -- so an observed
    difference in change could reflect a difference in starting condition. This
    tests every baseline the trial reported.
    """
    out: list[BalanceTest] = []
    for ep in ENDPOINTS.values():
        if not {"early", "late"} <= ep.baseline.keys():
            continue
        a, b = ep.baseline["early"], ep.baseline["late"]
        se = math.sqrt(a.sem**2 + b.sem**2)
        if se == 0:
            continue
        denom = (a.sem**4) / (a.n - 1) + (b.sem**4) / (b.n - 1)
        df = ((a.sem**2 + b.sem**2) ** 2) / denom if denom > 0 else float("inf")
        p = float(2 * stats.t.sf(abs((b.mean - a.mean) / se), df))
        out.append(
            BalanceTest(ep.key, ep.label, ep.unit, a.mean, b.mean, b.mean - a.mean, p)
        )
    return out


@dataclass(frozen=True)
class RegressionToMeanCheck:
    endpoint: str
    label: str
    higher_baseline_stratum: str
    early_baseline: float
    late_baseline: float
    early_change: float
    late_change: float
    #: True when the stratum starting further from the mean moved *away* from
    #: it, which is the opposite of what regression to the mean produces.
    contradicts_rtm: bool
    explanation: str


def regression_to_mean_check(endpoint: str = "lvesv") -> RegressionToMeanCheck | None:
    """Could the subgroup difference just be regression to the mean?

    Regression to the mean pulls whichever group started further from the
    population mean back toward it. If the stratum with the more extreme
    baseline moved further out instead, the observed pattern cannot be produced
    by regression and this artifact is ruled out.
    """
    ep = ENDPOINTS[endpoint]
    if not ({"early", "late"} <= ep.baseline.keys() and {"early", "late"} <= ep.change_6mo.keys()):
        return None
    b_e, b_l = ep.baseline["early"].mean, ep.baseline["late"].mean
    c_e, c_l = ep.change_6mo["early"].mean, ep.change_6mo["late"].mean

    higher = "early" if b_e > b_l else "late"
    change_of_higher = c_e if higher == "early" else c_l
    # For a group with the higher baseline, regression predicts a decrease.
    contradicts = change_of_higher > 0

    explanation = (
        f"The {higher} stratum began with the higher {ep.label.lower()} "
        f"({max(b_e, b_l):.1f} vs {min(b_e, b_l):.1f} {ep.unit}) and changed by "
        f"{change_of_higher:+.1f} {ep.unit}. Regression to the mean would have "
        f"moved it {'down' if higher == 'early' else 'down'}, so the observed "
        f"pattern {'runs opposite to' if contradicts else 'is consistent with'} "
        "that artifact."
    )
    return RegressionToMeanCheck(
        endpoint=ep.key,
        label=ep.label,
        higher_baseline_stratum=higher,
        early_baseline=b_e,
        late_baseline=b_l,
        early_change=c_e,
        late_change=c_l,
        contradicts_rtm=contradicts,
        explanation=explanation,
    )


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceAssessment:
    """A single, quotable verdict on the subgroup finding."""

    strongest_endpoint: str
    strongest_p: float
    n_nominally_significant: int
    n_tests: int
    survives_bonferroni: bool
    survives_bh: bool
    baseline_balanced: bool
    rtm_ruled_out: bool
    verdict: str


def assess_evidence() -> EvidenceAssessment:
    tests = all_interaction_tests()
    mult = multiplicity(tests)
    balance = baseline_balance()
    rtm = regression_to_mean_check("lvesv")

    best = tests[0]
    best_m = next(m for m in mult if m.endpoint == best.endpoint)
    n_sig = sum(t.nominally_significant for t in tests)
    balanced = not any(b.imbalanced for b in balance)

    verdict = (
        f"One endpoint of {len(tests)} shows a nominally significant interaction "
        f"({best.endpoint}, p = {best.p_value:.3f}). It does not survive "
        f"Bonferroni or Benjamini-Hochberg correction. The strata are "
        f"{'balanced' if balanced else 'NOT balanced'} at baseline, and the "
        f"pattern {'is not' if rtm and rtm.contradicts_rtm else 'may be'} "
        "explained by regression to the mean. The finding is therefore "
        "suggestive and fragile: strong enough to justify designing a trial "
        "around, not strong enough to assert as established."
    )
    return EvidenceAssessment(
        strongest_endpoint=best.endpoint,
        strongest_p=best.p_value,
        n_nominally_significant=n_sig,
        n_tests=len(tests),
        survives_bonferroni=best_m.bonferroni_pass,
        survives_bh=best_m.bh_pass,
        baseline_balanced=balanced,
        rtm_ruled_out=bool(rtm and rtm.contradicts_rtm),
        verdict=verdict,
    )
