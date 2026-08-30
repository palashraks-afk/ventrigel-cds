"""
Recover the variance structure of the VentriGel trial from its published
summary statistics, and prove the recovery is faithful.

The Online Tables report every endpoint as ``mean (SEM)`` with n. Because
SEM = SD / sqrt(n), the sample standard deviation follows exactly:

    SD = SEM * sqrt(n)

That single identity supplies everything a power calculation needs. Nothing
has to be digitized off a figure and nothing has to be simulated.

Two independent checks confirm the recovered numbers are right.

**Check 1 - p-value reproduction.** The trial compared each follow-up with
baseline using a paired Student's t-test. Given the mean and SEM of the paired
change, that test statistic is t = mean / SEM on n - 1 degrees of freedom.
Recomputing every published p-value from the transcribed mean, SEM and n and
comparing against the printed value tests the transcription, the SD identity,
and the stated test all at once.

**Check 2 - mixture reconstruction.** The total cohort is the union of the two
subgroups, so its mean and variance must be recoverable from the subgroup means
and variances by the law of total expectation and the law of total variance:

    mean_total = sum_g w_g * mean_g
    var_total  = sum_g w_g * var_g  +  sum_g w_g * (mean_g - mean_total)^2
                 \\_____ within _____/    \\________ between ________/

If the recovered subgroup SDs are correct, this must reproduce the separately
published total-cohort SD. It is a strong test because the between-group term
is driven entirely by the subgroup separation, which is the quantity the whole
enrichment argument rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .trial_data import ENDPOINTS, Endpoint, Measurement


# --------------------------------------------------------------------------
# Check 1: reproduce the published paired t-test p-values
# --------------------------------------------------------------------------


#: Cells where the published p-value cannot be reproduced from the printed
#: mean and SEM, with the reason. Excluding a cell here is a claim about the
#: source that has itself been checked; see ``run_analysis.py`` for the
#: arithmetic behind each one.
PCHECK_EXCLUSIONS: dict[tuple[str, str, str], str] = {
    ("bnp", "6mo", "total"): (
        "Online Table 8 reports BNP as percent change but its paired t-test "
        "column tests absolute BNP. The published p = 0.63 corresponds to the "
        "absolute change of +17.2 pg/mL with a paired SEM near 35 pg/mL, which "
        "is plausible on a 294.8 pg/mL baseline. The percent-change mean and "
        "SEM describe a different quantity, so they cannot reproduce it."
    ),
    ("bnp", "6mo", "early"): "Same cause as the total-cohort BNP cell.",
    ("bnp", "6mo", "late"): "Same cause as the total-cohort BNP cell.",
    ("scar", "6mo", "total"): (
        "Unresolved discrepancy in the source. The printed 0.50 (1.9) with "
        "n = 12 gives p = 0.80, not the printed p = 0.58; reproducing 0.58 "
        "would require a SEM near 0.88. Recorded, not corrected. Scar fraction "
        "is not used in any headline result."
    ),
}


@dataclass(frozen=True)
class PValueCheck:
    endpoint: str
    timepoint: str
    group: str
    mean: float
    sem: float
    n: int
    t_stat: float
    p_recomputed: float
    p_published: float
    #: Absolute difference after rounding the recomputed value to the same
    #: number of decimals the source printed.
    abs_error: float
    agrees: bool
    #: Set when the cell is a documented exception rather than a real failure.
    excluded_reason: str | None = None


def _decimals(x: float) -> int:
    """Number of decimal places used to print a published p-value."""
    s = f"{x!r}"
    return len(s.split(".")[1]) if "." in s else 0


def check_p_value(
    endpoint: str, timepoint: str, group: str, m: Measurement
) -> PValueCheck | None:
    """Recompute one published paired t-test p-value from mean, SEM and n."""
    if m.p_published is None or m.sem == 0:
        return None
    t_stat = m.mean / m.sem
    df = m.n - 1
    p = float(2 * stats.t.sf(abs(t_stat), df))
    d = _decimals(m.p_published)
    err = abs(round(p, d) - m.p_published)
    # The source prints p-values to 2 significant figures at most, and rounds
    # the mean and SEM that feed them. Agreement is judged at the precision
    # actually published, with two units of slack in the last place to absorb
    # rounding that has already occurred in three separate printed numbers.
    tol = 2.5 * 10 ** (-d)
    reason = PCHECK_EXCLUSIONS.get((endpoint, timepoint, group))
    return PValueCheck(
        endpoint=endpoint,
        timepoint=timepoint,
        group=group,
        mean=m.mean,
        sem=m.sem,
        n=m.n,
        t_stat=t_stat,
        p_recomputed=p,
        p_published=m.p_published,
        abs_error=err,
        agrees=bool(err <= tol),
        excluded_reason=reason,
    )


def check_all_p_values() -> list[PValueCheck]:
    """Run check 1 across every published cell that carries a p-value."""
    out: list[PValueCheck] = []
    for key, ep in ENDPOINTS.items():
        for tp, table in (("3mo", ep.change_3mo), ("6mo", ep.change_6mo)):
            for group, m in table.items():
                chk = check_p_value(key, tp, group, m)
                if chk is not None:
                    out.append(chk)
    return out


# --------------------------------------------------------------------------
# Check 2: reconstruct the total cohort from its subgroups
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureCheck:
    endpoint: str
    label: str
    unit: str
    w_early: float
    mean_early: float
    mean_late: float
    sd_early: float
    sd_late: float
    mean_reconstructed: float
    mean_published: float
    mean_abs_error: float
    sd_within: float
    sd_between: float
    sd_reconstructed: float
    sd_published: float
    sd_rel_error: float
    #: Difference in benefit between the subgroups, signed so that positive
    #: means the late subgroup did better. This is the enrichment signal.
    separation: float


def mixture_moments(
    means: np.ndarray, sds: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    """Mean and SD of a finite mixture, with the variance decomposed.

    Returns ``(mean, sd_within, sd_between, sd_total)`` where the within and
    between components are the two halves of the law of total variance.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = float(np.sum(w * means))
    var_within = float(np.sum(w * sds**2))
    var_between = float(np.sum(w * (means - mean) ** 2))
    return mean, float(np.sqrt(var_within)), float(np.sqrt(var_between)), float(
        np.sqrt(var_within + var_between)
    )


def check_mixture(ep: Endpoint, timepoint: str = "6mo") -> MixtureCheck | None:
    """Run check 2 for one endpoint."""
    table = ep.change_6mo if timepoint == "6mo" else ep.change_3mo
    if not {"total", "early", "late"} <= table.keys():
        return None
    early, late, total = table["early"], table["late"], table["total"]

    # Weight the subgroups by the n that actually contributed at this visit,
    # not by enrolment, because CMR follow-up was incomplete for some patients.
    n_e, n_l = early.n, late.n
    w = np.array([n_e, n_l], dtype=float)
    means = np.array([early.mean, late.mean])
    sds = np.array([early.sd, late.sd])

    mean_rec, sd_within, sd_between, sd_rec = mixture_moments(means, sds, w)

    sign = -1.0 if ep.lower_is_better else 1.0
    separation = sign * (late.mean - early.mean)

    return MixtureCheck(
        endpoint=ep.key,
        label=ep.label,
        unit=ep.unit,
        w_early=float(n_e / (n_e + n_l)),
        mean_early=early.mean,
        mean_late=late.mean,
        sd_early=early.sd,
        sd_late=late.sd,
        mean_reconstructed=mean_rec,
        mean_published=total.mean,
        mean_abs_error=abs(mean_rec - total.mean),
        sd_within=sd_within,
        sd_between=sd_between,
        sd_reconstructed=sd_rec,
        sd_published=total.sd,
        sd_rel_error=abs(sd_rec - total.sd) / total.sd if total.sd else float("nan"),
        separation=separation,
    )


def check_all_mixtures(timepoint: str = "6mo") -> list[MixtureCheck]:
    out = []
    for ep in ENDPOINTS.values():
        chk = check_mixture(ep, timepoint)
        if chk is not None:
            out.append(chk)
    return out


# --------------------------------------------------------------------------
# Convenience accessors used by the power and sensitivity modules
# --------------------------------------------------------------------------


def recovered_sd(endpoint: str, group: str, timepoint: str = "6mo") -> float:
    ep = ENDPOINTS[endpoint]
    table = ep.change_6mo if timepoint == "6mo" else ep.change_3mo
    return table[group].sd


def subgroup_effects(endpoint: str, timepoint: str = "6mo") -> dict[str, dict[str, float]]:
    """Mean change, recovered SD and n for each group, one endpoint."""
    ep = ENDPOINTS[endpoint]
    table = ep.change_6mo if timepoint == "6mo" else ep.change_3mo
    return {
        g: {"mean": m.mean, "sd": m.sd, "n": m.n, "benefit": (-m.mean if ep.lower_is_better else m.mean)}
        for g, m in table.items()
    }
