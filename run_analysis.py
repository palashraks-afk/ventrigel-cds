"""
Reproduce every number and figure in the manuscript.

    python run_analysis.py

Writes machine-readable results to ``results/`` and figures to
``results/figures/``. Runs in well under a minute and takes no arguments,
because a reviewer should be able to check the whole analysis with one command.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ventrigel import __version__
from ventrigel.economics import CostModel, cost, optimal_enrichment, savings_vs_unselected
from ventrigel.power import design, enrichment_curve
from ventrigel.recovery import check_all_mixtures, check_all_p_values
from ventrigel.sensitivity import (
    DEFAULT_SHRINKAGE,
    bootstrap_designs,
    shrinkage_curve,
    sweep_assumptions,
)
from ventrigel.trial_data import (
    CANDIDATE_PRIMARY_ENDPOINTS,
    CITATION,
    ENDPOINTS,
    N_EARLY,
    N_LATE,
    TRIAL_ID,
)

RESULTS = Path("results")
FIGURES = RESULTS / "figures"

#: Fraction of the eligible pool treated more than 12 months post-MI. The
#: trial's own split, 8 of 15, is the only empirical anchor available; it is
#: swept from 0.15 to 0.85 in the sensitivity analysis because the trial
#: deliberately balanced the two arms and so does not estimate the natural
#: prevalence.
PI = N_LATE / (N_EARLY + N_LATE)

PRIMARY = "lvesv"
ALPHA, POWER, DROPOUT = 0.05, 0.80, 0.10


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fmt_n(x: float) -> str:
    if not math.isfinite(x):
        return "infeasible"
    if x >= 1e6:
        return ">1,000,000"
    return f"{x:,.0f}"


# --------------------------------------------------------------------------


def step_validation() -> dict:
    rule("[1/6]  Validating the transcription against the published tables")

    checks = check_all_p_values()
    real = [c for c in checks if c.excluded_reason is None]
    excluded = [c for c in checks if c.excluded_reason is not None]
    agree = sum(c.agrees for c in real)
    print(
        f"  Recomputed {len(checks)} published paired t-test p-values from mean, SEM and n."
    )
    print(f"  {agree}/{len(real)} agree at the precision the source prints.")
    for c in real:
        if not c.agrees:
            print(
                f"    disagreement  {c.endpoint} {c.timepoint} {c.group}: "
                f"recomputed {c.p_recomputed:.3f} vs published {c.p_published}"
            )
    if excluded:
        print(f"  {len(excluded)} cells excluded for documented reasons:")
        seen = set()
        for c in excluded:
            if c.excluded_reason in seen:
                continue
            seen.add(c.excluded_reason)
            print(f"    {c.endpoint}: {c.excluded_reason.splitlines()[0]}")

    mixtures = check_all_mixtures()
    print(
        "\n  Reconstructing each total-cohort mean and SD from its two subgroups\n"
        "  (law of total expectation and law of total variance):"
    )
    print(
        f"    {'endpoint':14s} {'mean rec':>9s} {'mean pub':>9s} "
        f"{'SD rec':>8s} {'SD pub':>8s} {'SD err':>7s}"
    )
    for m in mixtures:
        print(
            f"    {m.endpoint:14s} {m.mean_reconstructed:9.2f} {m.mean_published:9.2f} "
            f"{m.sd_reconstructed:8.2f} {m.sd_published:8.2f} {m.sd_rel_error * 100:6.1f}%"
        )
    med = float(np.median([m.sd_rel_error for m in mixtures]))
    print(f"    median SD reconstruction error: {med * 100:.1f}%")

    pd.DataFrame([c.__dict__ for c in checks]).to_csv(RESULTS / "validation_pvalues.csv", index=False)
    pd.DataFrame([m.__dict__ for m in mixtures]).to_csv(
        RESULTS / "validation_mixture.csv", index=False
    )
    return {
        "p_checks_total": len(checks),
        "p_checks_tested": len(real),
        "p_checks_agree": agree,
        "p_checks_excluded": len(excluded),
        "sd_reconstruction_median_rel_error": med,
    }


def step_effects() -> pd.DataFrame:
    rule("[2/6]  Subgroup separation in the published data")
    rows = []
    for key, ep in ENDPOINTS.items():
        t = ep.change_6mo
        if not {"early", "late"} <= t.keys():
            continue
        rows.append(
            {
                "endpoint": key,
                "label": ep.label,
                "unit": ep.unit,
                "source": ep.source_table,
                "early_change": t["early"].mean,
                "early_sd": t["early"].sd,
                "early_n": t["early"].n,
                "early_p": t["early"].p_published,
                "late_change": t["late"].mean,
                "late_sd": t["late"].sd,
                "late_n": t["late"].n,
                "late_p": t["late"].p_published,
                "total_change": t["total"].mean if "total" in t else np.nan,
                "separation_benefit_units": ep.benefit("late") - ep.benefit("early"),
                "opposite_direction": bool(
                    "total" in t and t["early"].mean * t["late"].mean < 0
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("separation_benefit_units", ascending=False)
    print(
        f"  {'endpoint':14s} {'early':>10s} {'late':>10s} {'total':>10s} "
        f"{'separation':>11s}  opposite"
    )
    for _, r in df.iterrows():
        print(
            f"  {r['endpoint']:14s} {r['early_change']:10.2f} {r['late_change']:10.2f} "
            f"{r['total_change']:10.2f} {r['separation_benefit_units']:11.2f}  "
            f"{'yes' if r['opposite_direction'] else ''}"
        )
    print(
        "\n  Where the two strata move in opposite directions, the total-cohort\n"
        "  mean is not a small effect. It is a cancellation."
    )
    df.to_csv(RESULTS / "subgroup_effects.csv", index=False)
    return df


def step_sample_sizes() -> pd.DataFrame:
    rule("[3/6]  Sample size, unselected versus enriched")
    print(
        f"  Two-arm parallel design, alpha={ALPHA}, power={POWER:.0%}, "
        f"{DROPOUT:.0%} dropout, noncentral t.\n"
        f"  Eligible-pool late fraction pi={PI:.3f}. No control drift and no\n"
        f"  shrinkage applied here; both are imposed in step 5."
    )
    rows = []
    for key in CANDIDATE_PRIMARY_ENDPOINTS:
        u = design(key, PI, PI, alpha=ALPHA, power=POWER, dropout=DROPOUT)
        e = design(key, 1.0, PI, alpha=ALPHA, power=POWER, dropout=DROPOUT)
        ratio = u.n_total / e.n_total if e.feasible and e.n_total else np.nan
        rows.append(
            {
                "endpoint": key,
                "effect_unselected": u.effect,
                "sd_unselected": u.sd,
                "n_unselected": u.n_total,
                "effect_enriched": e.effect,
                "sd_enriched": e.sd,
                "cohens_d_enriched": e.standardized_effect,
                "n_enriched": e.n_total,
                "screens_per_enrolled": e.screens_per_enrolled,
                "advantage": ratio,
                "unselected_favors_treatment": u.favors_treatment,
                "enriched_favors_treatment": e.favors_treatment,
            }
        )
    df = pd.DataFrame(rows)
    print(
        f"\n  {'endpoint':14s} {'eff unsel':>10s} {'N unsel':>12s} "
        f"{'eff enr':>8s} {'N enrich':>10s} {'d':>6s} {'ratio':>9s}"
    )
    for _, r in df.iterrows():
        rat = r["advantage"]
        rs = "worse" if rat < 1 else (f"{rat:,.0f}x" if math.isfinite(rat) else "inf")
        flag = "" if r["unselected_favors_treatment"] else "  <- wrong sign"
        print(
            f"  {r['endpoint']:14s} {r['effect_unselected']:10.2f} "
            f"{fmt_n(r['n_unselected']):>12s} {r['effect_enriched']:8.2f} "
            f"{fmt_n(r['n_enriched']):>10s} {r['cohens_d_enriched']:6.2f} {rs:>9s}{flag}"
        )
    print(
        "\n  Effects are signed so that positive means benefit. Sample size depends\n"
        "  on the square of the effect, so a design whose effect points the wrong\n"
        "  way still returns a finite n; that n is the size needed to detect a\n"
        "  difference of that magnitude in either direction, not the cost of\n"
        "  demonstrating benefit. The flagged rows have no benefit to detect."
    )
    print(
        "\n  Enrichment is not uniformly good. It is decisive for LVESV and viable\n"
        "  mass, where the strata oppose; worth about a factor of two for the\n"
        "  6-min walk, where both strata improve; and counterproductive for\n"
        "  ejection fraction, whose pooled signal is produced entirely by the\n"
        "  early stratum worsening -- a safety signal enrichment would erase."
    )
    df.to_csv(RESULTS / "sample_sizes.csv", index=False)
    return df


def step_uncertainty() -> dict:
    rule("[4/6]  Estimation uncertainty (parametric bootstrap)")
    print(
        "  Subgroup means and SDs come from 6-8 patients. Each draw samples a\n"
        "  plausible true mean and variance for both strata and re-solves the\n"
        "  design. Draws where the effect reverses sign are kept as infeasible."
    )
    out = {}
    print(
        f"\n  {'endpoint':14s} {'e':>5s} {'N median':>10s} {'80% interval':>18s} {'solvable':>9s}"
    )
    for key in ("lvesv", "viable_mass", "six_min_walk"):
        for e in (PI, 1.0):
            b = bootstrap_designs(
                key, e, PI, shrinkage=DEFAULT_SHRINKAGE, alpha=ALPHA, power=POWER, dropout=DROPOUT
            )
            out[f"{key}_e{e:.2f}"] = {
                "median": b.n_total_median,
                "q10": b.n_total_q10,
                "q90": b.n_total_q90,
                "feasible_fraction": b.feasible_fraction,
            }
            print(
                f"  {key:14s} {e:5.2f} {fmt_n(b.n_total_median):>10s} "
                f"{fmt_n(b.n_total_q10) + '-' + fmt_n(b.n_total_q90):>18s} "
                f"{b.feasible_fraction:8.0%}"
            )
    print(
        "\n  The solvable fraction is itself a result. For the enriched LVESV design\n"
        "  it states how often the late-subgroup effect keeps its sign once the\n"
        "  small sample it was measured in is taken seriously."
    )
    with open(RESULTS / "bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def step_assumptions() -> dict:
    rule("[5/6]  Structural assumptions: control drift and winner's curse")

    print(
        "  The Phase I was single-arm, so control-arm drift is unmeasured. If\n"
        "  untreated early patients dilate on their own, part of the early-versus-\n"
        "  late gap is natural history rather than effect modification.\n"
    )
    grid = sweep_assumptions(
        PRIMARY, PI, shrinkage=DEFAULT_SHRINKAGE, alpha=ALPHA, power=POWER, dropout=DROPOUT
    )
    print(f"  Enriched N for {PRIMARY} depends only on the late stratum, so it is")
    print("  constant in the early-arm drift. What the drift changes is the")
    print("  *unselected* comparator, and so the size of the advantage. Grid of")
    print("  N_unselected / N_enriched, rows = early-arm drift (mL), columns =")
    print("  late-arm drift (mL); 'none' means the unselected effect points at harm.")
    header = "        " + "".join(f"{c:>9.1f}" for c in grid.control_late_values)
    print(header)
    for i, ce in enumerate(grid.control_early_values):
        cells = ""
        for j in range(grid.control_late_values.size):
            a = grid.advantage[i, j]
            if not np.isfinite(a):
                cells += f"{'none':>9s}" if np.isnan(a) else f"{'>1e4':>9s}"
            elif a < 1:
                cells += f"{'worse':>9s}"
            else:
                cells += f"{a:>8,.0f}x"
        print(f"  {ce:6.1f}{cells}")
    print(
        "\n  Read the top-left corner as the hostile case: no early-arm drift means\n"
        "  the early stratum's +9.3 mL is entirely attributed to treatment, while\n"
        "  negative late-arm drift means untreated late patients would have improved\n"
        "  on their own. Even there the enriched design stays solvable; what\n"
        "  collapses is the claimed size of the advantage."
    )

    print("\n  Winner's curse: the late subgroup was chosen post hoc in these same data.")
    curve = shrinkage_curve(PRIMARY, PI, alpha=ALPHA, power=POWER, dropout=DROPOUT)
    print(
        f"  {'shrinkage':>10s} {'effect (mL)':>12s} {'N enriched':>12s} {'N unselected':>14s}"
    )
    for p in curve:
        print(
            f"  {p.shrinkage:10.2f} {p.effect_enriched:12.2f} "
            f"{fmt_n(p.n_enriched):>12s} {fmt_n(p.n_unselected):>14s}"
        )
    print(
        "\n  Discounting the Phase I effect by 40% still leaves an enriched trial of\n"
        "  roughly 300 patients. The unselected column stays infeasible throughout,\n"
        "  because shrinking both strata proportionally shrinks the cancellation\n"
        "  too: the pooled effect keeps pointing very slightly the wrong way."
    )

    np.savez(
        RESULTS / "assumption_grid.npz",
        control_early=grid.control_early_values,
        control_late=grid.control_late_values,
        n_total=grid.n_total,
        advantage=grid.advantage,
    )
    pd.DataFrame([p.__dict__ for p in curve]).to_csv(RESULTS / "shrinkage_curve.csv", index=False)
    return {"shrinkage_curve": [p.__dict__ for p in curve]}


def step_economics() -> dict:
    rule("[6/6]  Cost, and why maximal enrichment is not optimal")
    model = CostModel()
    print(f"  Unit costs (assumptions, not measurements): {model.label()}")
    print(
        "\n  Screening cost rises as 1/pi for a fully enriched trial, so the cheapest\n"
        "  design generally sits below full enrichment.\n"
    )
    out = {}
    for key in ("lvesv", "viable_mass"):
        sweep = enrichment_curve(
            key, PI, alpha=ALPHA, power=POWER, dropout=DROPOUT, n_points=81
        )
        best, costed = optimal_enrichment(sweep, model)
        unsel, best2, saving, frac = savings_vs_unselected(costed)
        print(f"  {key}:")
        if unsel.feasible and not unsel.design.favors_treatment:
            print(
                f"    unselected (e={PI:.2f}): pooled effect points toward harm "
                f"({unsel.design.effect:+.2f}); no benefit to power for"
            )
        elif unsel.feasible:
            print(
                f"    unselected (e={PI:.2f}): N={fmt_n(unsel.design.n_total)}, "
                f"${unsel.total_cost / 1e6:,.1f}M"
            )
        else:
            print(f"    unselected (e={PI:.2f}): infeasible at any realistic size")
        print(
            f"    cheapest   (e={best.design.e:.2f}): N={fmt_n(best.design.n_total)}, "
            f"screened={fmt_n(best.design.n_screened)}, ${best.total_cost / 1e6:,.1f}M, "
            f"{best.duration_months:.0f} months"
        )
        full = costed[-1]
        print(
            f"    full       (e=1.00): N={fmt_n(full.design.n_total)}, "
            f"screened={fmt_n(full.design.n_screened)}, ${full.total_cost / 1e6:,.1f}M"
        )
        out[key] = {
            "e_optimal": best.design.e,
            "n_optimal": best.design.n_total,
            "cost_optimal": best.total_cost,
            "cost_full": full.total_cost,
            "cost_unselected": unsel.total_cost,
            "saving_fraction": frac,
            "unselected_favors_treatment": unsel.design.favors_treatment,
        }
        pd.DataFrame(
            [
                {
                    "e": c.design.e,
                    "n_total": c.design.n_total,
                    "n_screened": c.design.n_screened,
                    "effect": c.design.effect,
                    "sd": c.design.sd,
                    "duration_months": c.duration_months,
                    "total_cost": c.total_cost,
                }
                for c in costed
            ]
        ).to_csv(RESULTS / f"cost_curve_{key}.csv", index=False)
    # At the trial's own 53% late fraction, screening is cheap and full
    # enrichment always wins. The screening penalty only bites when eligible
    # responders are rare, so the honest way to show the tradeoff is to sweep
    # prevalence and find where the optimum leaves the boundary.
    print(
        "\n  Whether an interior optimum exists depends on the sign of the effect in\n"
        "  the stratum being excluded, and the two candidate endpoints differ:\n"
        "    lvesv         excluded stratum has a NEGATIVE effect (+9.3 mL, i.e. harm)\n"
        "    six_min_walk  excluded stratum has a POSITIVE effect (+40.5 m, smaller)\n"
        "  Sweeping the eligible-pool composition makes the difference visible."
    )
    prevalence_rows = []
    for key in (PRIMARY, "six_min_walk"):
        print(
            f"\n    {key}:\n"
            f"    {'pi':>6s} {'e*':>6s} {'N':>7s} {'screened':>10s} {'months':>7s} {'cost':>10s}"
        )
        for pi in (0.60, 0.45, 0.30, 0.20, 0.12, 0.08, 0.05, 0.03):
            sweep = enrichment_curve(
                key, pi, alpha=ALPHA, power=POWER, dropout=DROPOUT, n_points=81
            )
            best, _ = optimal_enrichment(sweep, model)
            row = {
                "endpoint": key,
                "pi": pi,
                "e_star": best.design.e,
                "n_total": best.design.n_total,
                "n_screened": best.design.n_screened,
                "duration_months": best.duration_months,
                "total_cost": best.total_cost,
            }
            prevalence_rows.append(row)
            flag = "  <- over 10 years" if best.duration_months > 120 else ""
            print(
                f"    {pi:6.2f} {best.design.e:6.2f} {best.design.n_total:7.0f} "
                f"{best.design.n_screened:10,.0f} {best.duration_months:7.0f} "
                f"${best.total_cost / 1e6:9,.1f}M{flag}"
            )
    print(
        "\n  For LVESV the optimum never leaves full enrichment, because every early\n"
        "  patient admitted carries an effect of the wrong sign and strictly reduces\n"
        "  the pooled effect. The screening penalty still exists; it just appears as\n"
        "  duration rather than sample size, and duration is what actually kills the\n"
        "  design. At a 3% eligible-pool prevalence the enriched trial is still only\n"
        "  52 patients but takes over three decades to enroll at six sites.\n\n"
        "  For the 6-min walk, where excluded patients still improve, the optimum is\n"
        "  interior and moves off the boundary as responders get scarce. An analysis\n"
        "  that models enrolment but not screening would miss both behaviours and\n"
        "  conclude that tighter criteria are always better."
    )
    pd.DataFrame(prevalence_rows).to_csv(RESULTS / "prevalence_optimum.csv", index=False)
    out["prevalence_optimum"] = prevalence_rows

    with open(RESULTS / "economics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"VentriGel Phase II enrichment analysis, v{__version__}")
    print(f"Source trial: {TRIAL_ID}")
    print(CITATION)

    summary = {"version": __version__, "pi": PI, "alpha": ALPHA, "power": POWER, "dropout": DROPOUT}
    summary["validation"] = step_validation()
    step_effects()
    step_sample_sizes()
    summary["bootstrap"] = step_uncertainty()
    step_assumptions()
    summary["economics"] = step_economics()

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    rule("Done")
    print(f"  Results written to {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
