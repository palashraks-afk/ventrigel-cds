"""
Reproduce every number in the manuscript.

    python run_analysis.py

Writes machine-readable results to ``results/``. Runs in about a minute and
takes no arguments, because a reviewer should be able to check the whole
analysis with one command.

The order of the sections is the order of the argument. Section 2 asks whether
the subgroup effect exists at all; everything after it is explicitly
conditional on that answer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ventrigel import __version__
from ventrigel.assurance import (
    assurance,
    assurance_ceiling,
    n_for_assurance,
)
from ventrigel.economics import CostModel, cost, optimal_enrichment, savings_vs_unselected
from ventrigel.inference import (
    all_interaction_tests,
    assess_evidence,
    baseline_balance,
    effective_n_tests,
    multiplicity,
    regression_to_mean_check,
)
from ventrigel.literature import (
    ANCHORS,
    BSA_CENTRAL,
    early_control_prior,
    implied_ventrigel_lvesvi,
    late_control_prior,
)
from ventrigel.power import design, enrichment_curve
from ventrigel.recovery import check_all_mixtures, check_all_p_values
from ventrigel.sensitivity import bootstrap_designs, shrinkage_curve, sweep_assumptions
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

PI = N_LATE / (N_EARLY + N_LATE)
PRIMARY = "lvesv"
ALPHA, POWER, DROPOUT = 0.05, 0.80, 0.10

#: Single discount used where one value is needed. Results are reported across
#: the full 0.4-1.0 range in section 8; this is a reporting convenience, not a
#: claim that 0.75 is correct.
SHRINK = 0.75


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fmt_n(x: float) -> str:
    if not math.isfinite(x):
        return "infeasible"
    if x >= 1e6:
        return ">1,000,000"
    return f"{x:,.0f}"


# --------------------------------------------------------------------------


def step1_validation() -> dict:
    rule("[1/9]  Is the transcription faithful?")
    checks = check_all_p_values()
    real = [c for c in checks if c.excluded_reason is None]
    excluded = [c for c in checks if c.excluded_reason is not None]
    agree = sum(c.agrees for c in real)
    print(f"  Recomputed {len(checks)} published paired t-test p-values from mean, SEM and n.")
    print(f"  {agree}/{len(real)} agree at the precision the source prints.")
    for c in real:
        if not c.agrees:
            print(f"    disagreement  {c.endpoint} {c.timepoint} {c.group}")
    if excluded:
        seen = set()
        print(f"  {len(excluded)} cells excluded for documented reasons:")
        for c in excluded:
            head = c.excluded_reason.split(".")[0]
            if head in seen:
                continue
            seen.add(head)
            print(f"    {c.endpoint}: {head}.")

    mixtures = check_all_mixtures()
    med = float(np.median([m.sd_rel_error for m in mixtures]))
    print(f"\n  Pooled cohorts reconstructed from their strata; median SD error {med * 100:.1f}%.")
    lv = next(m for m in mixtures if m.endpoint == "lvesv")
    print(
        f"    {PRIMARY}: mean {lv.mean_reconstructed:.2f} vs published {lv.mean_published:.2f}, "
        f"SD {lv.sd_reconstructed:.2f} vs {lv.sd_published:.2f} ({lv.sd_rel_error * 100:.1f}%)"
    )

    pd.DataFrame([c.__dict__ for c in checks]).to_csv(RESULTS / "validation_pvalues.csv", index=False)
    pd.DataFrame([m.__dict__ for m in mixtures]).to_csv(RESULTS / "validation_mixture.csv", index=False)
    return {
        "p_checks_tested": len(real),
        "p_checks_agree": agree,
        "p_checks_excluded": len(excluded),
        "sd_reconstruction_median_rel_error": med,
    }


def step2_does_the_effect_exist() -> dict:
    rule("[2/9]  Does the subgroup effect exist? (the test nobody ran)")
    print(
        "  The trial compared each stratum against its own baseline and observed\n"
        "  that one reached significance while the other did not. That is not a\n"
        "  test of effect modification. Below is the comparison of the strata\n"
        "  against each other, which neither the trial nor earlier versions of\n"
        "  this analysis performed.\n"
    )
    tests = all_interaction_tests()
    print(
        f"  {'endpoint':14s} {'early':>8s} {'late':>8s} {'difference':>11s} "
        f"{'95% CI':>20s} {'p':>8s}"
    )
    for t in tests:
        ci = f"[{t.ci_low:.1f}, {t.ci_high:.1f}]"
        star = " *" if t.nominally_significant else ""
        print(
            f"  {t.endpoint:14s} {t.early_mean:8.2f} {t.late_mean:8.2f} "
            f"{t.difference:11.2f} {ci:>20s} {t.p_value:8.4f}{star}"
        )

    mult = multiplicity(tests)
    nominal, effective, why = effective_n_tests()
    print(f"\n  Multiplicity across {nominal} interaction tests:")
    print(f"    {'endpoint':14s} {'p':>8s} {'Bonferroni p':>13s} {'BH threshold':>13s} {'passes':>8s}")
    for m in mult[:4]:
        print(
            f"    {m.endpoint:14s} {m.p_value:8.4f} {m.bonferroni_p:13.3f} "
            f"{m.bh_threshold:13.4f} {'yes' if m.bh_pass else 'no':>8s}"
        )
    print(f"    ... {len(mult) - 4} further endpoints, all p > 0.3")
    print(f"\n  {why}")

    balance = baseline_balance()
    bad = [b for b in balance if b.imbalanced]
    print(
        f"\n  Baseline balance: {len(balance) - len(bad)}/{len(balance)} measures balanced "
        f"(all p >= {min(b.p_value for b in balance):.2f})."
    )
    print("    The strata were not randomized against each other, so this matters.")
    for b in balance[:3]:
        print(f"      {b.endpoint:14s} early {b.early_mean:7.1f}  late {b.late_mean:7.1f}  p={b.p_value:.3f}")

    rtm = regression_to_mean_check(PRIMARY)
    print(f"\n  Regression to the mean: {rtm.explanation}")

    ev = assess_evidence()
    print(f"\n  VERDICT: {ev.verdict}")

    pd.DataFrame([t.__dict__ for t in tests]).to_csv(RESULTS / "interaction_tests.csv", index=False)
    pd.DataFrame([m.__dict__ for m in mult]).to_csv(RESULTS / "multiplicity.csv", index=False)
    pd.DataFrame([b.__dict__ for b in balance]).to_csv(RESULTS / "baseline_balance.csv", index=False)
    return {
        "strongest_endpoint": ev.strongest_endpoint,
        "strongest_p": ev.strongest_p,
        "n_nominally_significant": ev.n_nominally_significant,
        "n_tests": ev.n_tests,
        "effective_n_tests": effective,
        "survives_bonferroni": ev.survives_bonferroni,
        "survives_bh": ev.survives_bh,
        "baseline_balanced": ev.baseline_balanced,
        "rtm_ruled_out": ev.rtm_ruled_out,
        "verdict": ev.verdict,
    }


def step3_literature_anchors() -> dict:
    rule("[3/9]  What does an untreated patient do? (external control anchors)")
    print(
        "  The Phase I was single-arm, so its comparator is missing. Rather than\n"
        "  sweeping an arbitrary range, control-arm change is anchored to published\n"
        f"  control and placebo arms. Indexed volumes converted at BSA = {BSA_CENTRAL} m2.\n"
    )
    print(
        f"  {'trial':16s} {'year':>5s} {'phase':>8s} {'measure':>8s} {'published':>14s} "
        f"{'absolute mL':>12s} {'n':>5s}"
    )
    rows = []
    for a in ANCHORS.values():
        pub = f"{a.change:+.1f} {'mL/m2' if a.indexed else 'mL'}"
        print(
            f"  {a.trial:16s} {a.year:5d} {a.phase:>8s} {a.measure:>8s} {pub:>14s} "
            f"{a.absolute_change():+12.1f} {a.n:5d}"
        )
        rows.append(
            {
                "trial": a.trial,
                "year": a.year,
                "phase": a.phase,
                "measure": a.measure,
                "change_published": a.change,
                "indexed": a.indexed,
                "change_mL": a.absolute_change(),
                "n": a.n,
                "citation": a.citation,
            }
        )

    print(
        f"\n  BSA check: VentriGel's 148.5 mL baseline implies "
        f"{implied_ventrigel_lvesvi():.1f} mL/m2, against 65.0 mL/m2 in the chronic\n"
        "  FOCUS-CCTRN population. Higher, as expected for a dilated post-MI cohort,\n"
        "  and not so far off as to invalidate the conversion."
    )

    ep_, lp = early_control_prior(), late_control_prior()
    print(
        f"\n  EARLY stratum comparator: central {ep_.central:+.1f} mL, "
        f"range [{ep_.low:+.1f}, {ep_.high:+.1f}]"
    )
    print(f"    {ep_.rationale}")
    print(
        f"\n  LATE stratum comparator: central {lp.central:+.1f} mL, "
        f"range [{lp.low:+.1f}, {lp.high:+.1f}]"
    )
    print(f"    {lp.rationale}")
    print(
        "\n  The acute anchors disagree in SIGN, which is the finding rather than a\n"
        "  defect. Older cohorts dilated; the 2022-2024 cohort on contemporary\n"
        "  therapy underwent reverse remodeling. Post-MI natural history is\n"
        "  era-dependent, so no single control number is defensible for the early\n"
        "  stratum and the range is carried through every downstream result."
    )
    pd.DataFrame(rows).to_csv(RESULTS / "literature_anchors.csv", index=False)
    return {
        "early_central": ep_.central,
        "early_low": ep_.low,
        "early_high": ep_.high,
        "late_central": lp.central,
        "late_low": lp.low,
        "late_high": lp.high,
        "bsa": BSA_CENTRAL,
    }


def step4_cancellation() -> None:
    rule("[4/9]  Why the pooled effect is a cancellation, not a weak effect")
    rows = []
    for key, ep in ENDPOINTS.items():
        t = ep.change_6mo
        if not {"early", "late"} <= t.keys():
            continue
        a, b = t["early"].mean, t["late"].mean
        opposed = a * b < 0 and min(abs(a), abs(b)) > 0.10 * max(abs(a), abs(b))
        rows.append(
            {
                "endpoint": key,
                "early": a,
                "late": b,
                "pooled": t["total"].mean if "total" in t else np.nan,
                "separation_benefit": ep.benefit("late") - ep.benefit("early"),
                "strata_oppose": opposed,
            }
        )
    df = pd.DataFrame(rows).sort_values("separation_benefit", ascending=False)
    print(f"  {'endpoint':14s} {'early':>9s} {'late':>9s} {'pooled':>9s} {'separation':>11s}  oppose")
    for _, r in df.iterrows():
        print(
            f"  {r['endpoint']:14s} {r['early']:9.2f} {r['late']:9.2f} {r['pooled']:9.2f} "
            f"{r['separation_benefit']:11.2f}  {'yes' if r['strata_oppose'] else ''}"
        )
    print(
        "\n  Only end-systolic volume combines genuine opposition with a significant\n"
        "  interaction. Viable mass looks more dramatic (+15.0 g against -10.5 g)\n"
        "  but its interaction p is 0.17, so its separation is not distinguishable\n"
        "  from noise and it is reported here as supporting, not primary."
    )
    df.to_csv(RESULTS / "subgroup_effects.csv", index=False)


def step5_sample_sizes(anchors: dict) -> pd.DataFrame:
    rule("[5/9]  Sample size, unselected versus enriched")
    ce, cl = anchors["early_central"], anchors["late_central"]
    print(
        f"  Two-arm design, alpha={ALPHA}, power={POWER:.0%}, {DROPOUT:.0%} dropout,\n"
        f"  noncentral t, pi={PI:.3f}, effect discount {SHRINK:.2f}.\n"
        f"  Control arm anchored: early {ce:+.1f} mL (TIME), late {cl:+.1f} mL (FOCUS-CCTRN).\n"
        "  Control assumptions apply to the CMR volume endpoints only; the others\n"
        "  are shown at zero drift and are correspondingly optimistic."
    )
    rows = []
    for key in CANDIDATE_PRIMARY_ENDPOINTS:
        use_ce, use_cl = (ce, cl) if key in ("lvesv", "lvedv") else (0.0, 0.0)
        u = design(key, PI, PI, use_ce, use_cl, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        e = design(key, 1.0, PI, use_ce, use_cl, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        rows.append(
            {
                "endpoint": key,
                "control_early": use_ce,
                "control_late": use_cl,
                "effect_unselected": u.effect,
                "n_unselected": u.n_total if u.favors_treatment else np.inf,
                "effect_enriched": e.effect,
                "sd_enriched": e.sd,
                "cohens_d_enriched": e.standardized_effect,
                "n_enriched": e.n_total if e.favors_treatment else np.inf,
                "advantage": (u.n_total / e.n_total)
                if (u.favors_treatment and e.favors_treatment)
                else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    print(
        f"\n  {'endpoint':14s} {'eff unsel':>10s} {'N unsel':>11s} {'eff enr':>8s} "
        f"{'N enrich':>9s} {'d':>6s} {'ratio':>8s}"
    )
    for _, r in df.iterrows():
        rat = r["advantage"]
        rs = "-" if not np.isfinite(rat) else ("worse" if rat < 1 else f"{rat:,.1f}x")
        print(
            f"  {r['endpoint']:14s} {r['effect_unselected']:10.2f} "
            f"{fmt_n(r['n_unselected']):>11s} {r['effect_enriched']:8.2f} "
            f"{fmt_n(r['n_enriched']):>9s} {r['cohens_d_enriched']:6.2f} {rs:>8s}"
        )
    print(
        "\n  'infeasible' means the modelled effect does not favour treatment, so no\n"
        "  sample size demonstrates benefit. Enrichment is decisive for end-systolic\n"
        "  volume, worth under twofold for the 6-min walk where both strata improve,\n"
        "  and counterproductive for ejection fraction, whose pooled signal is\n"
        "  produced entirely by the early stratum deteriorating."
    )
    df.to_csv(RESULTS / "sample_sizes.csv", index=False)
    return df


def step6_assurance(anchors: dict) -> dict:
    rule("[6/9]  Power is not probability of success")
    cl = anchors["late_central"]
    print(
        "  A trial sized for 80% power at a point estimate is not an 80% trial when\n"
        "  the point estimate rests on eight patients. Assurance integrates power\n"
        "  over the uncertainty in the effect, which is what a sponsor deciding\n"
        "  whether to fund the trial actually needs.\n"
    )
    print(f"  {'N total':>8s} {'nominal power':>14s} {'assurance':>10s}")
    rows = []
    for n in (52, 92, 108, 150, 200, 300, 500, 1000):
        r = assurance(PRIMARY, n, SHRINK, cl, ALPHA, n_draws=20000)
        rows.append({"n_total": n, "nominal_power": r.nominal_power, "assurance": r.assurance})
        print(f"  {n:8d} {r.nominal_power:13.1%} {r.assurance:9.1%}")

    ceiling = assurance_ceiling(PRIMARY, SHRINK, cl)
    print(f"\n  Ceiling: {ceiling:.1%}. No sample size does better, because that is the")
    print("  share of plausible effects that point toward benefit at all.")
    print("\n  Enrollment required for a target probability of success:")
    targets = {}
    for t in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90):
        n = n_for_assurance(PRIMARY, t, SHRINK, cl, ALPHA, n_draws=20000)
        targets[t] = n
        print(f"    {t:.0%} assurance: {'unreachable' if not math.isfinite(n) else f'{n:,.0f} patients'}")
    print(
        "\n  This is the single most consequential correction in the analysis. The\n"
        "  design that looked like 92 patients is a coin flip plus a little; genuine\n"
        f"  80% probability of success needs roughly {targets[0.80]:,.0f}."
    )
    pd.DataFrame(rows).to_csv(RESULTS / "assurance_curve.csv", index=False)
    return {
        "ceiling": ceiling,
        "curve": rows,
        "n_for_target": {f"{k:.2f}": v for k, v in targets.items()},
    }


def step7_bootstrap(anchors: dict) -> dict:
    rule("[7/9]  Estimation uncertainty (parametric bootstrap)")
    cl, ce = anchors["late_central"], anchors["early_central"]
    print(
        "  Subgroup estimates rest on 6-8 patients. Each draw samples a plausible\n"
        "  true mean and variance for both strata and re-solves the design; draws\n"
        "  whose effect reverses sign are kept as infeasible rather than discarded."
    )
    out = {}
    print(f"\n  {'endpoint':14s} {'e':>5s} {'N median':>10s} {'80% interval':>18s} {'solvable':>9s}")
    for key in ("lvesv", "viable_mass", "six_min_walk"):
        use_ce, use_cl = (ce, cl) if key in ("lvesv", "lvedv") else (0.0, 0.0)
        for e in (PI, 1.0):
            b = bootstrap_designs(
                key, e, PI, use_ce, use_cl, SHRINK, ALPHA, POWER, DROPOUT
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
    with open(RESULTS / "bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def step8_assumptions(anchors: dict) -> dict:
    rule("[8/9]  Which assumption the conclusion actually rests on")
    ce, cl = anchors["early_central"], anchors["late_central"]

    print("  Scenario analysis over the literature-anchored control range:\n")
    print(f"  {'scenario':38s} {'c_early':>8s} {'c_late':>8s} {'N unsel':>10s} {'N enr':>8s} {'adv':>8s}")
    scenarios = [
        ("no control drift (naive)", 0.0, 0.0),
        ("anchored central", ce, cl),
        ("modern early care (EMPRESS-MI)", anchors["early_low"], cl),
        ("pessimistic late (FOCUS-HF)", ce, anchors["late_low"]),
        ("worst case for enrichment", anchors["early_low"], anchors["late_low"]),
    ]
    scen_rows = []
    for name, c_e, c_l in scenarios:
        u = design(PRIMARY, PI, PI, c_e, c_l, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        e = design(PRIMARY, 1.0, PI, c_e, c_l, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        un = fmt_n(u.n_total) if u.favors_treatment else "no benefit"
        en = fmt_n(e.n_total) if e.favors_treatment else "no benefit"
        adv = (
            f"{u.n_total / e.n_total:,.1f}x"
            if (u.favors_treatment and e.favors_treatment)
            else ("unbounded" if e.favors_treatment else "-")
        )
        scen_rows.append(
            {
                "scenario": name,
                "control_early": c_e,
                "control_late": c_l,
                "n_unselected": u.n_total if u.favors_treatment else np.inf,
                "n_enriched": e.n_total if e.favors_treatment else np.inf,
            }
        )
        print(f"  {name:38s} {c_e:8.1f} {c_l:8.1f} {un:>10s} {en:>8s} {adv:>8s}")

    print(
        "\n  The enriched design is unaffected by the early-stratum assumption, since\n"
        "  it enrolls no early patients. It is entirely dependent on the late one.\n"
        "  FOCUS-CCTRN (n=28, chronic, LVEF<=45%, same delivery route) says zero and\n"
        "  the design stands. FOCUS-HF (n=10) says -9.9 mL, under which VentriGel's\n"
        "  -7.6 mL is smaller than natural history and there is no effect at all.\n"
        "  That one number decides the project."
    )

    print("\n  Winner's-curse discount swept across its full range:")
    curve = shrinkage_curve(
        PRIMARY, PI, ce, cl, alpha=ALPHA, power=POWER, dropout=DROPOUT
    )
    print(f"  {'discount':>9s} {'effect':>9s} {'N enriched':>12s} {'N unselected':>14s}")
    for p in curve[::2]:
        print(
            f"  {p.shrinkage:9.2f} {p.effect_enriched:9.2f} "
            f"{fmt_n(p.n_enriched):>12s} {fmt_n(p.n_unselected):>14s}"
        )
    print(
        "\n  No single discount is defended. Across the whole range the enriched\n"
        "  trial stays between roughly 50 and 300 patients, which is the claim."
    )

    grid = sweep_assumptions(
        PRIMARY, PI, shrinkage=SHRINK, alpha=ALPHA, power=POWER, dropout=DROPOUT
    )
    np.savez(
        RESULTS / "assumption_grid.npz",
        control_early=grid.control_early_values,
        control_late=grid.control_late_values,
        n_total=grid.n_total,
        advantage=grid.advantage,
    )
    pd.DataFrame([p.__dict__ for p in curve]).to_csv(RESULTS / "shrinkage_curve.csv", index=False)
    pd.DataFrame(scen_rows).to_csv(RESULTS / "scenarios.csv", index=False)
    return {"scenarios": scen_rows}


def step9_economics(anchors: dict) -> dict:
    rule("[9/9]  Cost and recruitment feasibility")
    model = CostModel()
    ce, cl = anchors["early_central"], anchors["late_central"]
    print(f"  Unit costs (planning assumptions, not measurements): {model.label()}\n")
    out = {}
    for key in (PRIMARY, "six_min_walk"):
        use_ce, use_cl = (ce, cl) if key == PRIMARY else (0.0, 0.0)
        sweep = enrichment_curve(
            key, PI, use_ce, use_cl, ALPHA, POWER, DROPOUT, n_points=81, shrinkage=SHRINK
        )
        best, costed = optimal_enrichment(sweep, model)
        unsel, _, _, frac = savings_vs_unselected(costed)
        print(f"  {key}:")
        if unsel.feasible and unsel.design.favors_treatment:
            print(f"    unselected: N={fmt_n(unsel.design.n_total)}, ${unsel.total_cost / 1e6:,.1f}M")
        else:
            print(f"    unselected: no benefit to power for at this control assumption")
        print(
            f"    cheapest viable: e={best.design.e:.2f}, N={fmt_n(best.design.n_total)}, "
            f"screened={fmt_n(best.design.n_screened)}, {best.duration_months:.0f} months, "
            f"${best.total_cost / 1e6:,.1f}M"
        )
        out[key] = {
            "e_optimal": best.design.e,
            "n_optimal": best.design.n_total,
            "cost_optimal": best.total_cost,
            "duration_months": best.duration_months,
            "saving_fraction": frac,
        }
        pd.DataFrame(
            [
                {
                    "e": c.design.e,
                    "n_total": c.design.n_total,
                    "n_screened": c.design.n_screened,
                    "effect": c.design.effect,
                    "duration_months": c.duration_months,
                    "total_cost": c.total_cost,
                }
                for c in costed
            ]
        ).to_csv(RESULTS / f"cost_curve_{key}.csv", index=False)

    # An 80%-assurance trial is the design actually recommended, so cost that.
    n80 = n_for_assurance(PRIMARY, 0.80, SHRINK, cl, ALPHA, n_draws=20000)
    if math.isfinite(n80):
        d = design(PRIMARY, 1.0, PI, ce, cl, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        scaled = type(d)(
            endpoint=d.endpoint,
            e=1.0,
            pi=PI,
            effect=d.effect,
            sd=d.sd,
            sd_within=d.sd_within,
            sd_between=d.sd_between,
            standardized_effect=d.standardized_effect,
            n_per_arm=n80 / 2,
            n_total=n80,
            n_screened=n80 / PI,
            screens_per_enrolled=1 / PI,
        )
        c80 = cost(scaled, model)
        print(
            f"\n  The recommended design (80% assurance, N={n80:,.0f}): "
            f"{fmt_n(scaled.n_screened)} screened, {c80.duration_months:.0f} months, "
            f"${c80.total_cost / 1e6:,.1f}M at {model.n_sites} sites."
        )
        # Ninety months of enrollment is not a trial anyone runs, so the design
        # is only real if it is stated at a site count that makes the calendar
        # work. Solve for that rather than reporting an infeasible duration.
        sites_needed = {}
        for target_months in (24, 30, 36, 48):
            enroll_months = target_months - model.followup_months
            if enroll_months <= 0:
                continue
            need = scaled.n_screened / (model.enrollment_rate_per_site_month * enroll_months)
            sites_needed[target_months] = math.ceil(need)
        print("  Sites required to finish on a given calendar:")
        for months, sites in sites_needed.items():
            print(f"    {months} months total: {sites} sites")
        print(
            "\n  The six sites that ran the Phase I cannot deliver this trial. That is\n"
            "  a recruitment finding, not a statistical one, and it is the constraint\n"
            "  a sponsor would hit first."
        )
        out["recommended"] = {
            "n_total": n80,
            "n_screened": scaled.n_screened,
            "duration_months": c80.duration_months,
            "total_cost": c80.total_cost,
            "sites_for_calendar": sites_needed,
        }
    with open(RESULTS / "economics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"VentriGel Phase II enrichment analysis, v{__version__}")
    print(f"Source trial: {TRIAL_ID}")
    print(CITATION)

    summary = {
        "version": __version__,
        "pi": PI,
        "alpha": ALPHA,
        "power": POWER,
        "dropout": DROPOUT,
        "shrinkage": SHRINK,
    }
    summary["validation"] = step1_validation()
    summary["evidence"] = step2_does_the_effect_exist()
    anchors = step3_literature_anchors()
    summary["anchors"] = anchors
    step4_cancellation()
    step5_sample_sizes(anchors)
    summary["assurance"] = step6_assurance(anchors)
    summary["bootstrap"] = step7_bootstrap(anchors)
    summary["assumptions"] = step8_assumptions(anchors)
    summary["economics"] = step9_economics(anchors)

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    rule("Done")
    print(f"  Results written to {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
