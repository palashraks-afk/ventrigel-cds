"""
Reproduce every number in the manuscript.

    python run_analysis.py

Writes machine-readable results to ``results/``. Takes no arguments, because a
reviewer should be able to check the whole analysis with one command.

The order of the sections is the order of the argument. Section 2 asks whether
the subgroup effect exists at all, and section 10 asks what the probability of
success is once that question is left open. Everything between is conditional.
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
    programme_success,
)
from ventrigel.economics import CostModel, cost, optimal_enrichment
from ventrigel.inference import (
    all_interaction_tests,
    assess_evidence,
    baseline_balance,
    effective_n_tests,
    multiplicity,
    regression_to_mean_check,
)
from ventrigel.literature import (
    ANCHOR_CHOICE_RATIONALE,
    ANCHORS,
    BSA_CENTRAL,
    DEFAULT_RETEST_R,
    anchor_coverage,
    anchored_control,
    bsa_sensitivity,
    implied_ventrigel_lvesvi,
    retest_sensitivity,
)
from ventrigel.power import design, enrichment_curve, interaction_design
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
SHRINK = 0.75
DRAWS = 20000


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fmt_n(x: float) -> str:
    if not math.isfinite(x):
        return "infeasible"
    return ">1,000,000" if x >= 1e6 else f"{x:,.0f}"


def controls(endpoint: str) -> tuple[float, float, float, float]:
    """``(c_early, se_early, c_late, se_late)`` from the anchor table.

    Cells with no anchor fall back to zero drift with zero uncertainty, which
    is an assumption and is reported as one in section 3.
    """
    e = anchored_control(endpoint, "early") or (0.0, 0.0)
    l = anchored_control(endpoint, "late") or (0.0, 0.0)
    return e[0], e[1], l[0], l[1]


# --------------------------------------------------------------------------


def step1_validation() -> dict:
    rule("[1/10]  Is the transcription faithful?")
    checks = check_all_p_values()
    real = [c for c in checks if c.excluded_reason is None]
    excluded = [c for c in checks if c.excluded_reason is not None]
    agree = sum(c.agrees for c in real)
    print(f"  Recomputed {len(checks)} published paired t-test p-values from mean, SEM and n.")
    print(f"  {agree}/{len(real)} agree at the precision the source prints.")
    if excluded:
        seen = set()
        print(f"  {len(excluded)} cells excluded for documented reasons:")
        for c in excluded:
            head = c.excluded_reason.split(".")[0]
            if head not in seen:
                seen.add(head)
                print(f"    {c.endpoint}: {head}.")

    mixtures = check_all_mixtures()
    med = float(np.median([m.sd_rel_error for m in mixtures]))
    lv = next(m for m in mixtures if m.endpoint == PRIMARY)
    print(f"\n  Pooled cohorts reconstructed from their strata; median SD error {med * 100:.1f}%.")
    print(
        f"    {PRIMARY}: mean {lv.mean_reconstructed:.2f} vs published {lv.mean_published:.2f}, "
        f"SD {lv.sd_reconstructed:.2f} vs {lv.sd_published:.2f} ({lv.sd_rel_error * 100:.1f}%)"
    )
    pd.DataFrame([c.__dict__ for c in checks]).to_csv(RESULTS / "validation_pvalues.csv", index=False)
    pd.DataFrame([m.__dict__ for m in mixtures]).to_csv(RESULTS / "validation_mixture.csv", index=False)
    return {
        "p_checks_tested": len(real),
        "p_checks_agree": agree,
        "sd_reconstruction_median_rel_error": med,
    }


def step2_does_the_effect_exist() -> dict:
    rule("[2/10]  Does the subgroup effect exist? (the test nobody ran)")
    print(
        "  The trial compared each stratum against its own baseline and observed\n"
        "  that one reached significance while the other did not. That is not a\n"
        "  test of effect modification.\n"
    )
    tests = all_interaction_tests()
    print(f"  {'endpoint':14s} {'early':>8s} {'late':>8s} {'diff':>9s} {'95% CI':>19s} {'p':>8s}")
    for t in tests:
        ci = f"[{t.ci_low:.1f}, {t.ci_high:.1f}]"
        print(
            f"  {t.endpoint:14s} {t.early_mean:8.2f} {t.late_mean:8.2f} "
            f"{t.difference:9.2f} {ci:>19s} {t.p_value:8.4f}"
            + (" *" if t.nominally_significant else "")
        )

    mult = multiplicity(tests)
    nominal, effective, why = effective_n_tests()
    print(f"\n  Multiplicity across {nominal} interaction tests:")
    for m in mult[:3]:
        print(
            f"    {m.endpoint:14s} p={m.p_value:.4f}  Bonferroni p={m.bonferroni_p:.3f}  "
            f"BH threshold={m.bh_threshold:.4f}  passes={'yes' if m.bh_pass else 'no'}"
        )
    print(f"    ... {len(mult) - 3} further endpoints, all p > 0.16")
    print(f"\n  {why}")
    print(
        "\n  Two further caveats on the test itself, neither of which can be resolved\n"
        "  without patient-level data:\n"
        "    - It is a comparison of change scores, not an ANCOVA adjusting for\n"
        "      baseline. With balanced baselines ANCOVA is the standard and more\n"
        "      powerful choice, so p = 0.034 is probably conservative.\n"
        "    - The family is nine endpoints at the 6-month visit. The trial also\n"
        "      reported 1- and 3-month visits; counting those would enlarge the\n"
        "      family and weaken the result further. Six months is used because it\n"
        "      is the prespecified secondary-endpoint timepoint."
    )

    balance = baseline_balance()
    print(
        f"\n  Baseline balance: {sum(not b.imbalanced for b in balance)}/{len(balance)} "
        f"measures balanced (minimum p = {min(b.p_value for b in balance):.2f})."
    )
    rtm = regression_to_mean_check(PRIMARY)
    print(f"  Regression to the mean: {rtm.explanation}")

    ev = assess_evidence()
    print(f"\n  VERDICT: {ev.verdict}")

    pd.DataFrame([t.__dict__ for t in tests]).to_csv(RESULTS / "interaction_tests.csv", index=False)
    pd.DataFrame([m.__dict__ for m in mult]).to_csv(RESULTS / "multiplicity.csv", index=False)
    pd.DataFrame([b.__dict__ for b in balance]).to_csv(RESULTS / "baseline_balance.csv", index=False)
    return {
        "strongest_endpoint": ev.strongest_endpoint,
        "strongest_p": ev.strongest_p,
        "n_tests": ev.n_tests,
        "effective_n_tests": effective,
        "survives_bonferroni": ev.survives_bonferroni,
        "survives_bh": ev.survives_bh,
        "baseline_balanced": ev.baseline_balanced,
        "rtm_ruled_out": ev.rtm_ruled_out,
        "verdict": ev.verdict,
    }


def step3_literature_anchors() -> dict:
    rule("[3/10]  What does an untreated patient do? (external control anchors)")
    print(
        "  The Phase I was single-arm, so its comparator is missing. Control-arm\n"
        f"  change is anchored to published control and placebo arms. Indexed\n"
        f"  volumes converted at BSA = {BSA_CENTRAL} m2; change SDs recovered from\n"
        f"  level SDs assume a test-retest correlation of {DEFAULT_RETEST_R}.\n"
    )
    print(
        f"  {'trial':22s} {'endpoint':13s} {'phase':>8s} {'change':>9s} {'SE':>6s} "
        f"{'n':>6s}  evidence"
    )
    rows = []
    for a in ANCHORS.values():
        se = a.standard_error()
        print(
            f"  {a.trial[:22]:22s} {a.endpoint:13s} {a.phase:>8s} "
            f"{a.absolute_change():+9.1f} {('n/a' if se is None else f'{se:.2f}'):>6s} "
            f"{a.n:6d}  {a.evidence}"
        )
        rows.append(
            {
                "trial": a.trial, "year": a.year, "endpoint": a.endpoint,
                "phase": a.phase, "measure": a.measure,
                "change_published": a.change, "change_absolute": a.absolute_change(),
                "standard_error": se, "n": a.n, "evidence": a.evidence,
                "citation": a.citation,
            }
        )

    print("\n  Anchor coverage (which endpoint-stratum cells have one):")
    cov = anchor_coverage()
    for ep in CANDIDATE_PRIMARY_ENDPOINTS:
        d = cov.get(ep, {})
        e_, l_ = d.get("early"), d.get("late")
        print(f"    {ep:14s} early={str(e_ or 'NONE'):24s} late={l_ or 'NONE'}")
    print(
        "    Cells marked NONE fall back to zero drift, which is an assumption that\n"
        "    flatters the treatment whenever the untreated course is favourable."
    )

    print("\n  Where two anchors compete, the choice is explicit:")
    for (ep, st), why in ANCHOR_CHOICE_RATIONALE.items():
        print(f"    {ep}/{st}: {why.split('.')[0]}.")

    print(
        f"\n  BSA check: VentriGel's 148.5 mL baseline implies "
        f"{implied_ventrigel_lvesvi():.1f} mL/m2, against 65.0 mL/m2 in chronic\n"
        "  FOCUS-CCTRN. Higher, as expected for a dilated post-MI cohort."
    )
    print("\n  BSA sensitivity for the late LVESV anchor (the one that matters):")
    for b, ch, se in bsa_sensitivity():
        print(f"    BSA {b:.2f}: change {ch:+.2f} mL, SE {se:.2f}")
    print("    The point estimate is exactly zero at every BSA; only its SE moves.")

    print("\n  Test-retest correlation sensitivity for that anchor's SE:")
    for r, se in retest_sensitivity():
        print(f"    r = {r:.2f}: SE {se:.2f} mL")

    ce, se_e, cl, se_l = controls(PRIMARY)
    print(
        f"\n  LVESV comparators used downstream: early {ce:+.2f} +/- {se_e:.2f}, "
        f"late {cl:+.2f} +/- {se_l:.2f} mL."
    )
    print(
        "  That late standard error is the single most consequential number in this\n"
        "  analysis after the interaction p-value. It is the same order as the 7.6 mL\n"
        "  effect being measured against it, and section 6 shows what happens when it\n"
        "  is propagated rather than ignored."
    )
    pd.DataFrame(rows).to_csv(RESULTS / "literature_anchors.csv", index=False)
    return {
        "control_early": ce, "control_early_se": se_e,
        "control_late": cl, "control_late_se": se_l,
        "bsa": BSA_CENTRAL, "retest_r": DEFAULT_RETEST_R,
        "coverage": cov,
    }


def step4_cancellation() -> None:
    rule("[4/10]  Why the pooled effect is a cancellation, not a weak effect")
    rows = []
    for key, ep in ENDPOINTS.items():
        t = ep.change_6mo
        if not {"early", "late"} <= t.keys():
            continue
        a, b = t["early"].mean, t["late"].mean
        rows.append(
            {
                "endpoint": key, "early": a, "late": b,
                "pooled": t["total"].mean if "total" in t else np.nan,
                "separation_benefit": ep.benefit("late") - ep.benefit("early"),
                "strata_oppose": bool(a * b < 0 and min(abs(a), abs(b)) > 0.10 * max(abs(a), abs(b))),
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
        "  interaction. Viable mass looks more dramatic but its interaction p is 0.17."
    )
    df.to_csv(RESULTS / "subgroup_effects.csv", index=False)


def step5_sample_sizes() -> pd.DataFrame:
    rule("[5/10]  Sample size, unselected versus enriched")
    print(
        f"  Two-arm design, alpha={ALPHA}, power={POWER:.0%}, {DROPOUT:.0%} dropout,\n"
        f"  noncentral t, pi={PI:.3f}, effect discount {SHRINK:.2f}, anchored controls.\n"
    )
    rows = []
    for key in CANDIDATE_PRIMARY_ENDPOINTS:
        ce, _, cl, _ = controls(key)
        u = design(key, PI, PI, ce, cl, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        e = design(key, 1.0, PI, ce, cl, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        rows.append(
            {
                "endpoint": key, "control_early": ce, "control_late": cl,
                "effect_unselected": u.effect,
                "n_unselected": u.n_total if u.favors_treatment else np.inf,
                "effect_enriched": e.effect, "sd_enriched": e.sd,
                "cohens_d_enriched": e.standardized_effect,
                "n_enriched": e.n_total if e.favors_treatment else np.inf,
                "advantage": (u.n_total / e.n_total)
                if (u.favors_treatment and e.favors_treatment) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    print(
        f"  {'endpoint':14s} {'c_early':>8s} {'c_late':>7s} {'eff unsel':>10s} "
        f"{'N unsel':>11s} {'eff enr':>8s} {'N enrich':>9s} {'ratio':>8s}"
    )
    for _, r in df.iterrows():
        rat = r["advantage"]
        rs = "-" if not np.isfinite(rat) else ("worse" if rat < 1 else f"{rat:,.1f}x")
        print(
            f"  {r['endpoint']:14s} {r['control_early']:8.1f} {r['control_late']:7.1f} "
            f"{r['effect_unselected']:10.2f} {fmt_n(r['n_unselected']):>11s} "
            f"{r['effect_enriched']:8.2f} {fmt_n(r['n_enriched']):>9s} {rs:>8s}"
        )
    print(
        "\n  Anchoring changes two endpoints qualitatively. Ejection fraction, which\n"
        "  looked like a pure harm signal against no comparator, has a late stratum\n"
        "  falling 0.6 points against a comparator falling 1.3 -- a small benefit.\n"
        "  The 6-min walk loses a third of its effect to the placebo response and\n"
        "  is no longer the cheap option it appeared to be."
    )
    df.to_csv(RESULTS / "sample_sizes.csv", index=False)
    return df


def step6_anchor_uncertainty(anchors: dict) -> dict:
    rule("[6/10]  What the anchor's own uncertainty costs")
    cl, se_l = anchors["control_late"], anchors["control_late_se"]
    print(
        f"  The late comparator is {cl:+.2f} +/- {se_l:.2f} mL from 28 patients. Treating\n"
        "  it as exact is the difference between a plausible trial and an optimistic one.\n"
    )
    print(f"  {'':26s} {'assurance @174':>15s} {'ceiling':>9s} {'N for 80%':>11s}")
    out = {}
    for label, se in (("anchor treated as exact", 0.0), ("anchor uncertainty propagated", se_l)):
        a = assurance(PRIMARY, 174, SHRINK, cl, ALPHA, n_draws=DRAWS, control_se=se)
        ceil = assurance_ceiling(PRIMARY, SHRINK, cl, n_draws=DRAWS, control_se=se)
        n80 = n_for_assurance(PRIMARY, 0.80, SHRINK, cl, ALPHA, n_draws=DRAWS, control_se=se)
        print(f"  {label:26s} {a.assurance:14.1%} {ceil:8.1%} {fmt_n(n80):>11s}")
        out[label] = {"assurance_174": a.assurance, "ceiling": ceil, "n_for_80": n80}
    print(
        "\n  Propagating it roughly doubles the trial. Every headline below uses the\n"
        "  propagated version; the exact-anchor row is shown only to size the error\n"
        "  that omitting it would introduce."
    )
    return out


def step7_assurance(anchors: dict) -> dict:
    rule("[7/10]  Power is not probability of success")
    cl, se_l = anchors["control_late"], anchors["control_late_se"]
    print(
        "  Assurance integrates power over the uncertainty in the effect and in the\n"
        "  comparator. It is what a sponsor deciding whether to fund actually needs.\n"
    )
    print(f"  {'N total':>8s} {'nominal power':>14s} {'assurance':>10s}")
    rows = []
    for n in (52, 92, 174, 300, 406, 600, 1000):
        r = assurance(PRIMARY, n, SHRINK, cl, ALPHA, n_draws=DRAWS, control_se=se_l)
        rows.append({"n_total": n, "nominal_power": r.nominal_power, "assurance": r.assurance})
        print(f"  {n:8d} {r.nominal_power:13.1%} {r.assurance:9.1%}")

    ceiling = assurance_ceiling(PRIMARY, SHRINK, cl, n_draws=DRAWS, control_se=se_l)
    print(f"\n  Ceiling: {ceiling:.1%} -- the share of plausible effects favouring benefit.")
    targets = {}
    print("\n  Enrollment for a target probability of success:")
    for t in (0.50, 0.60, 0.70, 0.80, 0.85):
        n = n_for_assurance(PRIMARY, t, SHRINK, cl, ALPHA, n_draws=DRAWS, control_se=se_l)
        targets[t] = n
        print(f"    {t:.0%}: {'unreachable' if not math.isfinite(n) else f'{n:,.0f} patients'}")
    pd.DataFrame(rows).to_csv(RESULTS / "assurance_curve.csv", index=False)
    return {"ceiling": ceiling, "curve": rows, "n_for_target": {f"{k:.2f}": v for k, v in targets.items()}}


def step8_confirming_the_claim(anchors: dict) -> dict:
    rule("[8/10]  Confirming the claim itself, not just the effect")
    print(
        "  The enriched design enrolls only late patients. It can show the therapy\n"
        "  works in that stratum; it can never show that timing matters, because it\n"
        "  contains no early patients. But 'treat late, not early' is the actual\n"
        "  claim. Confirming it needs a 2x2 trial powered on the interaction.\n"
    )
    ce, _, cl, _ = controls(PRIMARY)
    print(
        f"  {'scenario':22s} {'contrast':>9s} {'N/cell':>7s} {'N total':>9s} "
        f"{'enriched ref':>13s} {'ratio':>7s}"
    )
    rows = []
    for label, (a, b) in (("no control drift", (0.0, 0.0)), ("anchored controls", (ce, cl))):
        d = interaction_design(PRIMARY, a, b, ALPHA, POWER, DROPOUT, SHRINK)
        rows.append(
            {
                "scenario": label, "control_early": a, "control_late": b,
                "contrast": d.contrast, "n_per_cell": d.n_per_cell,
                "n_total": d.n_total, "n_enriched": d.n_enriched_reference,
                "ratio": d.ratio_to_enriched,
            }
        )
        print(
            f"  {label:22s} {d.contrast:9.2f} {d.n_per_cell:7.0f} {d.n_total:9.0f} "
            f"{d.n_enriched_reference:13.0f} {d.ratio_to_enriched:6.1f}x"
        )
    print(
        "\n  Anchoring halves the contrast, because most of the early stratum's apparent\n"
        "  harm turns out to be natural history rather than a failure of treatment.\n"
        "  The interaction design then costs about 440 patients -- close to the 406 an\n"
        "  80%-assurance enriched trial needs. For roughly the same money a sponsor can\n"
        "  answer the question they actually have instead of half of it."
    )
    pd.DataFrame(rows).to_csv(RESULTS / "interaction_design.csv", index=False)
    return {"designs": rows}


def step9_uncertainty_and_assumptions(anchors: dict) -> dict:
    rule("[9/10]  Estimation uncertainty and the assumptions it rests on")
    ce, cl = anchors["control_early"], anchors["control_late"]
    print(f"  {'endpoint':14s} {'e':>5s} {'N median':>10s} {'80% interval':>18s} {'solvable':>9s}")
    for key in ("lvesv", "six_min_walk", "ef"):
        k_ce, _, k_cl, _ = controls(key)
        for e in (PI, 1.0):
            b = bootstrap_designs(key, e, PI, k_ce, k_cl, SHRINK, ALPHA, POWER, DROPOUT)
            # A median over a handful of surviving draws is noise dressed as a
            # number. Below 5% solvable the only honest summary is that the
            # design almost never favours treatment.
            if b.feasible_fraction < 0.05:
                med = interval = "--"
            else:
                med = fmt_n(b.n_total_median)
                interval = fmt_n(b.n_total_q10) + "-" + fmt_n(b.n_total_q90)
            print(
                f"  {key:14s} {e:5.2f} {med:>10s} {interval:>18s} "
                f"{b.feasible_fraction:8.1%}"
            )

    print("\n  Scenario analysis over the anchored control range:")
    print(f"  {'scenario':34s} {'c_early':>8s} {'c_late':>8s} {'N unsel':>10s} {'N enr':>9s}")
    scen = [
        ("no control drift (naive)", 0.0, 0.0),
        ("anchored central", ce, cl),
        ("EMPRESS-MI early comparator", -14.82, cl),
        ("FOCUS-HF late comparator", ce, -9.9),
    ]
    rows = []
    for name, c_e, c_l in scen:
        u = design(PRIMARY, PI, PI, c_e, c_l, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        e = design(PRIMARY, 1.0, PI, c_e, c_l, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        un = fmt_n(u.n_total) if u.favors_treatment else "no benefit"
        en = fmt_n(e.n_total) if e.favors_treatment else "no benefit"
        rows.append({"scenario": name, "control_early": c_e, "control_late": c_l,
                     "n_unselected": u.n_total if u.favors_treatment else np.inf,
                     "n_enriched": e.n_total if e.favors_treatment else np.inf})
        print(f"  {name:34s} {c_e:8.1f} {c_l:8.1f} {un:>10s} {en:>9s}")
    print(
        "\n  The enriched design is unaffected by the early comparator, since it enrolls\n"
        "  no early patients. It depends entirely on the late one, and FOCUS-HF's -9.9 mL\n"
        "  would nullify it outright."
    )

    curve = shrinkage_curve(PRIMARY, PI, ce, cl, alpha=ALPHA, power=POWER, dropout=DROPOUT)
    print("\n  Winner's-curse discount across its full range:")
    for p in curve[::3]:
        print(f"    discount {p.shrinkage:.2f}: enriched N {fmt_n(p.n_enriched)}")

    grid = sweep_assumptions(PRIMARY, PI, shrinkage=SHRINK, alpha=ALPHA, power=POWER, dropout=DROPOUT)
    np.savez(
        RESULTS / "assumption_grid.npz",
        control_early=grid.control_early_values, control_late=grid.control_late_values,
        n_total=grid.n_total, advantage=grid.advantage,
    )
    pd.DataFrame([p.__dict__ for p in curve]).to_csv(RESULTS / "shrinkage_curve.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "scenarios.csv", index=False)
    return {"scenarios": rows}


def step10_programme(anchors: dict) -> dict:
    rule("[10/10]  The number a sponsor actually needs")
    cl, se_l = anchors["control_late"], anchors["control_late_se"]
    print(
        "  Everything above is conditional on the interaction being real. It rests on\n"
        "  one nominally significant result out of nine that survives no correction.\n"
        "  Unconditional probability of success is that prior times the assurance.\n"
    )
    priors = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
    print(f"  {'N':>7s} " + " ".join(f"{p:>7.0%}" for p in priors))
    rows = []
    for n in (92, 174, 300, 406, 800):
        vals = [
            programme_success(PRIMARY, n, p, SHRINK, cl, se_l, ALPHA, n_draws=DRAWS).unconditional
            for p in priors
        ]
        rows.append({"n_total": n, **{f"prior_{p:.1f}": v for p, v in zip(priors, vals)}})
        print(f"  {n:7d} " + " ".join(f"{v:7.1%}" for v in vals))
    print(
        "\n  The prior enters multiplicatively, so no sample size lifts the programme\n"
        "  above it. At an even-odds prior the best achievable is about 42%, and the\n"
        "  marginal return on patients beyond roughly 400 is close to nothing. That is\n"
        "  the argument for spending the next increment of money on measuring the\n"
        "  control-arm comparator rather than on more patients."
    )
    pd.DataFrame(rows).to_csv(RESULTS / "programme_success.csv", index=False)

    # Cost the recommended design.
    model = CostModel()
    n80 = n_for_assurance(PRIMARY, 0.80, SHRINK, cl, ALPHA, n_draws=DRAWS, control_se=se_l)
    econ = {}
    if math.isfinite(n80):
        d = design(PRIMARY, 1.0, PI, anchors["control_early"], cl, ALPHA, POWER, DROPOUT, shrinkage=SHRINK)
        scaled = type(d)(
            endpoint=d.endpoint, e=1.0, pi=PI, effect=d.effect, sd=d.sd,
            sd_within=d.sd_within, sd_between=d.sd_between,
            standardized_effect=d.standardized_effect, n_per_arm=n80 / 2,
            n_total=n80, n_screened=n80 / PI, screens_per_enrolled=1 / PI,
        )
        c = cost(scaled, model)
        print(
            f"\n  Recommended design: N={n80:,.0f}, {scaled.n_screened:,.0f} screened, "
            f"${c.total_cost / 1e6:,.1f}M."
        )
        sites = {}
        for months in (24, 30, 36, 48):
            em = months - model.followup_months
            if em > 0:
                sites[months] = math.ceil(
                    scaled.n_screened / (model.enrollment_rate_per_site_month * em)
                )
        print("  Sites required: " + ", ".join(f"{v} for {k} months" for k, v in sites.items()))
        econ = {
            "n_total": n80, "n_screened": scaled.n_screened,
            "total_cost": c.total_cost, "sites_for_calendar": sites,
        }
    with open(RESULTS / "economics.json", "w") as f:
        json.dump(econ, f, indent=2, default=float)
    return {"programme": rows, "recommended": econ}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"VentriGel Phase II enrichment analysis, v{__version__}")
    print(f"Source trial: {TRIAL_ID}")
    print(CITATION)

    summary = {
        "version": __version__, "pi": PI, "alpha": ALPHA, "power": POWER,
        "dropout": DROPOUT, "shrinkage": SHRINK,
    }
    summary["validation"] = step1_validation()
    summary["evidence"] = step2_does_the_effect_exist()
    anchors = step3_literature_anchors()
    summary["anchors"] = anchors
    step4_cancellation()
    step5_sample_sizes()
    summary["anchor_uncertainty"] = step6_anchor_uncertainty(anchors)
    summary["assurance"] = step7_assurance(anchors)
    summary["confirmatory"] = step8_confirming_the_claim(anchors)
    summary["sensitivity"] = step9_uncertainty_and_assumptions(anchors)
    summary["programme"] = step10_programme(anchors)

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    rule("Done")
    print(f"  Results written to {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
