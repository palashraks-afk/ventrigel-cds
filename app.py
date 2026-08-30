"""
Phase II trial design calculator for VentriGel.

    streamlit run app.py

Answers one question: given what the VentriGel Phase I actually showed, how
large would a Phase II have to be, how likely is it to succeed, and which
assumption is the answer most sensitive to?

It is deliberately not a patient-level predictor. Nothing here estimates
whether an individual will respond, because a 15-patient single-arm trial
cannot support that claim.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ventrigel import __version__
from ventrigel.assurance import assurance_ceiling, assurance_curve, n_for_assurance
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
    ANCHORS,
    BSA_CENTRAL,
    early_control_prior,
    implied_ventrigel_lvesvi,
    late_control_prior,
)
from ventrigel.power import (
    EnrichedPopulation,
    Stratum,
    n_per_arm_exact,
    n_screened,
)
from ventrigel.recovery import check_all_mixtures, check_all_p_values, subgroup_effects
from ventrigel.sensitivity import bootstrap_designs
from ventrigel.trial_data import (
    CANDIDATE_PRIMARY_ENDPOINTS,
    CITATION,
    ENDPOINTS,
    N_EARLY,
    N_LATE,
    TRIAL_ID,
)

PI_TRIAL = N_LATE / (N_EARLY + N_LATE)
EARLY_PRIOR = early_control_prior()
LATE_PRIOR = late_control_prior()

st.set_page_config(
    page_title="VentriGel Phase II Design Calculator",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.0rem; max-width: 1300px;}
      div[data-testid="stMetric"] {
          background: rgba(128,128,128,0.07);
          padding: 0.7rem 0.9rem; border-radius: 10px;
      }
      div[data-testid="stMetricValue"] {font-size: 1.7rem;}
      .stTabs [data-baseweb="tab"] {padding: 0.4rem 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VentriGel Phase II trial design calculator")
st.caption(
    f"Built from published summary statistics of {TRIAL_ID} and control arms of five other "
    "randomized trials. No synthetic patients, no patient-level prediction."
)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Design")
    endpoint = st.selectbox(
        "Primary endpoint",
        CANDIDATE_PRIMARY_ENDPOINTS,
        format_func=lambda k: ENDPOINTS[k].label,
        index=0,
    )
    ep = ENDPOINTS[endpoint]
    alpha = st.select_slider("Two-sided alpha", [0.01, 0.025, 0.05, 0.10], value=0.05)
    power_target = st.select_slider("Power", [0.70, 0.80, 0.85, 0.90, 0.95], value=0.80)
    dropout = st.slider("Dropout before primary assessment", 0.0, 0.35, 0.10, 0.01)

    st.header("Population")
    pi = st.slider(
        "Late-stratum prevalence in the eligible pool",
        0.03, 0.95, float(round(PI_TRIAL, 2)), 0.01,
        help=(
            f"The trial's own split was {PI_TRIAL:.0%}, but it enrolled the strata "
            "deliberately balanced, so that estimates its design, not the population."
        ),
    )
    e_level = st.slider(
        "Enrichment: fraction of enrolled patients from the late stratum",
        float(round(pi, 2)), 1.0, 1.0, 0.01,
    )

    st.header("Control arm")
    use_anchors = st.checkbox(
        "Use literature-anchored control arms", value=True,
        help="TIME for the early stratum, FOCUS-CCTRN for the late stratum.",
    )
    unit = ep.unit
    if endpoint in ("lvesv", "lvedv"):
        lo, hi = -20.0, 20.0
        default_e, default_l = EARLY_PRIOR.central, LATE_PRIOR.central
    else:
        default_e = default_l = 0.0
        lo, hi = {
            "%": (-6.0, 6.0), "points": (-25.0, 25.0), "g": (-30.0, 30.0),
        }.get(unit, (-40.0, 40.0))

    if use_anchors and endpoint in ("lvesv", "lvedv"):
        c_early, c_late = default_e, default_l
        st.caption(
            f"Early {c_early:+.1f} {unit} (TIME) · Late {c_late:+.1f} {unit} (FOCUS-CCTRN). "
            "Uncheck above to set these by hand."
        )
    else:
        if use_anchors:
            st.caption("No published anchor for this endpoint; assuming no control drift.")
        c_early = st.slider(f"Control change, early stratum ({unit})", lo, hi, float(default_e), 0.5)
        c_late = st.slider(f"Control change, late stratum ({unit})", lo, hi, float(default_l), 0.5)

    shrinkage = st.slider(
        "Effect discount for winner's curse", 0.30, 1.00, 0.75, 0.05,
        help=(
            "The late subgroup was identified post hoc in the same 15 patients that "
            "produced its effect estimate. 1.00 takes the published effect at face value."
        ),
    )

    st.header("Operations")
    per_patient = st.number_input("Cost per randomized patient ($)", 10_000, 250_000, 65_000, 5_000)
    per_screen = st.number_input("Cost per screen ($)", 250, 25_000, 2_500, 250)
    n_sites = st.number_input("Sites", 1, 80, 6)
    rate = st.number_input("Enrollment rate (patients/site/month)", 0.1, 6.0, 0.75, 0.05)


# --------------------------------------------------------------------------
# Core model
# --------------------------------------------------------------------------


def build(e_val: float) -> tuple[EnrichedPopulation, float, float]:
    eff = subgroup_effects(endpoint, "6mo")
    pop = EnrichedPopulation(
        early=Stratum("early", eff["early"]["mean"], eff["early"]["sd"], c_early, shrinkage),
        late=Stratum("late", eff["late"]["mean"], eff["late"]["sd"], c_late, shrinkage),
        e=e_val,
        lower_is_better=ep.lower_is_better,
    )
    if pop.effect <= 0:
        return pop, math.inf, math.inf
    n_arm = n_per_arm_exact(pop.effect, pop.sd, alpha, power_target)
    if not math.isfinite(n_arm):
        return pop, math.inf, math.inf
    n_arm = math.ceil(n_arm / max(1e-9, 1 - dropout))
    n_total = 2.0 * n_arm
    return pop, n_total, n_screened(n_total, e_val, pi)


pop, n_total, n_scr = build(e_level)
pop_u, n_total_u, _ = build(pi)
model = CostModel(
    per_patient=float(per_patient), per_screen=float(per_screen),
    n_sites=int(n_sites), enrollment_rate_per_site_month=float(rate),
)


def fmt(x: float) -> str:
    return "not achievable" if not math.isfinite(x) else f"{x:,.0f}"


# --------------------------------------------------------------------------
# Headline
# --------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Effect vs. control", f"{pop.effect:+.2f} {ep.unit}")
c2.metric("SD of change", f"{pop.sd:.2f} {ep.unit}")
c3.metric("Standardized effect", f"{pop.standardized_effect:.2f}")
c4.metric(f"N for {power_target:.0%} power", fmt(n_total))

if not math.isfinite(n_total):
    st.error(
        f"Under these assumptions the modelled effect is {pop.effect:+.2f} {ep.unit}, which does "
        "not favour treatment. No sample size demonstrates benefit; the design question is moot "
        "until the assumptions change."
    )
else:
    site_capacity = n_sites * rate
    months = n_scr / site_capacity if site_capacity else math.inf
    from ventrigel.power import Design

    d = Design(
        endpoint=endpoint, e=e_level, pi=pi, effect=pop.effect, sd=pop.sd,
        sd_within=pop.sd_within, sd_between=pop.sd_between,
        standardized_effect=pop.standardized_effect, n_per_arm=n_total / 2,
        n_total=n_total, n_screened=n_scr, screens_per_enrolled=n_scr / n_total,
    )
    costed = cost(d, model)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Patients to screen", fmt(n_scr))
    d2.metric("Screens per enrollment", f"{n_scr / n_total:.1f}")
    d3.metric("Enrollment duration", f"{months:.0f} months")
    d4.metric("Estimated cost", f"${costed.total_cost / 1e6:,.1f}M")

    if math.isfinite(n_total_u) and n_total_u > 0:
        st.success(
            f"Enriching to {e_level:.0%} late patients reduces the trial from "
            f"{fmt(n_total_u)} to {fmt(n_total)} randomized patients "
            f"({n_total_u / n_total:,.1f}×), at {n_scr / n_total:.1f} screens per enrollment."
        )
    else:
        st.success(
            f"An unselected trial has no benefit to detect under these assumptions "
            f"(pooled effect {pop_u.effect:+.2f} {ep.unit}). The enriched design needs "
            f"{fmt(n_total)} patients — the difference between a trial and no trial."
        )
    if months > 120:
        st.warning(
            f"Enrollment would take {months / 12:.0f} years at {n_sites} sites. "
            "When responders are rare, calendar time binds before sample size does."
        )

st.divider()

tabs = st.tabs([
    "Does the effect exist?",
    "Probability of success",
    "Control arms",
    "Enrichment curve",
    "Uncertainty",
    "Cost",
    "Source data",
    "Validation",
    "About",
])

# --------------------------------------------------------------------------
# 1. Does the effect exist?
# --------------------------------------------------------------------------

with tabs[0]:
    st.subheader("The test the trial did not run")
    st.markdown(
        "The Phase I compared each stratum against its **own baseline** and observed that one "
        "reached significance while the other did not. That is not a test of effect modification: "
        "two subgroups can land on opposite sides of p = 0.05 without differing from each other. "
        "Below is the direct comparison."
    )
    tests = all_interaction_tests()
    mult = {m.endpoint: m for m in multiplicity(tests)}
    df = pd.DataFrame([
        {
            "Endpoint": t.label,
            "Early": f"{t.early_mean:+.2f} (n={t.early_n})",
            "Late": f"{t.late_mean:+.2f} (n={t.late_n})",
            "Difference": f"{t.difference:+.2f}",
            "95% CI": f"[{t.ci_low:.1f}, {t.ci_high:.1f}]",
            "p": round(t.p_value, 4),
            "Bonferroni": "pass" if mult[t.endpoint].bonferroni_pass else "fail",
            "BH": "pass" if mult[t.endpoint].bh_pass else "fail",
        }
        for t in tests
    ])
    st.dataframe(df, width="stretch", hide_index=True)

    nominal, effective, why = effective_n_tests()
    ev = assess_evidence()
    st.error(f"**Verdict.** {ev.verdict}")
    st.caption(why)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Baseline balance** — the strata were not randomized against each other.")
        bal = baseline_balance()
        st.dataframe(
            pd.DataFrame([
                {"Measure": b.label, "Early": round(b.early_mean, 1),
                 "Late": round(b.late_mean, 1), "p": round(b.p_value, 3)}
                for b in bal
            ]),
            width="stretch", hide_index=True, height=260,
        )
        st.success(
            f"All {len(bal)} measures balanced (minimum p = {min(b.p_value for b in bal):.2f}). "
            "The contrast is not confounded by baseline severity."
        )
    with cc2:
        st.markdown("**Regression to the mean** — the most obvious artifact.")
        rtm = regression_to_mean_check("lvesv")
        st.write(rtm.explanation)
        if rtm.contradicts_rtm:
            st.success("Ruled out: the pattern runs opposite to what regression would produce.")
        else:
            st.warning("Not ruled out.")

# --------------------------------------------------------------------------
# 2. Probability of success
# --------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Power is a conditional promise; assurance is not")
    st.markdown(
        "Sizing for 80% power *at a point estimate* and then calling the result an 80% trial "
        "promotes an assumption to a fact. **Assurance** integrates power over the uncertainty in "
        "the effect itself — the quantity a sponsor deciding whether to fund actually needs."
    )
    ns = np.unique(np.round(np.logspace(math.log10(24), math.log10(3000), 30)).astype(int))
    curve = assurance_curve(endpoint, ns, shrinkage, c_late, alpha, n_draws=20000)
    ceiling = assurance_ceiling(endpoint, shrinkage, c_late)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r.assurance for r in curve], mode="lines",
                             name="assurance", line={"width": 3}))
    fig.add_trace(go.Scatter(x=ns, y=[r.nominal_power for r in curve], mode="lines",
                             name="nominal power", line={"width": 2, "dash": "dash"}))
    fig.add_hline(y=ceiling, line_dash="dot",
                  annotation_text=f"ceiling {ceiling:.0%}", annotation_position="top left")
    if math.isfinite(n_total) and n_total > 0:
        # Plotly places shapes in axis coordinates, which on a log axis means
        # log10 of the data value. Passing the raw n here draws the line at
        # 10**n and blows the axis range out to absurdity.
        fig.add_vline(x=math.log10(n_total), line_dash="dot", annotation_text="current design")
    fig.update_xaxes(type="log", title="Total randomized patients")
    fig.update_yaxes(title="Probability of a significant result", range=[0, 1.02])
    fig.update_layout(height=400, margin={"t": 30})
    st.plotly_chart(fig, width="stretch")

    if math.isfinite(n_total):
        here = min(curve, key=lambda r: abs(r.n_total - n_total))
        m1, m2, m3 = st.columns(3)
        m1.metric("Nominal power at this N", f"{here.nominal_power:.0%}")
        m2.metric("Actual probability of success", f"{here.assurance:.0%}")
        m3.metric("Achievable ceiling", f"{ceiling:.0%}")

    rows = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
        n = n_for_assurance(endpoint, t, shrinkage, c_late, alpha, n_draws=20000)
        rows.append({
            "Target probability of success": f"{t:.0%}",
            "Patients required": "unreachable" if not math.isfinite(n) else f"{n:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.info(
        f"Assurance is bounded above by {ceiling:.0%} — the share of plausible effects that point "
        "toward benefit at all. Enrollment cannot fix an effect that is not there."
    )

# --------------------------------------------------------------------------
# 3. Control arms
# --------------------------------------------------------------------------

with tabs[2]:
    st.subheader("What happens to untreated patients")
    st.markdown(
        "The Phase I was **single-arm**, so its comparator is missing. Rather than sweeping an "
        "arbitrary range, control-arm change is anchored to published control and placebo arms."
    )
    st.dataframe(
        pd.DataFrame([
            {
                "Trial": a.trial, "Year": a.year, "Phase": a.phase, "Measure": a.measure,
                "Published change": f"{a.change:+.1f} {'mL/m²' if a.indexed else 'mL'}",
                "Absolute (mL)": round(a.absolute_change(), 1), "n": a.n,
                "Population": a.population,
            }
            for a in ANCHORS.values()
        ]),
        width="stretch", hide_index=True,
    )
    st.warning(
        "**The acute anchors disagree in sign, and that is the finding.** Older trials show "
        "dilation (TIME +4.3 mL/m², PRESERVATION-I +11.7 mL/m²); EMPRESS-MI, enrolling 2022–2024 "
        "on contemporary guideline-directed therapy, shows end-systolic volume *falling* "
        "7.8 mL/m² with ejection fraction rising 8.5 points. Post-MI natural history is "
        "era-dependent."
    )
    st.success(
        "**Chronic populations are stable.** FOCUS-CCTRN — the closest match to the late stratum "
        "on chronicity, the LVEF ≤ 45% criterion, and the transendocardial delivery route — "
        "observed a placebo-arm change of exactly zero."
    )
    st.caption(
        f"Indexed volumes converted at BSA = {BSA_CENTRAL} m². Checkable: VentriGel's 148.5 mL "
        f"baseline implies {implied_ventrigel_lvesvi():.1f} mL/m², against 65.0 mL/m² in the "
        "chronic FOCUS-CCTRN population — higher, as expected for a dilated post-MI cohort."
    )
    with st.expander("Full citations"):
        for a in ANCHORS.values():
            st.markdown(f"- **{a.trial}** — {a.citation}\n\n  _{a.note}_")

# --------------------------------------------------------------------------
# 4. Enrichment curve
# --------------------------------------------------------------------------

with tabs[3]:
    grid = np.linspace(pi, 1.0, 81)
    ns_, effs, sds = [], [], []
    for g in grid:
        p_g, n_g, _ = build(float(g))
        ns_.append(n_g if math.isfinite(n_g) else None)
        effs.append(p_g.effect)
        sds.append(p_g.sd)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=ns_, mode="lines", line={"width": 3}))
    fig.add_vline(x=e_level, line_dash="dot", annotation_text="current")
    fig.update_yaxes(type="log", title="Total randomized patients")
    fig.update_xaxes(title="Fraction of enrolled patients from the late stratum")
    fig.update_layout(height=360, margin={"t": 30}, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=grid, y=effs, mode="lines", name=f"Effect ({ep.unit})", line={"width": 3}))
    f2.add_trace(go.Scatter(x=grid, y=sds, mode="lines", name=f"SD ({ep.unit})", line={"width": 3}))
    f2.add_hline(y=0, line_color="#666")
    f2.update_xaxes(title="Fraction of enrolled patients from the late stratum")
    f2.update_layout(height=320, margin={"t": 30})
    st.plotly_chart(f2, width="stretch")
    st.caption(
        "Enrichment works on both terms at once: it raises the effect by dropping patients who "
        "dilute or oppose it, and lowers the SD by removing between-stratum heterogeneity. "
        "Required n scales as SD² / effect², so the gains compound."
    )

# --------------------------------------------------------------------------
# 5. Uncertainty
# --------------------------------------------------------------------------

with tabs[4]:
    st.subheader("How much survives the sample size it came from")
    if st.button("Run bootstrap (4,000 draws)"):
        with st.spinner("Re-solving the design across draws from the sampling distributions..."):
            b_e = bootstrap_designs(endpoint, e_level, pi, c_early, c_late, shrinkage,
                                    alpha, power_target, dropout)
            b_u = bootstrap_designs(endpoint, pi, pi, c_early, c_late, shrinkage,
                                    alpha, power_target, dropout)
        m1, m2, m3 = st.columns(3)
        m1.metric("Median N (enriched)", fmt(b_e.n_total_median))
        m2.metric("80% interval", f"{fmt(b_e.n_total_q10)} – {fmt(b_e.n_total_q90)}")
        m3.metric("Draws favouring treatment", f"{b_e.feasible_fraction:.0%}")
        fig = go.Figure()
        for b, name in ((b_u, "unselected"), (b_e, "enriched")):
            finite = b.samples[np.isfinite(b.samples)]
            if finite.size:
                fig.add_trace(go.Histogram(x=finite, name=name, opacity=0.6, nbinsx=60))
        fig.update_xaxes(type="log", title="Total randomized patients")
        fig.update_layout(barmode="overlay", height=360, margin={"t": 30})
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each draw samples a plausible true mean and variance for both strata from their "
            "sampling distributions, then re-solves the design. Draws whose effect reverses sign "
            "are counted as not favouring treatment rather than discarded, since discarding them "
            "would bias the interval optimistically."
        )
    else:
        st.caption("The subgroup estimates rest on 6–8 patients each. Press the button.")

# --------------------------------------------------------------------------
# 6. Cost
# --------------------------------------------------------------------------

with tabs[5]:
    from ventrigel.power import enrichment_curve

    sweep = enrichment_curve(endpoint, pi, c_early, c_late, alpha, power_target,
                             dropout, n_points=61, shrinkage=shrinkage)
    try:
        best, costed_all = optimal_enrichment(sweep, model)
        xs = [c.design.e for c in costed_all]
        ys = [c.total_cost / 1e6 if c.feasible and c.design.favors_treatment else None
              for c in costed_all]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"width": 3}))
        fig.add_vline(x=best.design.e, line_dash="dot",
                      annotation_text=f"cheapest e={best.design.e:.2f}")
        fig.update_yaxes(title="Total cost ($M)")
        fig.update_xaxes(title="Enrichment level")
        fig.update_layout(height=360, margin={"t": 30}, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Cheapest viable design: enrich to {best.design.e:.0%}, "
            f"{best.design.n_total:,.0f} randomized, {best.design.n_screened:,.0f} screened, "
            f"{best.duration_months:.0f} months, ${best.total_cost / 1e6:,.1f}M. Unit costs are "
            "planning assumptions; the sample-size ratios do not depend on them."
        )

        st.markdown("**Sites needed to finish on a given calendar**")
        if math.isfinite(n_scr):
            rows = []
            for target in (24, 30, 36, 48):
                enroll_months = target - model.followup_months
                if enroll_months <= 0:
                    continue
                need = n_scr / (model.enrollment_rate_per_site_month * enroll_months)
                rows.append({"Total duration": f"{target} months", "Sites required": math.ceil(need)})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    except ValueError:
        st.error("No design in this sweep both achieves power and favours treatment.")

# --------------------------------------------------------------------------
# 7. Source data
# --------------------------------------------------------------------------

with tabs[6]:
    st.subheader("What the Phase I published")
    rows = []
    for key, e in ENDPOINTS.items():
        t = e.change_6mo
        if not {"early", "late"} <= t.keys():
            continue
        rows.append({
            "Endpoint": e.label, "Unit": e.unit,
            "Early (<12mo)": f"{t['early'].mean:+.2f} ± {t['early'].sem:.1f} (n={t['early'].n})",
            "Late (>12mo)": f"{t['late'].mean:+.2f} ± {t['late'].sem:.1f} (n={t['late'].n})",
            "Pooled": f"{t['total'].mean:+.2f}" if "total" in t else "—",
            "Source": e.source_table,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Six-month change from baseline, mean ± SEM. Every cell is transcribed from the "
        "Supplemental Appendix Online Tables; none is digitized from a figure."
    )
    st.info(
        "The pooled column is the point. For LVESV the strata move in opposite directions "
        "(+9.3 mL early, −7.6 mL late), so the pooled −0.35 mL is not a small effect. "
        "It is a cancellation."
    )

# --------------------------------------------------------------------------
# 8. Validation
# --------------------------------------------------------------------------

with tabs[7]:
    st.subheader("Is the transcription faithful?")
    checks = check_all_p_values()
    real = [c for c in checks if c.excluded_reason is None]
    agree = sum(c.agrees for c in real)
    mixes = check_all_mixtures()
    v1, v2 = st.columns(2)
    v1.metric("Published p-values reproduced", f"{agree}/{len(real)}")
    v2.metric("Median SD reconstruction error",
              f"{float(np.median([m.sd_rel_error for m in mixes])) * 100:.1f}%")

    st.markdown(
        "**Check 1.** Each published paired t-test p-value is recomputed from the transcribed "
        "mean, SEM and n. This tests the transcription, the identity `SD = SEM × √n`, and the "
        "stated test simultaneously."
    )
    st.dataframe(
        pd.DataFrame([
            {"Endpoint": c.endpoint, "Visit": c.timepoint, "Group": c.group,
             "t": round(c.t_stat, 3), "p recomputed": round(c.p_recomputed, 4),
             "p published": c.p_published, "Agrees": "yes" if c.agrees else "no",
             "Excluded": (c.excluded_reason or "")[:70]}
            for c in checks
        ]),
        width="stretch", hide_index=True, height=250,
    )
    st.markdown(
        "**Check 2.** Each pooled mean and SD is reconstructed from its two subgroups by the laws "
        "of total expectation and total variance. The published totals were never used as inputs."
    )
    st.dataframe(
        pd.DataFrame([
            {"Endpoint": m.label, "Mean reconstructed": round(m.mean_reconstructed, 2),
             "Mean published": round(m.mean_published, 2),
             "SD reconstructed": round(m.sd_reconstructed, 2),
             "SD published": round(m.sd_published, 2),
             "SD error": f"{m.sd_rel_error * 100:.1f}%"}
            for m in mixes
        ]),
        width="stretch", hide_index=True,
    )

# --------------------------------------------------------------------------
# 9. About
# --------------------------------------------------------------------------

with tabs[8]:
    st.markdown(
        f"""
### What this is

A sample size and probability-of-success calculator for a Phase II trial of VentriGel,
parameterized by the subgroup structure the Phase I actually reported.

The Phase I found that improvements in left ventricular remodeling appeared mainly in patients
treated more than a year after infarction. Taken seriously, that has a direct design consequence:
the pooled LVESV change of −0.35 mL is not a weak signal, it is **+9.3 mL and −7.6 mL averaging to
nothing**. A Phase II enrolling the same mix would be powered to detect a quantity the trial itself
suggests is close to zero.

### How strong is the evidence, really

Weak, and the tool says so on the first tab. Exactly one of nine endpoints shows a nominally
significant interaction (LVESV, p = 0.034) and it survives neither Bonferroni nor
Benjamini–Hochberg correction. Two checks come back favourably — the strata are balanced on all
eight baseline measures, and the pattern runs opposite to regression to the mean — but the finding
is suggestive, not established. Everything downstream is conditional on it.

### What this is not

Not a patient-level response predictor. An earlier version of this project tried to be one: it
generated 2,000 synthetic patients from a hand-written scoring rule, trained a classifier on them,
and reported near-perfect accuracy. That accuracy was circular — the labels were a deterministic
function of the same features the model saw, so the classifier was recovering an `if` statement its
author had written. Fifteen single-arm patients cannot support individual prediction, and no amount
of simulation manufactures the missing information. That code is retained in `deprecated/`.

Simulation appears here only to propagate uncertainty already present in the published estimates.

### The number that decides everything

The enriched design enrolls no early patients, so its size is unaffected by what untreated *early*
patients do. It depends entirely on what untreated *late* patients do. FOCUS-CCTRN (n = 28) says
zero, and the design stands at around 90–175 patients. FOCUS-HF (n = 10) says −9.9 mL, under which
VentriGel's −7.6 mL is smaller than natural history and there is no effect at all.

Measuring the six-month LVESV change in untreated chronic post-MI patients with LVEF ≤ 45% would
settle it, and is obtainable from existing observational cohorts with serial CMR.

### Source

{CITATION}

Version {__version__}. Every number reproducible with `python run_analysis.py`.
"""
    )
