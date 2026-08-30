"""
Phase II trial design calculator for VentriGel.

    streamlit run app.py

This tool answers one question: given what the VentriGel Phase I actually
showed, how large would a Phase II have to be, and how much does restricting
enrollment to the late post-MI stratum change that?

It is deliberately not a patient-level predictor. Nothing here estimates
whether an individual will respond, because a 15-patient single-arm trial
cannot support that claim. It sizes trials, which is what the published data
can support.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ventrigel import __version__
from ventrigel.economics import CostModel, cost, optimal_enrichment
from ventrigel.power import achieved_power, design, enrichment_curve
from ventrigel.recovery import check_all_mixtures, check_all_p_values
from ventrigel.sensitivity import DEFAULT_SHRINKAGE, bootstrap_designs
from ventrigel.trial_data import (
    CANDIDATE_PRIMARY_ENDPOINTS,
    CITATION,
    ENDPOINTS,
    N_EARLY,
    N_LATE,
    TRIAL_ID,
)

PI_TRIAL = N_LATE / (N_EARLY + N_LATE)

st.set_page_config(
    page_title="VentriGel Phase II Design Calculator",
    page_icon="ecg",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      .stMetric {background: rgba(128,128,128,0.06); padding: 0.7rem 0.9rem; border-radius: 8px;}
      .caption-tight p {margin-bottom: 0.2rem; font-size: 0.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VentriGel Phase II trial design calculator")
st.caption(
    f"Built entirely from published summary statistics of {TRIAL_ID}. "
    "No synthetic patients, no patient-level prediction."
)


# --------------------------------------------------------------------------
# Inputs
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
    power = st.select_slider("Power", [0.70, 0.80, 0.85, 0.90, 0.95], value=0.80)
    dropout = st.slider("Dropout before primary assessment", 0.0, 0.35, 0.10, 0.01)

    st.header("Population")
    pi = st.slider(
        "Late-stratum prevalence in the eligible pool",
        0.03,
        0.95,
        float(round(PI_TRIAL, 2)),
        0.01,
        help=(
            "Fraction of screenable patients who are more than 12 months post-MI. "
            f"The trial's own split was {PI_TRIAL:.0%}, but it enrolled the two "
            "strata deliberately balanced, so that is not an estimate of the "
            "natural prevalence."
        ),
    )
    e_level = st.slider(
        "Enrichment: fraction of enrolled patients from the late stratum",
        float(round(pi, 2)),
        1.0,
        1.0,
        0.01,
    )

    st.header("Assumptions")
    st.caption(
        "The Phase I was single-arm, so control-arm drift is unmeasured. "
        "These two sliders are the dominant unknowns in the whole analysis."
    )
    unit = ep.unit
    lo, hi = (-15.0, 15.0) if unit in ("mL", "g") else (-40.0, 40.0)
    if unit == "%":
        lo, hi = -6.0, 6.0
    elif unit == "points":
        lo, hi = -25.0, 25.0
    c_early = st.slider(f"Control-arm change, early stratum ({unit})", lo, hi, 0.0, 0.5)
    c_late = st.slider(f"Control-arm change, late stratum ({unit})", lo, hi, 0.0, 0.5)
    shrinkage = st.slider(
        "Effect discount for winner's curse",
        0.30,
        1.00,
        DEFAULT_SHRINKAGE,
        0.05,
        help=(
            "The late subgroup was identified post hoc in the same 15 patients "
            "that produced its effect estimate, so the estimate is biased upward. "
            "1.00 takes the published effect at face value."
        ),
    )

    st.header("Costs")
    per_patient = st.number_input("Cost per randomized patient ($)", 10_000, 250_000, 65_000, 5_000)
    per_screen = st.number_input("Cost per screen ($)", 250, 25_000, 2_500, 250)
    n_sites = st.number_input("Sites", 1, 60, 6)
    rate = st.number_input("Enrollment rate (patients/site/month)", 0.1, 6.0, 0.75, 0.05)


# Apply the discount by scaling the published stratum means. Scaling happens
# before the control arm is subtracted, because the discount is a statement
# about the treatment-arm estimate, not about natural history.
def _scaled_design(e_val: float):
    from ventrigel.power import EnrichedPopulation, Stratum, n_per_arm_exact, n_screened
    from ventrigel.recovery import subgroup_effects

    eff = subgroup_effects(endpoint, "6mo")
    pop = EnrichedPopulation(
        early=Stratum("early", eff["early"]["mean"] * shrinkage, eff["early"]["sd"], c_early),
        late=Stratum("late", eff["late"]["mean"] * shrinkage, eff["late"]["sd"], c_late),
        e=e_val,
        lower_is_better=ep.lower_is_better,
    )
    if pop.effect <= 0:
        return pop, math.inf, math.inf
    n_arm = n_per_arm_exact(pop.effect, pop.sd, alpha, power)
    if not math.isfinite(n_arm):
        return pop, math.inf, math.inf
    n_arm = math.ceil(n_arm / max(1e-9, 1 - dropout))
    n_total = 2.0 * n_arm
    return pop, n_total, n_screened(n_total, e_val, pi)


pop, n_total, n_scr = _scaled_design(e_level)
pop_u, n_total_u, _ = _scaled_design(pi)

model = CostModel(
    per_patient=float(per_patient),
    per_screen=float(per_screen),
    n_sites=int(n_sites),
    enrollment_rate_per_site_month=float(rate),
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
c4.metric("Randomized patients", fmt(n_total))

if not math.isfinite(n_total):
    st.error(
        f"Under these assumptions the modelled effect is {pop.effect:+.2f} {ep.unit}, "
        "which does not favour treatment. No sample size demonstrates benefit; the "
        "design question is moot until the assumptions change."
    )
else:
    site_capacity = n_sites * rate
    months = n_scr / site_capacity if site_capacity else math.inf
    d = design(endpoint, e_level, pi, c_early, c_late, alpha, power, dropout)
    costed = cost(
        type(d)(
            endpoint=endpoint,
            e=e_level,
            pi=pi,
            effect=pop.effect,
            sd=pop.sd,
            sd_within=pop.sd_within,
            sd_between=pop.sd_between,
            standardized_effect=pop.standardized_effect,
            n_per_arm=n_total / 2,
            n_total=n_total,
            n_screened=n_scr,
            screens_per_enrolled=n_scr / n_total,
        ),
        model,
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Patients to screen", fmt(n_scr))
    d2.metric("Screens per enrollment", f"{n_scr / n_total:.1f}")
    d3.metric("Enrollment duration", f"{months:.0f} months")
    d4.metric("Estimated cost", f"${costed.total_cost / 1e6:,.1f}M")

    if math.isfinite(n_total_u) and n_total_u > 0:
        st.success(
            f"Enrichment to {e_level:.0%} late patients reduces the trial from "
            f"{fmt(n_total_u)} to {fmt(n_total)} randomized patients "
            f"({n_total_u / n_total:,.1f}x), at the price of "
            f"{n_scr / n_total:.1f} screens per enrollment."
        )
    else:
        st.success(
            f"An unselected trial has no benefit to detect under these assumptions "
            f"(pooled effect {pop_u.effect:+.2f} {ep.unit}). The enriched design needs "
            f"{fmt(n_total)} patients. The enrichment advantage is not a percentage "
            "here; it is the difference between a trial and no trial."
        )
    if months > 120:
        st.warning(
            f"Enrollment would take {months / 12:.0f} years at {n_sites} sites. "
            "When responders are rare, calendar time binds before sample size does."
        )


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_curve, tab_data, tab_unc, tab_cost, tab_valid, tab_about = st.tabs(
    ["Enrichment curve", "Source data", "Uncertainty", "Cost", "Validation", "What this is"]
)

with tab_curve:
    grid = np.linspace(pi, 1.0, 81)
    ns, effs, sds = [], [], []
    for g in grid:
        p_g, n_g, _ = _scaled_design(float(g))
        ns.append(n_g if math.isfinite(n_g) else None)
        effs.append(p_g.effect)
        sds.append(p_g.sd)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=ns, mode="lines", name="Randomized patients", line={"width": 3}))
    fig.add_vline(x=e_level, line_dash="dot", annotation_text="current")
    fig.update_yaxes(type="log", title="Total randomized patients")
    fig.update_xaxes(title="Fraction of enrolled patients from the late stratum")
    fig.update_layout(height=380, margin={"t": 30, "b": 40}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=grid, y=effs, mode="lines", name=f"Effect ({ep.unit})", line={"width": 3}))
    f2.add_trace(go.Scatter(x=grid, y=sds, mode="lines", name=f"SD ({ep.unit})", line={"width": 3}))
    f2.add_hline(y=0, line_color="#666")
    f2.update_xaxes(title="Fraction of enrolled patients from the late stratum")
    f2.update_layout(height=330, margin={"t": 30, "b": 40})
    st.plotly_chart(f2, use_container_width=True)
    st.caption(
        "Enrichment works on both terms at once: it raises the effect by dropping "
        "patients who dilute it, and lowers the SD by removing between-stratum "
        "heterogeneity. Required n scales as SD squared over effect squared."
    )

with tab_data:
    st.subheader("What the Phase I published")
    rows = []
    for key, e in ENDPOINTS.items():
        t = e.change_6mo
        if not {"early", "late"} <= t.keys():
            continue
        rows.append(
            {
                "Endpoint": e.label,
                "Unit": e.unit,
                "Early (<12mo)": f"{t['early'].mean:+.2f} (SEM {t['early'].sem:.1f}, n={t['early'].n})",
                "Late (>12mo)": f"{t['late'].mean:+.2f} (SEM {t['late'].sem:.1f}, n={t['late'].n})",
                "Pooled": f"{t['total'].mean:+.2f}" if "total" in t else "-",
                "Source": e.source_table,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Six-month change from baseline. Every cell is transcribed from the "
        "Supplemental Appendix Online Tables; none is digitized from a figure."
    )
    st.info(
        "The pooled column is the point of the whole analysis. For LVESV the "
        "strata move in opposite directions (+9.3 mL early, -7.6 mL late), so the "
        "pooled -0.35 mL is not a small effect. It is a cancellation."
    )

with tab_unc:
    st.subheader("How much of this survives the sample size it came from")
    if st.button("Run bootstrap (4,000 draws)"):
        with st.spinner("Re-solving the design across draws from the sampling distributions..."):
            b_e = bootstrap_designs(
                endpoint, e_level, pi, c_early, c_late, shrinkage, alpha, power, dropout
            )
            b_u = bootstrap_designs(
                endpoint, pi, pi, c_early, c_late, shrinkage, alpha, power, dropout
            )
        m1, m2, m3 = st.columns(3)
        m1.metric("Median N (enriched)", fmt(b_e.n_total_median))
        m2.metric("80% interval", f"{fmt(b_e.n_total_q10)} - {fmt(b_e.n_total_q90)}")
        m3.metric("Draws favouring treatment", f"{b_e.feasible_fraction:.0%}")
        fig = go.Figure()
        for b, name in ((b_u, "unselected"), (b_e, "enriched")):
            finite = b.samples[np.isfinite(b.samples)]
            if finite.size:
                fig.add_trace(go.Histogram(x=finite, name=name, opacity=0.6, nbinsx=60))
        fig.update_xaxes(type="log", title="Total randomized patients")
        fig.update_layout(barmode="overlay", height=360, margin={"t": 30})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each draw samples a plausible true mean and variance for both strata "
            "from their sampling distributions, then re-solves the design. Draws "
            "where the effect reverses sign are counted as not favouring treatment "
            "rather than discarded, because discarding them would bias the interval."
        )
    else:
        st.caption("The subgroup estimates rest on 6-8 patients each. Press the button.")

with tab_cost:
    sweep = enrichment_curve(endpoint, pi, c_early, c_late, alpha, power, dropout, n_points=61)
    try:
        best, costed_all = optimal_enrichment(sweep, model)
        xs = [c.design.e for c in costed_all]
        ys = [c.total_cost / 1e6 if c.feasible and c.design.favors_treatment else None for c in costed_all]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"width": 3}, name="Total cost"))
        fig.add_vline(x=best.design.e, line_dash="dot", annotation_text=f"cheapest e={best.design.e:.2f}")
        fig.update_yaxes(title="Total cost ($M)")
        fig.update_xaxes(title="Enrichment level")
        fig.update_layout(height=380, margin={"t": 30})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Cheapest viable design: enrich to {best.design.e:.0%} late, "
            f"{best.design.n_total:,.0f} randomized, {best.design.n_screened:,.0f} screened, "
            f"{best.duration_months:.0f} months, ${best.total_cost / 1e6:,.1f}M. "
            "Unit costs are planning assumptions, not measurements; the sample-size "
            "ratios above do not depend on them."
        )
    except ValueError:
        st.error("No design in this sweep both achieves power and favours treatment.")

with tab_valid:
    st.subheader("Is the transcription faithful?")
    checks = check_all_p_values()
    real = [c for c in checks if c.excluded_reason is None]
    agree = sum(c.agrees for c in real)
    v1, v2 = st.columns(2)
    v1.metric("Published p-values reproduced", f"{agree}/{len(real)}")
    mixes = check_all_mixtures()
    v2.metric(
        "Median SD reconstruction error",
        f"{float(np.median([m.sd_rel_error for m in mixes])) * 100:.1f}%",
    )
    st.markdown(
        "**Check 1.** Each published paired t-test p-value is recomputed from the "
        "transcribed mean, SEM and n. This tests the transcription, the identity "
        "`SD = SEM x sqrt(n)`, and the stated test simultaneously."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Endpoint": c.endpoint,
                    "Visit": c.timepoint,
                    "Group": c.group,
                    "t": round(c.t_stat, 3),
                    "p recomputed": round(c.p_recomputed, 4),
                    "p published": c.p_published,
                    "Agrees": "yes" if c.agrees else "no",
                    "Excluded": (c.excluded_reason or "")[:60],
                }
                for c in checks
            ]
        ),
        use_container_width=True,
        hide_index=True,
        height=260,
    )
    st.markdown(
        "**Check 2.** Each total-cohort mean and SD is reconstructed from its two "
        "subgroups by the laws of total expectation and total variance. The "
        "published totals were never used as inputs, so agreement is a real test."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Endpoint": m.label,
                    "Mean reconstructed": round(m.mean_reconstructed, 2),
                    "Mean published": round(m.mean_published, 2),
                    "SD reconstructed": round(m.sd_reconstructed, 2),
                    "SD published": round(m.sd_published, 2),
                    "SD error": f"{m.sd_rel_error * 100:.1f}%",
                }
                for m in mixes
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_about:
    st.markdown(
        f"""
### What this is

A sample size calculator for a Phase II trial of VentriGel, parameterized by
the subgroup structure the Phase I actually reported.

The Phase I found that improvements in left ventricular remodeling appeared
mainly in patients treated more than a year after their infarction. Taken
seriously, that finding has a direct consequence for trial design: the pooled
LVESV change of -0.35 mL is not a weak signal, it is a **+9.3 mL and a -7.6 mL
averaging to nothing**. A Phase II that enrolls the same mix would be powered
to detect a quantity the trial itself suggests is close to zero.

### What this is not

It is not a patient-level response predictor, and an earlier version of this
project that tried to be one has been retired to `deprecated/`. That version
generated 2,000 synthetic patients from a hand-written scoring rule, trained a
classifier on them, and reported near-perfect accuracy. The accuracy was
circular: the labels were a deterministic function of the same features the
model saw, so the classifier was recovering an `if` statement its author had
written. Fifteen single-arm patients cannot support individual prediction, and
no amount of simulation manufactures the information.

Simulation appears in this version only in the uncertainty tab, and only to
propagate variance that is already present in the published estimates.

### The caveat that matters most

The Phase I was **single-arm**. It shows what happened to treated patients, not
what would have happened untreated. The early-versus-late gap is therefore
consistent with genuine effect modification *or* with untreated early patients
dilating on their own while late patients stay stable. Those are not
distinguishable from these data.

That is why control-arm drift is a slider rather than a footnote. Set the
early-stratum control drift to +9.3 mL and the entire early-group change is
attributed to natural history; the enrichment advantage falls from enormous to
roughly fivefold, and the enriched trial still needs about 50-100 patients.
The robust conclusion is the enriched design's size, not the size of the
advantage over an unselected one.

### Source

{CITATION}

Version {__version__}. Every number is reproducible with `python run_analysis.py`.
"""
    )
