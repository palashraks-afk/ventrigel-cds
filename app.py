"""
Phase II trial design calculator for VentriGel.

    streamlit run app.py

Given what the VentriGel Phase I actually showed: how large would a Phase II
have to be, how likely is it to succeed, and which assumption is that answer
most sensitive to?

Deliberately not a patient-level predictor. Nothing here estimates whether an
individual will respond, because a 15-patient single-arm trial cannot support
that claim.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ventrigel import __version__
from ventrigel.assurance import (
    assurance_at_enrichment,
    assurance_curve_at_enrichment,
)
from ventrigel.economics import CostModel, cost
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
from ventrigel.power import (
    Design,
    EnrichedPopulation,
    Stratum,
    interaction_design,
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

st.set_page_config(
    page_title="VentriGel Phase II Design Calculator",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; max-width: 1320px;}
      div[data-testid="stMetric"] {
          background: rgba(128,128,128,0.07);
          border: 1px solid rgba(128,128,128,0.16);
          padding: 0.75rem 0.95rem; border-radius: 10px;
      }
      div[data-testid="stMetricValue"] {font-size: 1.65rem; font-weight: 600;}
      div[data-testid="stMetricLabel"] {opacity: 0.75;}
      .stTabs [data-baseweb="tab"] {padding: 0.45rem 0.95rem;}
      .verdict {
          border-left: 4px solid #D98C1F; padding: 0.75rem 1rem;
          background: rgba(217,140,31,0.08); border-radius: 6px;
          margin: 0.4rem 0 1.0rem 0; font-size: 0.93rem; line-height: 1.5;
      }
      .keyfig {font-size: 1.05rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VentriGel Phase II trial design calculator")
st.caption(
    f"Built from published summary statistics of {TRIAL_ID} plus control arms of "
    "eight other published sources. No synthetic patients, no patient-level prediction."
)

_ev = assess_evidence()
st.markdown(
    f'<div class="verdict"><b>Read this first.</b> Everything in this tool is '
    f"conditional on a subgroup effect that is <b>nominally significant for one of nine "
    f"endpoints (p = {_ev.strongest_p:.3f}) and survives no multiplicity correction</b>. "
    "The strata are balanced at baseline and the pattern is not regression to the mean, "
    "so it is worth designing around, but it is not established. The last tab converts "
    "that into an unconditional probability of success.</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Design")
    endpoint = st.selectbox(
        "Primary endpoint", CANDIDATE_PRIMARY_ENDPOINTS,
        format_func=lambda k: ENDPOINTS[k].label, index=0,
    )
    ep = ENDPOINTS[endpoint]
    alpha = st.select_slider("Two-sided alpha", [0.01, 0.025, 0.05, 0.10], value=0.05)
    power_target = st.select_slider("Power", [0.70, 0.80, 0.85, 0.90, 0.95], value=0.80)
    dropout = st.slider("Dropout before primary assessment", 0.0, 0.35, 0.10, 0.01)

    st.header("Population")
    pi = st.slider(
        "Late-stratum prevalence in the eligible pool", 0.03, 0.95,
        float(round(PI_TRIAL, 2)), 0.01,
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
    _ae = anchored_control(endpoint, "early")
    _al = anchored_control(endpoint, "late")
    use_anchors = st.checkbox("Use published control arms", value=True)

    if use_anchors:
        c_early, se_early = _ae or (0.0, 0.0)
        c_late, se_late = _al or (0.0, 0.0)
        bits = []
        bits.append(
            f"early {c_early:+.1f} ± {se_early:.1f}" if _ae else "early: **no anchor** (assumes 0)"
        )
        bits.append(
            f"late {c_late:+.1f} ± {se_late:.1f}" if _al else "late: **no anchor** (assumes 0)"
        )
        st.caption(f"{ep.unit} · " + " · ".join(bits))
        if not (_ae and _al):
            st.warning(
                "This endpoint is only partly anchored. Unanchored strata assume no "
                "control drift, which flatters the treatment.", icon="⚠️",
            )
    else:
        span = {"%": 8.0, "points": 25.0, "g": 30.0, "m": 45.0}.get(ep.unit, 20.0)
        c_early = st.slider(f"Control change, early ({ep.unit})", -span, span,
                            float(_ae[0]) if _ae else 0.0, 0.5)
        c_late = st.slider(f"Control change, late ({ep.unit})", -span, span,
                           float(_al[0]) if _al else 0.0, 0.5)
        se_early = se_late = 0.0

    propagate = st.checkbox(
        "Propagate the anchors' own uncertainty", value=True,
        help=(
            "The comparator is an estimate from a few dozen patients, not a constant. "
            "Ignoring its standard error roughly halves the apparent trial size."
        ),
    )
    if not propagate:
        se_early = se_late = 0.0

    shrinkage = st.slider(
        "Effect discount for winner's curse", 0.30, 1.00, 0.75, 0.05,
        help="The late subgroup was identified post hoc in the same 15 patients.",
    )

    st.header("Operations")
    per_patient = st.number_input("Cost per randomized patient ($)", 10_000, 250_000, 65_000, 5_000)
    per_screen = st.number_input("Cost per screen ($)", 250, 25_000, 2_500, 250)
    n_sites = st.number_input("Sites", 1, 120, 6)
    rate = st.number_input("Enrollment rate (patients/site/month)", 0.1, 6.0, 0.75, 0.05)


# --------------------------------------------------------------------------
# Core model
# --------------------------------------------------------------------------


def build(e_val: float) -> tuple[EnrichedPopulation, float, float]:
    eff = subgroup_effects(endpoint, "6mo")
    pop = EnrichedPopulation(
        early=Stratum("early", eff["early"]["mean"], eff["early"]["sd"], c_early, shrinkage),
        late=Stratum("late", eff["late"]["mean"], eff["late"]["sd"], c_late, shrinkage),
        e=e_val, lower_is_better=ep.lower_is_better,
    )
    if pop.effect <= 0:
        return pop, math.inf, math.inf
    n_arm = n_per_arm_exact(pop.effect, pop.sd, alpha, power_target)
    if not math.isfinite(n_arm):
        return pop, math.inf, math.inf
    n_arm = math.ceil(n_arm / max(1e-9, 1 - dropout))
    return pop, 2.0 * n_arm, n_screened(2.0 * n_arm, e_val, pi)


pop, n_total, n_scr = build(e_level)
pop_u, n_total_u, _ = build(pi)
model = CostModel(
    per_patient=float(per_patient), per_screen=float(per_screen),
    n_sites=int(n_sites), enrollment_rate_per_site_month=float(rate),
)


def fmt(x: float) -> str:
    return "not achievable" if not math.isfinite(x) else f"{x:,.0f}"


def costed_design(n: float, screened: float) -> tuple[Design, object]:
    d = Design(
        endpoint=endpoint, e=e_level, pi=pi, effect=pop.effect, sd=pop.sd,
        sd_within=pop.sd_within, sd_between=pop.sd_between,
        standardized_effect=pop.standardized_effect, n_per_arm=n / 2,
        n_total=n, n_screened=screened,
        screens_per_enrolled=screened / n if n else float("inf"),
    )
    return d, cost(d, model)


# --------------------------------------------------------------------------
# Headline
# --------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Effect vs. control", f"{pop.effect:+.2f} {ep.unit}")
c2.metric("SD of change", f"{pop.sd:.2f} {ep.unit}")
c3.metric("Standardized effect", f"{pop.standardized_effect:.2f}")
c4.metric(f"N for {power_target:.0%} nominal power", fmt(n_total))

if not math.isfinite(n_total):
    st.error(
        f"Under these assumptions the modelled effect is {pop.effect:+.2f} {ep.unit}, which "
        "does not favour treatment. No sample size demonstrates benefit."
    )
else:
    months = n_scr / (n_sites * rate) if n_sites * rate else math.inf
    d, costed = costed_design(n_total, n_scr)
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
            f"An unselected trial has no benefit to detect here (pooled effect "
            f"{pop_u.effect:+.2f} {ep.unit}). The enriched design needs {fmt(n_total)} patients."
        )
    if months > 120:
        st.warning(f"Enrollment would take {months / 12:.0f} years at {n_sites} sites.")

st.divider()

tabs = st.tabs([
    "1 · Does the effect exist?",
    "2 · Control arms",
    "3 · Enrichment",
    "4 · Probability of success",
    "5 · Confirming the claim",
    "6 · Uncertainty",
    "7 · Source data",
    "8 · Validation",
    "About",
])

# -- 1. Evidence -----------------------------------------------------------

with tabs[0]:
    st.subheader("The test the trial did not run")
    st.markdown(
        "The Phase I compared each stratum against its **own baseline** and observed that one "
        "reached significance while the other did not. That is not a test of effect "
        "modification: two subgroups can land on opposite sides of p = 0.05 without differing "
        "from each other."
    )
    tests = all_interaction_tests()
    mult = {m.endpoint: m for m in multiplicity(tests)}
    st.dataframe(
        pd.DataFrame([
            {
                "Endpoint": t.label,
                "Early": f"{t.early_mean:+.2f} (n={t.early_n})",
                "Late": f"{t.late_mean:+.2f} (n={t.late_n})",
                "Difference": f"{t.difference:+.2f}",
                "95% CI": f"[{t.ci_low:.1f}, {t.ci_high:.1f}]",
                "p": round(t.p_value, 4),
                "Bonferroni": "pass" if mult[t.endpoint].bonferroni_pass else "fail",
                "BH": "pass" if mult[t.endpoint].bh_pass else "fail",
            } for t in tests
        ]),
        width="stretch", hide_index=True,
    )
    nominal, effective, why = effective_n_tests()
    st.error(f"**Verdict.** {_ev.verdict}")
    st.caption(why)
    st.info(
        "**Two caveats on the test itself**, neither resolvable without patient-level data. "
        "It compares change scores rather than adjusting for baseline by ANCOVA, and with "
        "balanced baselines ANCOVA is the standard and more powerful choice, so p = 0.034 is "
        "probably conservative. And the family is nine endpoints at the 6-month visit; the "
        "trial also reported 1- and 3-month visits, and counting those would enlarge the "
        "family and weaken the result further."
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Baseline balance.** The strata were not randomized against each other.")
        bal = baseline_balance()
        st.dataframe(
            pd.DataFrame([
                {"Measure": b.label, "Early": round(b.early_mean, 1),
                 "Late": round(b.late_mean, 1), "p": round(b.p_value, 3)} for b in bal
            ]),
            width="stretch", hide_index=True, height=250,
        )
        st.success(f"All {len(bal)} balanced (minimum p = {min(b.p_value for b in bal):.2f}).")
    with cc2:
        st.markdown("**Regression to the mean.** The most obvious artifact.")
        rtm = regression_to_mean_check("lvesv")
        st.write(rtm.explanation)
        if rtm.contradicts_rtm:
            st.success("Ruled out: the pattern runs opposite to what regression produces.")

# -- 2. Control arms -------------------------------------------------------

with tabs[1]:
    st.subheader("What happens to untreated patients")
    st.markdown(
        "The Phase I was **single-arm**, so its comparator is missing. Rather than sweeping an "
        "arbitrary range, control-arm change is anchored to published control and placebo arms."
    )
    st.dataframe(
        pd.DataFrame([
            {
                "Trial": a.trial, "Year": a.year, "Endpoint": a.endpoint, "Phase": a.phase,
                "Published change": f"{a.change:+.1f}"
                + (" %" if a.percent else (" mL/m²" if a.indexed else f" {a.measure}")),
                "Absolute": round(a.absolute_change(), 1),
                "SE": None if a.standard_error() is None else round(a.standard_error(), 2),
                "n": a.n, "Evidence": a.evidence,
            } for a in ANCHORS.values()
        ]),
        width="stretch", hide_index=True,
    )
    st.warning(
        "**The acute anchors disagree in sign, and that is the finding.** Older trials show "
        "dilation (TIME +4.3 mL/m², PRESERVATION-I +11.7 mL/m²); EMPRESS-MI, enrolling "
        "2022-2024 on contemporary therapy, shows end-systolic volume *falling* 7.8 mL/m² with "
        "ejection fraction rising 8.5 points. Post-MI natural history is era-dependent."
    )

    st.markdown("**Anchor coverage.** Cells marked NONE assume no control drift.")
    cov = anchor_coverage()
    st.dataframe(
        pd.DataFrame([
            {"Endpoint": ENDPOINTS[k].label,
             "Early stratum": cov.get(k, {}).get("early") or "NONE",
             "Late stratum": cov.get(k, {}).get("late") or "NONE"}
            for k in CANDIDATE_PRIMARY_ENDPOINTS
        ]),
        width="stretch", hide_index=True,
    )

    st.markdown("**Where two anchors compete, the choice is explicit, not automatic.**")
    for (e_, s_), whytext in ANCHOR_CHOICE_RATIONALE.items():
        with st.expander(f"{ENDPOINTS[e_].label} · {s_} stratum"):
            st.write(whytext)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"**Body surface area** (assumed {BSA_CENTRAL} m²)")
        st.dataframe(
            pd.DataFrame([{"BSA": b, "Change (mL)": round(ch, 2), "SE (mL)": round(se, 2)}
                          for b, ch, se in bsa_sensitivity()]),
            width="stretch", hide_index=True, height=230,
        )
        st.caption(
            f"VentriGel's 148.5 mL baseline implies {implied_ventrigel_lvesvi():.1f} mL/m². "
            "The late anchor's point estimate is exactly zero at every BSA; only its SE moves."
        )
    with a2:
        st.markdown(f"**Test-retest correlation** (assumed {DEFAULT_RETEST_R})")
        st.dataframe(
            pd.DataFrame([{"r": r, "Anchor SE (mL)": round(se, 2)}
                          for r, se in retest_sensitivity()]),
            width="stretch", hide_index=True, height=230,
        )
        st.caption(
            "FOCUS-CCTRN publishes SDs of levels, not of the change, so the change SD depends "
            "on a correlation nobody reports. This is what that choice costs."
        )

# -- 3. Enrichment ---------------------------------------------------------

with tabs[2]:
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
    fig.update_layout(height=350, margin={"t": 30}, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=grid, y=effs, mode="lines", name=f"Effect ({ep.unit})", line={"width": 3}))
    f2.add_trace(go.Scatter(x=grid, y=sds, mode="lines", name=f"SD ({ep.unit})", line={"width": 3}))
    f2.add_hline(y=0, line_color="#666")
    f2.update_xaxes(title="Fraction of enrolled patients from the late stratum")
    f2.update_layout(height=310, margin={"t": 30})
    st.plotly_chart(f2, width="stretch")
    st.caption(
        "Enrichment works on both terms at once: it raises the effect by dropping patients who "
        "dilute or oppose it, and lowers the SD by removing between-stratum heterogeneity. "
        "Required n scales as SD² / effect², so the gains compound."
    )

# -- 4. Probability of success --------------------------------------------

with tabs[3]:
    st.subheader("Power is a conditional promise; assurance is not")
    st.markdown(
        "Sizing for 80% power *at a point estimate* and calling the result an 80% trial "
        "promotes an assumption to a fact. **Assurance** integrates power over the uncertainty "
        "in the effect and in the comparator, at the enrichment level actually selected."
    )
    ns = np.unique(np.round(np.logspace(math.log10(24), math.log10(4000), 28)).astype(int))
    vals, ceiling = assurance_curve_at_enrichment(
        endpoint, ns, e_level, shrinkage, c_early, c_late, se_early, se_late,
        alpha, n_draws=12000,
    )
    vals_exact, ceil_exact = assurance_curve_at_enrichment(
        endpoint, ns, e_level, shrinkage, c_early, c_late, 0.0, 0.0, alpha, n_draws=12000,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=vals_exact, mode="lines", name="comparator treated as exact",
                             line={"width": 2, "dash": "dash", "color": "#9AA5B1"}))
    fig.add_trace(go.Scatter(x=ns, y=vals, mode="lines", name="comparator uncertainty propagated",
                             line={"width": 3, "color": "#2E6E9E"}))
    fig.add_hline(y=ceiling, line_dash="dot", annotation_text=f"ceiling {ceiling:.0%}")
    if math.isfinite(n_total) and n_total > 0:
        fig.add_vline(x=math.log10(n_total), line_dash="dot", annotation_text="current design")
    fig.update_xaxes(type="log", title="Total randomized patients")
    fig.update_yaxes(title="Probability of a significant result", range=[0, 1.02])
    fig.update_layout(height=400, margin={"t": 30}, legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(fig, width="stretch")

    if math.isfinite(n_total):
        r = assurance_at_enrichment(endpoint, n_total, e_level, shrinkage, c_early, c_late,
                                    se_early, se_late, alpha, n_draws=12000)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Nominal power target", f"{power_target:.0%}")
        m2.metric("Actual probability of success", f"{r.assurance:.0%}")
        m3.metric("Achievable ceiling", f"{ceiling:.0%}")

    def n_for(target: float) -> float:
        above = np.where(vals >= target)[0]
        return float(ns[above[0]]) if above.size else math.inf

    st.dataframe(
        pd.DataFrame([
            {"Target probability of success": f"{t:.0%}",
             "Patients required": "beyond ceiling" if not math.isfinite(n_for(t)) else f"{n_for(t):,.0f}"}
            for t in (0.5, 0.6, 0.7, 0.8, 0.85)
        ]),
        width="stretch", hide_index=True,
    )

    st.divider()
    st.subheader("And the number a sponsor actually needs")
    st.markdown(
        "All of the above is **conditional on the subgroup effect being real**. Unconditional "
        "probability of success is that prior times the assurance."
    )
    prior = st.slider(
        "Probability the subgroup effect is real", 0.05, 1.00, 0.50, 0.05,
        help=(
            "A nominal p of 0.034 on one of nine tests surviving no correction is weak "
            "evidence; balanced baselines and the failure of regression to the mean pull the "
            "other way. Somewhere between 0.3 and 0.6 is defensible."
        ),
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ns, y=vals * prior, mode="lines", line={"width": 3},
                              name=f"prior {prior:.0%}"))
    fig2.add_trace(go.Scatter(x=ns, y=vals, mode="lines", name="conditional (effect certain)",
                              line={"width": 2, "dash": "dot", "color": "#9AA5B1"}))
    fig2.add_hline(y=prior, line_dash="dash", annotation_text=f"prior ceiling {prior:.0%}")
    fig2.update_xaxes(type="log", title="Total randomized patients")
    fig2.update_yaxes(title="Unconditional probability of success", range=[0, 1.02])
    fig2.update_layout(height=340, margin={"t": 30}, legend={"orientation": "h", "y": -0.25})
    st.plotly_chart(fig2, width="stretch")
    if math.isfinite(n_total):
        st.metric("Unconditional probability of success at this design",
                  f"{r.assurance * prior:.0%}")
    st.info(
        "The prior enters multiplicatively, so **no sample size lifts the programme above it**. "
        "That is the argument for spending the next increment of money on measuring the "
        "comparator rather than on more patients."
    )

# -- 5. Confirming the claim ----------------------------------------------

with tabs[4]:
    st.subheader("The enriched trial cannot confirm the claim it rests on")
    st.markdown(
        "A trial enrolling only late patients can show the therapy works **in that stratum**. "
        "It can never show that *timing matters*, because it contains no early patients to "
        "compare against. But \"treat late, not early\" is the actual claim, and it is what a "
        "sponsor would be acting on. Confirming it needs a 2×2 trial powered on the interaction."
    )
    rows = []
    for label, (a_, b_) in (
        ("no control drift", (0.0, 0.0)),
        ("published control arms", (c_early, c_late)),
    ):
        d_ = interaction_design(endpoint, a_, b_, alpha, power_target, dropout, shrinkage)
        rows.append({
            "Control assumption": label,
            f"Interaction contrast ({ep.unit})": round(d_.contrast, 2),
            "N per cell": "n/a" if not d_.feasible else f"{d_.n_per_cell:,.0f}",
            "N total (2×2)": "n/a" if not d_.feasible else f"{d_.n_total:,.0f}",
            "Enriched 2-arm N": f"{d_.n_enriched_reference:,.0f}",
            "Ratio": "n/a" if not d_.feasible else f"{d_.ratio_to_enriched:.1f}×",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown(
        "With four equally sized cells the interaction estimate carries **twice the variance** "
        "of a simple two-arm comparison, which is the familiar reason interactions are "
        "expensive. That penalty is offset here because the contrast being detected is larger "
        "than the late-stratum effect alone, but anchoring the comparator roughly halves the "
        "contrast, because most of the early stratum's apparent harm turns out to be natural "
        "history rather than a failure of treatment."
    )
    st.success(
        "**The practical consequence.** Against published control arms the 2×2 design costs "
        "about 440 patients, close to the ~406 an 80%-assurance enriched trial needs. For "
        "roughly the same money a sponsor can answer the question they actually have instead "
        "of half of it."
    )

# -- 6. Uncertainty --------------------------------------------------------

with tabs[5]:
    st.subheader("How much survives the sample size it came from")
    if st.button("Run bootstrap (4,000 draws)"):
        with st.spinner("Re-solving the design across draws from the sampling distributions..."):
            b_e = bootstrap_designs(endpoint, e_level, pi, c_early, c_late, shrinkage,
                                    alpha, power_target, dropout)
            b_u = bootstrap_designs(endpoint, pi, pi, c_early, c_late, shrinkage,
                                    alpha, power_target, dropout)
        m1, m2, m3 = st.columns(3)
        if b_e.feasible_fraction < 0.05:
            m1.metric("Median N (enriched)", "n/a")
            m2.metric("80% interval", "n/a")
        else:
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
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Draws whose effect reverses sign are counted as not favouring treatment rather "
            "than discarded, since discarding them would bias the interval optimistically."
        )
    else:
        st.caption("The subgroup estimates rest on 6-8 patients each. Press the button.")

    st.divider()
    st.markdown("**Sites needed to finish on a given calendar**")
    if math.isfinite(n_scr):
        st.dataframe(
            pd.DataFrame([
                {"Total duration": f"{t} months",
                 "Sites required": math.ceil(
                     n_scr / (model.enrollment_rate_per_site_month * (t - model.followup_months))
                 )}
                for t in (24, 30, 36, 48) if t > model.followup_months
            ]),
            width="stretch", hide_index=True,
        )

# -- 7. Source data --------------------------------------------------------

with tabs[6]:
    st.subheader("What the Phase I published")
    st.dataframe(
        pd.DataFrame([
            {
                "Endpoint": e.label, "Unit": e.unit,
                "Early (<12mo)": f"{t['early'].mean:+.2f} ± {t['early'].sem:.1f} (n={t['early'].n})",
                "Late (>12mo)": f"{t['late'].mean:+.2f} ± {t['late'].sem:.1f} (n={t['late'].n})",
                "Pooled": f"{t['total'].mean:+.2f}" if "total" in t else "n/a",
                "Source": e.source_table,
            }
            for _, e in ENDPOINTS.items()
            if {"early", "late"} <= (t := e.change_6mo).keys()
        ]),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Six-month change from baseline, mean ± SEM. Every cell is transcribed from the "
        "Supplemental Appendix Online Tables; none is digitized from a figure."
    )
    st.info(
        "For LVESV the strata move in opposite directions (+9.3 mL early, −7.6 mL late), so "
        "the pooled −0.35 mL is not a small effect. It is a cancellation."
    )

# -- 8. Validation ---------------------------------------------------------

with tabs[7]:
    st.subheader("Is the transcription faithful?")
    checks = check_all_p_values()
    real = [c for c in checks if c.excluded_reason is None]
    mixes = check_all_mixtures()
    v1, v2 = st.columns(2)
    v1.metric("Published p-values reproduced", f"{sum(c.agrees for c in real)}/{len(real)}")
    v2.metric("Median SD reconstruction error",
              f"{float(np.median([m.sd_rel_error for m in mixes])) * 100:.1f}%")
    st.markdown(
        "**Check 1.** Each published paired t-test p-value is recomputed from the transcribed "
        "mean, SEM and n, testing the transcription, the identity `SD = SEM × √n`, and the "
        "stated test simultaneously."
    )
    st.dataframe(
        pd.DataFrame([
            {"Endpoint": c.endpoint, "Visit": c.timepoint, "Group": c.group,
             "t": round(c.t_stat, 3), "p recomputed": round(c.p_recomputed, 4),
             "p published": c.p_published, "Agrees": "yes" if c.agrees else "no",
             "Excluded": (c.excluded_reason or "")[:70]} for c in checks
        ]),
        width="stretch", hide_index=True, height=250,
    )
    st.markdown(
        "**Check 2.** Each pooled mean and SD is reconstructed from its two subgroups by the "
        "laws of total expectation and total variance. The published totals were never inputs."
    )
    st.dataframe(
        pd.DataFrame([
            {"Endpoint": m.label, "Mean reconstructed": round(m.mean_reconstructed, 2),
             "Mean published": round(m.mean_published, 2),
             "SD reconstructed": round(m.sd_reconstructed, 2),
             "SD published": round(m.sd_published, 2),
             "SD error": f"{m.sd_rel_error * 100:.1f}%"} for m in mixes
        ]),
        width="stretch", hide_index=True,
    )

# -- About -----------------------------------------------------------------

with tabs[8]:
    st.markdown(
        f"""
### What this is

A sample size and probability-of-success calculator for a Phase II trial of VentriGel,
parameterized by the subgroup structure the Phase I actually reported.

The Phase I found that improvements in left ventricular remodeling appeared mainly in patients
treated more than a year after infarction. Taken seriously, that has a direct design consequence:
the pooled LVESV change of −0.35 mL is not a weak signal, it is **+9.3 mL and −7.6 mL averaging
to nothing**.

### How strong is the evidence

Weak, and the first tab says so. Exactly one of nine endpoints shows a nominally significant
interaction (p = 0.034) and it survives neither Bonferroni nor Benjamini-Hochberg. Two checks
come back favourably, namely balanced baselines and a pattern running opposite to regression to
the mean, but the finding is suggestive, not established.

### What this is not

Not a patient-level response predictor. An earlier version of this project tried to be one: it
generated 2,000 synthetic patients from a hand-written scoring rule, trained a classifier, and
reported near-perfect accuracy. That accuracy was circular. The labels were a deterministic
function of the same features the model saw. Fifteen single-arm patients cannot support
individual prediction. That code is retained in `deprecated/` with a written post-mortem, and
`CORRECTION.md` is the retraction notice.

Simulation appears here only to propagate uncertainty already present in published estimates.

### The number that decides everything

The enriched design enrolls no early patients, so its size depends entirely on what untreated
*late* patients do. FOCUS-CCTRN (n = 28) says zero; FOCUS-HF (n = 10) says −9.9 mL, under which
VentriGel's −7.6 mL is smaller than natural history and there is no effect at all.

That anchor also carries its own standard error of about 4 mL, the same order as the effect
being measured against it. Propagating it roughly doubles the required trial, which is why the
"propagate the anchors' own uncertainty" box is on by default.

Measuring the six-month LVESV change in untreated chronic post-MI patients with LVEF ≤ 45% would
settle it, and is obtainable from existing observational cohorts with serial CMR.

### Source

{CITATION}

Version {__version__}. Every number reproducible with `python run_analysis.py`.
"""
    )
