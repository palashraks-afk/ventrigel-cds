"""
Tests for the enrichment analysis.

    python -m pytest test_ventrigel.py -q     (or: python test_ventrigel.py)

The power calculations are checked against textbook reference values rather
than against themselves, so a regression in the solver shows up as a failure
instead of as a quietly different answer.
"""

from __future__ import annotations

import math

import numpy as np

from ventrigel.assurance import (
    assurance,
    assurance_ceiling,
    assurance_curve,
    n_for_assurance,
    power_at,
    power_at_vec,
)
from ventrigel.economics import CostModel, cost, optimal_enrichment
from ventrigel.inference import (
    all_interaction_tests,
    assess_evidence,
    baseline_balance,
    effective_n_tests,
    interaction_test,
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
    achieved_power,
    build_population,
    design,
    enrichment_curve,
    n_per_arm_exact,
    n_per_arm_normal,
    n_screened,
)
from ventrigel.recovery import check_all_mixtures, check_all_p_values, mixture_moments
from ventrigel.sensitivity import bootstrap_designs, shrinkage_curve
from ventrigel.trial_data import ENDPOINTS, LVESV, Measurement

PI = 8 / 15


# -- the SD identity --------------------------------------------------------


def test_sd_recovery_identity():
    m = Measurement(mean=-7.6, sem=3.2, n=8)
    assert math.isclose(m.sd, 3.2 * math.sqrt(8), rel_tol=1e-12)


def test_published_sd_matches_hand_calculation():
    # LVESV late stratum, Online Table 7: -7.6 (3.2), n = 8.
    assert math.isclose(LVESV.change_6mo["late"].sd, 9.0510, abs_tol=1e-3)


# -- validation of the transcription ---------------------------------------


def test_all_checkable_pvalues_reproduce():
    checks = [c for c in check_all_p_values() if c.excluded_reason is None]
    failures = [c for c in checks if not c.agrees]
    assert not failures, [
        (c.endpoint, c.timepoint, c.group, c.p_recomputed, c.p_published) for c in failures
    ]
    assert len(checks) >= 35


def test_mixture_reconstructs_published_totals():
    for m in check_all_mixtures():
        # Every endpoint's pooled SD must be recoverable from its strata to
        # within 10%; LVESV, which carries the headline result, much tighter.
        assert m.sd_rel_error < 0.10, (m.endpoint, m.sd_rel_error)
    lvesv = next(m for m in check_all_mixtures() if m.endpoint == "lvesv")
    assert lvesv.sd_rel_error < 0.01
    assert abs(lvesv.mean_reconstructed - lvesv.mean_published) < 0.05


def test_mixture_moments_against_pooled_samples():
    """The analytic mixture formula must match brute-force pooling."""
    rng = np.random.default_rng(0)
    a = rng.normal(9.3, 14.2, 400_000)
    b = rng.normal(-7.6, 9.05, 600_000)
    pooled = np.concatenate([a, b])
    mean, _, _, sd = mixture_moments(
        np.array([9.3, -7.6]), np.array([14.2, 9.05]), np.array([0.4, 0.6])
    )
    assert abs(mean - pooled.mean()) < 0.05
    assert abs(sd - pooled.std()) < 0.05


# -- power ------------------------------------------------------------------


def test_normal_approximation_matches_textbook():
    """For d = 0.5 the normal approximation gives 2(z_a+z_b)^2/d^2 = 62.79.

    The familiar textbook answer of 64 per arm is the noncentral-t figure,
    checked separately below; the roughly two-patient gap between them is the
    small-sample correction the exact solver exists to supply.
    """
    n = n_per_arm_normal(effect=0.5, sd=1.0, alpha=0.05, power=0.80)
    assert 62.5 < n < 63.0


def test_exact_solver_matches_textbook():
    """The standard exact answer for d = 0.5 is 64 per arm."""
    n = math.ceil(n_per_arm_exact(effect=0.5, sd=1.0, alpha=0.05, power=0.80))
    assert n == 64
    # d = 0.8 -> 26 per arm; d = 0.2 -> 394 per arm.
    assert math.ceil(n_per_arm_exact(0.8, 1.0, 0.05, 0.80)) == 26
    assert math.ceil(n_per_arm_exact(0.2, 1.0, 0.05, 0.80)) == 394


def test_exact_solver_is_conservative_relative_to_normal():
    for d in (0.2, 0.35, 0.5, 0.8, 1.2):
        assert n_per_arm_exact(d, 1.0, 0.05, 0.80) >= n_per_arm_normal(d, 1.0, 0.05, 0.80)


def test_solver_and_power_function_are_inverse():
    for d in (0.25, 0.5, 0.9):
        n = n_per_arm_exact(d, 1.0, 0.05, 0.80)
        assert abs(achieved_power(n, d, 1.0, 0.05) - 0.80) < 1e-3


def test_zero_effect_is_infeasible():
    assert n_per_arm_exact(0.0, 1.0) == math.inf
    assert achieved_power(100, 0.0, 1.0) == 0.0


def test_solver_survives_extreme_inputs():
    """The bootstrap reaches these routinely; none may raise."""
    for effect, sd in ((1e-6, 50.0), (500.0, 1e-3), (1e-9, 1e-9), (3.0, 1e6)):
        n = n_per_arm_exact(effect, sd, 0.05, 0.80)
        assert n > 0 and not math.isnan(n)


# -- the enrichment model ---------------------------------------------------


def test_enrichment_removes_between_stratum_variance():
    full = build_population("lvesv", e=1.0)
    assert full.sd_between < 1e-9
    assert math.isclose(full.sd, full.sd_within, rel_tol=1e-9)


def test_full_enrichment_reproduces_the_late_stratum():
    full = build_population("lvesv", e=1.0)
    late = LVESV.change_6mo["late"]
    assert math.isclose(full.effect, -late.mean, rel_tol=1e-9)
    assert math.isclose(full.sd, late.sd, rel_tol=1e-9)


def test_unselected_population_matches_published_pooled_values():
    """At the trial's own composition the model must reproduce the paper."""
    w_early = 6 / 14  # LVESV contributing n at 6 months
    pop = build_population("lvesv", e=1 - w_early)
    assert abs(-pop.effect - LVESV.change_6mo["total"].mean) < 0.05
    assert abs(pop.sd - LVESV.change_6mo["total"].sd) < 0.10


def test_lvesv_pooled_effect_does_not_favour_treatment():
    """The central claim of the paper, as an executable assertion."""
    u = design("lvesv", PI, PI)
    assert not u.favors_treatment
    e = design("lvesv", 1.0, PI)
    assert e.favors_treatment and e.n_total < 200


def test_enrichment_helps_little_when_strata_agree():
    """6-min walk: both strata improve, so enrichment buys little."""
    u = design("six_min_walk", PI, PI)
    e = design("six_min_walk", 1.0, PI)
    assert u.favors_treatment
    assert 1.2 < u.n_total / e.n_total < 2.5


def test_enrichment_hurts_for_ejection_fraction():
    """The pooled EF signal is produced by the early stratum, so enrichment
    removes it rather than sharpening it."""
    u = design("ef", PI, PI)
    e = design("ef", 1.0, PI)
    assert not u.favors_treatment
    assert e.n_total > u.n_total


def test_sample_size_decreases_monotonically_with_enrichment_for_lvesv():
    sweep = enrichment_curve("lvesv", PI, n_points=25)
    feasible = [d.n_total for d in sweep if d.favors_treatment and d.feasible]
    assert len(feasible) > 5
    assert all(a >= b for a, b in zip(feasible, feasible[1:]))


def test_control_drift_shifts_effect_one_for_one():
    a = build_population("lvesv", e=1.0, control_late=0.0)
    b = build_population("lvesv", e=1.0, control_late=-3.0)
    # LVESV is lower-is-better, so a control arm that also falls by 3 mL
    # reduces the benefit-signed effect by exactly 3.
    assert math.isclose(a.effect - b.effect, 3.0, abs_tol=1e-9)


# -- screening and cost -----------------------------------------------------


def test_screening_formula():
    # Fully enriched from a pool that is 25% late: 4 screens per enrollment.
    assert math.isclose(n_screened(100, e=1.0, pi=0.25), 400.0)
    # Unselected: no screening penalty at all.
    assert math.isclose(n_screened(100, e=0.25, pi=0.25), 100.0)


def test_screening_penalty_grows_as_responders_thin():
    prev = 0.0
    for pi in (0.5, 0.25, 0.1, 0.05):
        s = n_screened(100, 1.0, pi)
        assert s > prev
        prev = s


def test_optimizer_rejects_designs_pointing_at_harm():
    """At low prevalence the pooled effect is a large negative, which is cheap
    to detect and must not be selected as the optimum."""
    sweep = enrichment_curve("lvesv", 0.05, n_points=41)
    best, _ = optimal_enrichment(sweep, CostModel())
    assert best.design.favors_treatment
    assert best.design.effect > 0


def test_cost_is_monotone_in_per_patient_cost():
    d = design("lvesv", 1.0, PI)
    cheap = cost(d, CostModel(per_patient=20_000))
    dear = cost(d, CostModel(per_patient=200_000))
    assert dear.total_cost > cheap.total_cost


# -- sensitivity ------------------------------------------------------------


def test_bootstrap_is_reproducible():
    a = bootstrap_designs("lvesv", 1.0, PI, n_draws=300, seed=7)
    b = bootstrap_designs("lvesv", 1.0, PI, n_draws=300, seed=7)
    assert a.n_total_median == b.n_total_median


def test_bootstrap_interval_brackets_the_point_estimate():
    b = bootstrap_designs("lvesv", 1.0, PI, shrinkage=1.0, n_draws=2000, dropout=0.10)
    point = design("lvesv", 1.0, PI, dropout=0.10).n_total
    assert b.n_total_q10 <= point <= b.n_total_q90


def test_bootstrap_is_wider_than_the_point_estimate_suggests():
    """The whole reason the bootstrap exists: 6-8 patients per stratum cannot
    pin a sample size to two significant figures."""
    b = bootstrap_designs("lvesv", 1.0, PI, shrinkage=0.75, n_draws=2000)
    assert b.n_total_q90 / b.n_total_q10 > 5


def test_shrinkage_monotonically_increases_required_n():
    curve = shrinkage_curve("lvesv", PI)
    ns = [p.n_enriched for p in curve]
    assert all(a >= b for a, b in zip(ns, ns[1:]))


# -- the interaction test (the claim the whole project rests on) ------------


def test_lvesv_interaction_is_nominally_significant():
    """The single result the paper is built on, pinned as a regression test."""
    t = interaction_test(ENDPOINTS["lvesv"])
    assert t is not None
    assert abs(t.difference - (-16.9)) < 1e-9
    assert 0.030 < t.p_value < 0.038
    assert t.nominally_significant
    assert t.ci_high < 0  # interval excludes zero, consistent with the p-value


def test_only_one_endpoint_is_nominally_significant():
    sig = [t for t in all_interaction_tests() if t.nominally_significant]
    assert len(sig) == 1 and sig[0].endpoint == "lvesv"


def test_viable_mass_interaction_is_not_significant():
    """It reads dramatically and is not significant; it must not be headlined."""
    t = interaction_test(ENDPOINTS["viable_mass"])
    assert t.p_value > 0.10
    assert not t.nominally_significant


def test_nothing_survives_multiplicity():
    """If this ever starts passing, the paper's framing has to change."""
    mult = multiplicity(all_interaction_tests())
    assert not any(m.bonferroni_pass for m in mult)
    assert not any(m.bh_pass for m in mult)


def test_multiplicity_fails_even_at_the_reduced_denominator():
    nominal, effective, _ = effective_n_tests()
    assert effective < nominal
    assert all_interaction_tests()[0].p_value > 0.05 / effective


def test_benefit_signing_does_not_change_the_evidence():
    for t in all_interaction_tests():
        assert abs(abs(t.difference) - abs(t.difference_benefit)) < 1e-12


def test_strata_are_balanced_at_baseline():
    bal = baseline_balance()
    assert len(bal) >= 8
    assert not any(b.imbalanced for b in bal)


def test_regression_to_mean_is_ruled_out():
    rtm = regression_to_mean_check("lvesv")
    assert rtm is not None
    assert rtm.higher_baseline_stratum == "early"
    assert rtm.contradicts_rtm


def test_evidence_verdict_is_appropriately_hedged():
    ev = assess_evidence()
    assert ev.n_nominally_significant == 1
    assert not ev.survives_bonferroni and not ev.survives_bh
    assert ev.baseline_balanced and ev.rtm_ruled_out


# -- external control anchors ----------------------------------------------


def test_anchors_carry_citations_and_sample_sizes():
    for a in ANCHORS.values():
        assert a.n >= 10
        assert len(a.citation) > 60
        assert a.phase in ("acute", "subacute", "chronic")


def test_indexed_anchors_convert_by_bsa():
    a = ANCHORS["time"]
    assert a.indexed
    assert math.isclose(a.absolute_change(2.0), a.change * 2.0)
    assert math.isclose(a.absolute_change(BSA_CENTRAL), 4.3 * 1.9)


def test_absolute_anchor_is_not_rescaled():
    a = ANCHORS["focus_hf"]
    assert not a.indexed
    assert math.isclose(a.absolute_change(2.0), a.change)


def test_sd_recovered_from_published_confidence_interval():
    """TIME publishes a CI rather than an SD; the recovery must be consistent."""
    a = ANCHORS["time"]
    half = (a.ci[1] - a.ci[0]) / 2.0
    assert math.isclose(a.absolute_sd(1.0), (half / 1.959964) * math.sqrt(a.n), rel_tol=1e-9)


def test_acute_anchors_disagree_in_sign():
    """The central empirical finding of the literature review."""
    vals = [ANCHORS[k].absolute_change() for k in ("time", "empress_mi")]
    assert min(vals) < 0 < max(vals)


def test_chronic_anchor_is_near_zero():
    assert abs(ANCHORS["focus_cctrn"].absolute_change()) < 1e-9


def test_control_priors_bracket_their_sources():
    ep_, lp = early_control_prior(), late_control_prior()
    assert ep_.low < 0 < ep_.high
    assert ep_.low <= ep_.central <= ep_.high
    assert lp.low <= lp.central <= lp.high


def test_bsa_assumption_is_plausible():
    """A wrong BSA would make every conversion in the module wrong."""
    lvesvi = implied_ventrigel_lvesvi()
    assert 65.0 < lvesvi < 100.0


# -- shrinkage composition --------------------------------------------------


def test_shrinkage_applies_to_the_effect_not_the_raw_change():
    """Regression test for a real bug.

    Discounting the raw treatment change rather than the effect lets the
    assumed control arm leak into the discount. With a control change of -10,
    a raw-change discount would give 0.5*(-7.6) + 10 = 6.2, larger than the
    undiscounted 2.4 -- a discount that increases the effect.
    """
    a = Stratum("late", -7.6, 9.0, control_change=-10.0, shrinkage=1.0)
    b = Stratum("late", -7.6, 9.0, control_change=-10.0, shrinkage=0.5)
    assert math.isclose(a.effect, 2.4, abs_tol=1e-9)
    assert math.isclose(b.effect, 1.2, abs_tol=1e-9)
    assert abs(b.effect) < abs(a.effect)


def test_shrinkage_of_one_is_a_no_op():
    a = Stratum("late", -7.6, 9.0, control_change=2.0, shrinkage=1.0)
    assert math.isclose(a.effect, -9.6, abs_tol=1e-12)


# -- assurance --------------------------------------------------------------


def test_vectorized_power_matches_scalar():
    effects = np.array([0.5, 2.0, 7.6, 40.0, -1.0])
    sds = np.full(5, 9.0)
    vec = power_at_vec(50.0, effects, sds, 0.05)
    scalar = np.array([power_at(50.0, float(e), float(s), 0.05) for e, s in zip(effects, sds)])
    assert np.allclose(vec, scalar, atol=1e-9)


def test_power_is_zero_for_wrong_sign_effects():
    assert power_at(100.0, -5.0, 9.0) == 0.0
    assert power_at_vec(100.0, np.array([-5.0]), np.array([9.0]))[0] == 0.0


def test_assurance_curve_is_monotone_in_n():
    """Common random numbers must make this exactly monotone, not approximately."""
    ns = np.array([30, 60, 120, 250, 500, 1000, 2000])
    a = [c.assurance for c in assurance_curve("lvesv", ns, 0.75, 0.0, 0.05, n_draws=4000)]
    assert all(x <= y + 1e-12 for x, y in zip(a, a[1:]))


def test_assurance_is_below_nominal_power():
    """The whole reason the module exists."""
    r = assurance("lvesv", 92, 0.75, 0.0, n_draws=8000)
    assert r.assurance < r.nominal_power
    assert 0.55 < r.assurance < 0.80


def test_assurance_respects_its_ceiling():
    ceiling = assurance_ceiling("lvesv", 0.75, 0.0, n_draws=8000)
    huge = assurance("lvesv", 100000, 0.75, 0.0, n_draws=8000)
    assert huge.assurance <= ceiling + 1e-9
    assert 0.90 < ceiling < 1.0


def test_ceiling_equals_one_minus_wrong_sign_fraction():
    r = assurance("lvesv", 500, 0.75, 0.0, n_draws=8000)
    ceiling = assurance_ceiling("lvesv", 0.75, 0.0, n_draws=8000)
    assert math.isclose(ceiling, 1.0 - r.wrong_sign_fraction, abs_tol=1e-9)


def test_n_for_assurance_increases_with_target():
    ns = [n_for_assurance("lvesv", t, 0.75, 0.0, n_draws=4000) for t in (0.5, 0.6, 0.7, 0.8)]
    assert all(a < b for a, b in zip(ns, ns[1:]))


def test_unreachable_assurance_target_reports_infinity():
    """Targets above the ceiling must say unreachable, not return a huge number."""
    assert n_for_assurance("lvesv", 0.999, 0.75, 0.0, n_draws=4000) == math.inf


def test_assurance_needs_more_patients_than_nominal_power():
    """The headline correction, as an executable assertion."""
    n_power = design("lvesv", 1.0, PI, 0.0, 0.0, dropout=0.10, shrinkage=0.75).n_total
    n_assur = n_for_assurance("lvesv", 0.80, 0.75, 0.0, n_draws=8000)
    assert n_assur > n_power


# -- data integrity ---------------------------------------------------------


def test_every_endpoint_has_a_source_table():
    for ep in ENDPOINTS.values():
        assert ep.source_table.startswith("Online Table")


def test_no_endpoint_has_a_negative_sd():
    for ep in ENDPOINTS.values():
        for table in (ep.baseline, ep.change_3mo, ep.change_6mo):
            for m in table.values():
                assert m.sd >= 0 and m.n >= 2


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  pass  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
