"""
Cost model for a two-arm Phase II cardiovascular trial.

Every figure a cost model produces is only as good as its unit costs, so this
module keeps them in one visible place, gives each one a source or an explicit
label as an assumption, and makes the optimum depend on them rather than
hiding them inside a headline dollar figure.

The structure is deliberately simple, because a more elaborate model would
imply a precision the inputs do not support:

    total = fixed
          + per_patient      * n_enrolled
          + per_screen       * n_screened
          + per_site_month   * duration_months * n_sites

Screening is separated from enrolment because that separation is the entire
economic tension in an enrichment design. Tightening eligibility shrinks
``n_enrolled`` but inflates ``n_screened``, and past some point the second term
overwhelms the first. Any analysis that models only enrolment will conclude
that maximal enrichment is always optimal, which is false.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .power import Design


@dataclass(frozen=True)
class CostModel:
    """Unit costs for a Phase II cardiovascular trial, all in USD.

    The defaults are order-of-magnitude planning figures, not measurements.
    They are chosen to be defensible and are stated in the manuscript as
    assumptions; the conclusions are reported as sample-size ratios first and
    dollars second precisely because the ratios do not depend on them.
    """

    #: Protocol development, regulatory, biostatistics, DSMB, close-out.
    fixed: float = 2_000_000.0
    #: Per randomized patient: drug or device, catheterization, imaging at
    #: three timepoints, site payments, monitoring, adjudication.
    per_patient: float = 65_000.0
    #: Per patient screened but not randomized: consent, screening echo,
    #: laboratory panel, coordinator time.
    per_screen: float = 2_500.0
    #: Site activation and maintenance, per site per month.
    per_site_month: float = 12_000.0
    #: Patients one site can enroll per month.
    enrollment_rate_per_site_month: float = 0.75
    #: Months of follow-up after the last patient is randomized, plus
    #: database lock. The VentriGel primary window was 6 months.
    followup_months: float = 9.0
    n_sites: int = 6  # the Phase I ran at 6 sites

    def label(self) -> str:
        return (
            f"fixed ${self.fixed:,.0f}; ${self.per_patient:,.0f}/patient; "
            f"${self.per_screen:,.0f}/screen; ${self.per_site_month:,.0f}/site-month; "
            f"{self.n_sites} sites at {self.enrollment_rate_per_site_month}/site/month"
        )


@dataclass(frozen=True)
class CostedDesign:
    design: Design
    enrollment_months: float
    duration_months: float
    patient_cost: float
    screening_cost: float
    site_cost: float
    fixed_cost: float
    total_cost: float

    @property
    def feasible(self) -> bool:
        return math.isfinite(self.total_cost)


def cost(d: Design, model: CostModel | None = None) -> CostedDesign:
    """Cost one trial design."""
    m = model or CostModel()
    if not d.feasible:
        return CostedDesign(d, math.inf, math.inf, math.inf, math.inf, math.inf, m.fixed, math.inf)

    # Enrollment duration is driven by how many patients must be *screened*,
    # since screening capacity, not randomization, is the rate limit in an
    # enriched trial.
    site_capacity = m.n_sites * m.enrollment_rate_per_site_month
    enroll_months = d.n_screened / site_capacity if site_capacity > 0 else math.inf
    duration = enroll_months + m.followup_months

    patient = m.per_patient * d.n_total
    screening = m.per_screen * max(0.0, d.n_screened - d.n_total)
    site = m.per_site_month * duration * m.n_sites
    total = m.fixed + patient + screening + site
    return CostedDesign(d, enroll_months, duration, patient, screening, site, m.fixed, total)


def optimal_enrichment(
    designs: list[Design], model: CostModel | None = None
) -> tuple[CostedDesign, list[CostedDesign]]:
    """Cost a whole enrichment sweep and return the cheapest feasible design.

    The optimum is generally interior: cost falls steeply as enrichment removes
    diluting patients, then rises again as the screening multiplier takes over.
    """
    costed = [cost(d, model) for d in designs]
    # A design only counts as a candidate if its modelled effect points toward
    # benefit. Without this filter the optimizer happily selects a cheap,
    # well-powered trial whose effect points at harm -- at low responder
    # prevalence the pooled effect is dominated by the non-responding stratum
    # deteriorating, which is a large effect in the wrong direction and
    # therefore a small n. That is a trial designed to demonstrate the therapy
    # does not work.
    candidates = [c for c in costed if c.feasible and c.design.favors_treatment]
    if not candidates:
        raise ValueError("no design in the sweep both feasible and favouring treatment")
    best = min(candidates, key=lambda c: c.total_cost)
    return best, costed


def savings_vs_unselected(
    costed: list[CostedDesign],
) -> tuple[CostedDesign, CostedDesign, float, float]:
    """Compare the cheapest design against the unselected one.

    Returns ``(unselected, best, absolute_saving, fractional_saving)``. The
    unselected design is the first in the sweep, where ``e == pi``.
    """
    unselected = costed[0]
    candidates = [c for c in costed if c.feasible and c.design.favors_treatment]
    if not candidates:
        raise ValueError("no design in the sweep both feasible and favouring treatment")
    best = min(candidates, key=lambda c: c.total_cost)
    if not (unselected.feasible and unselected.design.favors_treatment):
        return unselected, best, math.inf, 1.0
    saving = unselected.total_cost - best.total_cost
    return unselected, best, saving, saving / unselected.total_cost


def with_scaled_costs(model: CostModel, factor: float) -> CostModel:
    """Scale every variable cost, for testing how much the optimum moves."""
    return replace(
        model,
        per_patient=model.per_patient * factor,
        per_screen=model.per_screen * factor,
        per_site_month=model.per_site_month * factor,
    )
