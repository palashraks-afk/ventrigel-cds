"""
External control-arm anchors for post-infarction left ventricular remodeling.

The VentriGel Phase I was single-arm, which leaves the control-arm change
unmeasured and makes it the dominant unknown in any attempt to size a Phase II
from its results. Earlier versions of this analysis handled that by sweeping the
parameter over an arbitrary range. This module replaces the arbitrary range with
published measurements.

Every entry is a control or placebo arm from a randomized trial that measured
left ventricular volumes serially over roughly six months in a post-infarction
or chronic ischemic population. None of these trials studied VentriGel; they are
being used only to estimate what happens to comparable untreated patients over a
comparable interval.

What the anchors show
---------------------

The natural history is **not** a single number, and it does not even have a
single sign. It separates cleanly by time since infarction, which is exactly the
axis the VentriGel subgroups are defined on:

*Acute (days to two weeks post-MI).* Older trials show progressive dilation
(TIME: LVESVI +4.3 mL/m^2; PRESERVATION-I: LVEDVI +11.7 mL/m^2). The most recent
trial, enrolling 2022-2024 on contemporary guideline-directed therapy, shows the
opposite -- LVESVI *falling* 7.8 mL/m^2 with ejection fraction rising 8.5 points
(EMPRESS-MI). Early post-MI natural history is era-dependent, because modern
revascularization and medical therapy produce functional recovery where older
cohorts dilated.

*Chronic (months to years post-MI).* Stable. FOCUS-CCTRN, the closest match to
VentriGel's late stratum in both chronicity and ejection fraction criterion,
found a placebo-arm LVESVI change of exactly zero over six months.

This has a direct consequence for the enrichment argument, and it cuts both
ways. It supports treating the early stratum's observed +9.3 mL as substantially
natural history rather than treatment harm, which shrinks the apparent
enrichment advantage. But it also supports a near-zero control assumption for
the late stratum, which is the stratum an enriched trial would actually enroll,
and therefore protects the enriched design's own effect estimate.

Units
-----

Published anchors are indexed to body surface area (mL/m^2); VentriGel reported
absolute volumes (mL). Conversion needs a BSA, which VentriGel did not report.
:data:`BSA_CENTRAL` is used throughout and swept in the sensitivity analysis.
Its plausibility is checkable: it implies a VentriGel baseline LVESVI of
148.5 / 1.9 = 78.2 mL/m^2, which sits in the expected range for a dilated
post-infarction cohort and slightly above the 65.0 mL/m^2 of the chronic
FOCUS-CCTRN population.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Body surface area assumed when converting indexed volumes to absolute.
#: The VentriGel cohort was 80% male with a mean BMI of 30.0 kg/m^2, for which
#: 1.9 m^2 is a standard central value.
BSA_CENTRAL = 1.9
BSA_RANGE = (1.75, 2.05)


@dataclass(frozen=True)
class ControlAnchor:
    """One published control or placebo arm."""

    key: str
    trial: str
    year: int
    citation: str
    population: str
    #: "acute" (days to ~2 weeks post-MI), "subacute", or "chronic".
    phase: str
    #: Free-text description of enrollment timing relative to the index MI.
    timing: str
    measure: str  # "LVESVI", "LVEDVI", "LVESV"
    indexed: bool
    change: float
    #: Standard deviation of the change where published, else None.
    sd: float | None
    #: 95% confidence interval of the change where published, else None.
    ci: tuple[float, float] | None
    n: int
    #: True when n was derived from a total randomization count rather than
    #: printed for the arm directly.
    n_approximate: bool = False
    interval_months: float = 6.0
    note: str = ""

    def absolute_change(self, bsa: float = BSA_CENTRAL) -> float:
        """Change expressed in mL, converting from indexed units if needed."""
        return self.change * bsa if self.indexed else self.change

    def absolute_sd(self, bsa: float = BSA_CENTRAL) -> float | None:
        if self.sd is not None:
            return self.sd * bsa if self.indexed else self.sd
        if self.ci is not None:
            # Recover SD from a 95% CI of the mean: half-width = 1.96 * SE.
            half = (self.ci[1] - self.ci[0]) / 2.0
            se = half / 1.959964
            sd = se * math.sqrt(self.n)
            return sd * bsa if self.indexed else sd
        return None


# --------------------------------------------------------------------------
# Acute post-MI: days to two weeks
# --------------------------------------------------------------------------

TIME = ControlAnchor(
    key="time",
    trial="TIME",
    year=2012,
    citation=(
        "Traverse JH, Henry TD, Pepine CJ, et al. Effect of the use and timing of "
        "bone marrow mononuclear cell delivery on left ventricular function after "
        "acute myocardial infarction: the TIME randomized trial. JAMA. "
        "2012;308(22):2380-2389. PMID 23129008."
    ),
    population="Anterior STEMI with LV dysfunction after primary PCI",
    phase="acute",
    timing="Cells or placebo at day 3 or day 7 post-PCI",
    measure="LVESVI",
    indexed=True,
    change=4.3,
    sd=None,
    ci=(-0.5, 9.1),
    n=37,
    note=(
        "Placebo arm. LVEDVI rose 10.9 mL/m^2 (95% CI 5.1-16.7) in the same arm. "
        "Same lead author as the VentriGel trial, which makes the imaging "
        "methodology comparable."
    ),
)

PRESERVATION_I = ControlAnchor(
    key="preservation_i",
    trial="PRESERVATION-I",
    year=2016,
    citation=(
        "Rao SV, Zeymer U, Douglas PS, et al. Bioabsorbable intracoronary matrix "
        "for prevention of ventricular remodeling after myocardial infarction. "
        "J Am Coll Cardiol. 2016;68(7):715-723. PMID 27515331."
    ),
    population="Large STEMI despite successful primary PCI",
    phase="acute",
    timing="Saline or bioabsorbable cardiac matrix 2-5 days after primary PCI",
    measure="LVEDVI",
    indexed=True,
    change=11.7,
    sd=26.9,
    ci=None,
    n=102,
    note=(
        "Saline control arm. This is the closest prior analogue to VentriGel: an "
        "injectable biomaterial for post-MI remodeling, which failed on the same "
        "class of endpoint. Reports LVEDVI rather than LVESVI, so it constrains "
        "the diastolic comparator only."
    ),
)

EMPRESS_MI = ControlAnchor(
    key="empress_mi",
    trial="EMPRESS-MI",
    year=2025,
    citation=(
        "Carberry J, Petrie MC, Lee MMY, et al. Empagliflozin to prevent worsening "
        "of left ventricular volumes and systolic function after myocardial "
        "infarction (EMPRESS-MI). Eur J Heart Fail. 2025;27(3):566-576. "
        "PMID 39675781."
    ),
    population="MI with LV systolic dysfunction, LVEF <45% by CMR",
    phase="acute",
    timing="Randomized >=12 h and <=14 days after acute MI",
    measure="LVESVI",
    indexed=True,
    change=-7.8,
    sd=16.3,
    ci=None,
    n=52,
    n_approximate=True,
    note=(
        "Placebo arm; n derived from 105 randomized. Enrolled 2022-2024 on "
        "contemporary guideline-directed therapy. LVESVI FELL and LVEF rose 8.5 "
        "points, the opposite of the older acute trials. This is the single most "
        "important anchor in the table because it shows the sign of early "
        "post-MI natural history is era-dependent."
    ),
)

# --------------------------------------------------------------------------
# Chronic ischemic: the late-stratum analogue
# --------------------------------------------------------------------------

FOCUS_CCTRN = ControlAnchor(
    key="focus_cctrn",
    trial="FOCUS-CCTRN",
    year=2012,
    citation=(
        "Perin EC, Willerson JT, Pepine CJ, et al. Effect of transendocardial "
        "delivery of autologous bone marrow mononuclear cells on functional "
        "capacity, left ventricular function, and perfusion in chronic heart "
        "failure: the FOCUS-CCTRN trial. JAMA. 2012;307(16):1717-1726. "
        "PMID 22447880."
    ),
    population="Chronic ischemic heart disease with LV dysfunction, LVEF <=45%",
    phase="chronic",
    timing="Chronic; not indexed to a recent infarction",
    measure="LVESVI",
    indexed=True,
    change=0.0,
    sd=None,
    ci=None,
    n=28,
    note=(
        "Placebo arm; LVESVI 65.0 (19.8) at baseline and 65.0 (23.3) at 6 months, "
        "p = 0.73. The best available match to VentriGel's late stratum on both "
        "chronicity and the LVEF <=45% criterion, and it uses the same "
        "transendocardial delivery route."
    ),
)

FOCUS_HF = ControlAnchor(
    key="focus_hf",
    trial="FOCUS-HF",
    year=2011,
    citation=(
        "Perin EC, Silva GV, Henry TD, et al. A randomized study of "
        "transendocardial injection of autologous bone marrow mononuclear cells "
        "and cell function analysis in ischemic heart failure (FOCUS-HF). "
        "Am Heart J. 2011;161(6):1078-1087. PMID 21641354."
    ),
    population="Chronic ischemic heart failure",
    phase="chronic",
    timing="Chronic",
    measure="LVESV",
    indexed=False,
    change=-9.9,
    sd=None,
    ci=None,
    n=10,
    note=(
        "Control arm; LVESV 81.7 (40.7) mL at baseline and 71.8 (27.2) mL at 6 "
        "months. Only ten patients, so this is the pessimistic bound on the late "
        "stratum rather than a usable point estimate: if untreated chronic "
        "patients really improve by 10 mL unaided, VentriGel's -7.6 mL is not a "
        "treatment effect at all."
    ),
)


ANCHORS: dict[str, ControlAnchor] = {
    a.key: a for a in (TIME, PRESERVATION_I, EMPRESS_MI, FOCUS_CCTRN, FOCUS_HF)
}


# --------------------------------------------------------------------------
# Deriving control-arm assumptions for the two VentriGel strata
# --------------------------------------------------------------------------


def anchors_for(phase: str, measure: str = "LVESV") -> list[ControlAnchor]:
    """Anchors matching a remodeling phase and volume measure.

    ``measure`` matching ignores the indexing suffix, so "LVESV" collects both
    LVESV and LVESVI entries.
    """
    return [
        a
        for a in ANCHORS.values()
        if a.phase == phase and a.measure.rstrip("I") == measure.rstrip("I")
    ]


@dataclass(frozen=True)
class ControlPrior:
    """A range of defensible control-arm assumptions for one stratum."""

    stratum: str
    phase: str
    central: float
    low: float
    high: float
    bsa: float
    sources: tuple[str, ...]
    rationale: str

    def as_range(self, n: int = 9) -> np.ndarray:
        return np.linspace(self.low, self.high, n)


def early_control_prior(bsa: float = BSA_CENTRAL) -> ControlPrior:
    """Control-arm LVESV change for patients treated within a year of MI.

    The three acute anchors disagree in sign, which is the finding rather than a
    problem with the data. The range is therefore reported honestly as spanning
    both, with the central value taken from TIME because it measures LVESVI
    directly, shares the VentriGel trial's lead author and imaging approach, and
    sits between the other two.
    """
    vals = [a.absolute_change(bsa) for a in anchors_for("acute", "LVESV")]
    return ControlPrior(
        stratum="early",
        phase="acute",
        central=TIME.absolute_change(bsa),
        low=min(vals),
        high=max(vals),
        bsa=bsa,
        sources=("TIME", "EMPRESS-MI", "PRESERVATION-I"),
        rationale=(
            "Acute post-MI natural history is era-dependent. Older trials show "
            "dilation (TIME +4.3 mL/m^2), the most recent shows reverse "
            "remodeling on contemporary therapy (EMPRESS-MI -7.8 mL/m^2). The "
            "range spans both signs deliberately."
        ),
    )


def late_control_prior(bsa: float = BSA_CENTRAL) -> ControlPrior:
    """Control-arm LVESV change for patients treated more than a year post-MI.

    FOCUS-CCTRN is the central estimate: chronic, LVEF <=45%, transendocardial
    delivery, n = 28, and an observed change of exactly zero. FOCUS-HF supplies
    the pessimistic bound.
    """
    return ControlPrior(
        stratum="late",
        phase="chronic",
        central=FOCUS_CCTRN.absolute_change(bsa),
        low=FOCUS_HF.absolute_change(bsa),
        high=abs(FOCUS_HF.absolute_change(bsa)),
        bsa=bsa,
        sources=("FOCUS-CCTRN", "FOCUS-HF"),
        rationale=(
            "Chronic ischemic populations are stable over six months. "
            "FOCUS-CCTRN, the closest match on chronicity, ejection fraction "
            "criterion and delivery route, observed no change. FOCUS-HF (n = 10) "
            "sets the pessimistic bound."
        ),
    )


def implied_ventrigel_lvesvi(baseline_mL: float = 148.5, bsa: float = BSA_CENTRAL) -> float:
    """Sanity check on the BSA assumption.

    Converting VentriGel's absolute baseline into indexed units should land in
    the range the anchor trials report for comparable populations. If it does
    not, the BSA is wrong and every conversion in this module is suspect.
    """
    return baseline_mL / bsa
