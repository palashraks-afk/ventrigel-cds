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
:data:`BSA_CENTRAL` is used throughout and swept over :data:`BSA_RANGE` by
:func:`bsa_sensitivity`. Its plausibility is checkable: it implies a VentriGel
baseline LVESVI of 148.5 / 1.9 = 78.2 mL/m^2, which sits in the expected range
for a dilated post-infarction cohort and slightly above the 65.0 mL/m^2 of the
chronic FOCUS-CCTRN population.

Anchors carry their own uncertainty
-----------------------------------

An anchor is an estimate, not a constant. FOCUS-CCTRN's zero comes from 28
patients, and the standard error on it is comparable in size to the treatment
effect it is being subtracted from. Treating anchors as exact understates the
uncertainty in every downstream sample size, sometimes by a factor of two, so
:meth:`ControlAnchor.standard_error` exposes it and the assurance and bootstrap
machinery propagates it.

Recovering that standard error is not always direct. Trials that report the SD
of the *change* give it immediately. Trials that report only the SDs of the
baseline and follow-up *levels* do not, because the SD of a paired change
depends on the test-retest correlation:

    sd_change = sqrt(sd_1^2 + sd_2^2 - 2 * r * sd_1 * sd_2)

and r is essentially never published. For serial cardiac magnetic resonance in
a stable population r is high, typically 0.85 to 0.95; :data:`DEFAULT_RETEST_R`
takes the conservative end of that range and :func:`retest_sensitivity` shows
what the choice costs.
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

#: Test-retest correlation assumed when recovering a change SD from published
#: baseline and follow-up level SDs. Serial CMR in a stable population is
#: highly reproducible; 0.85 is the conservative end of the usual 0.85-0.95
#: range, so it produces the *larger* change SD and the wider uncertainty.
DEFAULT_RETEST_R = 0.85


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
    #: Which VentriGel endpoint this anchor constrains.
    endpoint: str
    measure: str
    indexed: bool
    change: float
    #: Standard deviation of the change where published, else None.
    sd: float | None
    #: 95% confidence interval of the change where published, else None.
    ci: tuple[float, float] | None
    n: int
    #: SDs of the baseline and follow-up *levels*, where the change SD was not
    #: published. Used with the test-retest correlation to recover it.
    sd_levels: tuple[float, float] | None = None
    #: True when ``change`` and ``sd`` are percentages of baseline rather than
    #: absolute units. Converted using ``percent_reference``.
    percent: bool = False
    #: Baseline value the percentage refers to, in the endpoint's own units.
    #: For VentriGel comparisons this is the relevant stratum's baseline.
    percent_reference: float | None = None
    #: True when n was derived from a total randomization count rather than
    #: printed for the arm directly.
    n_approximate: bool = False
    #: "trial" for a peer-reviewed primary report, "review" for a systematic
    #: review, "abstract" for a conference abstract. Abstracts have not been
    #: through full peer review and are labelled as such wherever used.
    evidence: str = "trial"
    interval_months: float = 6.0
    note: str = ""

    def absolute_change(self, bsa: float = BSA_CENTRAL) -> float:
        """Change expressed in native absolute units.

        Handles the two ways a source can decline to give an absolute number:
        indexing to body surface area, and reporting a percentage of baseline.
        """
        if self.percent:
            if self.percent_reference is None:
                raise ValueError(f"{self.key}: percent anchor needs percent_reference")
            return self.change / 100.0 * self.percent_reference
        return self.change * bsa if self.indexed else self.change

    def absolute_sd(
        self, bsa: float = BSA_CENTRAL, retest_r: float = DEFAULT_RETEST_R
    ) -> float | None:
        """SD of the change, from whichever route the source supports."""
        sd = self.sd
        if sd is None and self.ci is not None:
            # Recover SD from a 95% CI of the mean: half-width = 1.96 * SE.
            half = (self.ci[1] - self.ci[0]) / 2.0
            sd = (half / 1.959964) * math.sqrt(self.n)
        if sd is None and self.sd_levels is not None:
            s1, s2 = self.sd_levels
            var = s1**2 + s2**2 - 2.0 * retest_r * s1 * s2
            sd = math.sqrt(max(var, 0.0))
        if sd is None:
            return None
        if self.percent:
            if self.percent_reference is None:
                raise ValueError(f"{self.key}: percent anchor needs percent_reference")
            return sd / 100.0 * self.percent_reference
        return sd * bsa if self.indexed else sd

    def standard_error(
        self, bsa: float = BSA_CENTRAL, retest_r: float = DEFAULT_RETEST_R
    ) -> float | None:
        """Standard error of this anchor's mean change.

        This is the quantity that has to be propagated. An anchor of "zero"
        from 28 patients is not zero; it is zero plus or minus several mL, and
        several mL is the same order as the effect being estimated against it.
        """
        sd = self.absolute_sd(bsa, retest_r)
        return None if sd is None else sd / math.sqrt(self.n)


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
    endpoint="lvesv",
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
    endpoint="lvedv",
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
    endpoint="lvesv",
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
    endpoint="lvesv",
    measure="LVESVI",
    indexed=True,
    change=0.0,
    sd=None,
    ci=None,
    n=28,
    sd_levels=(19.8, 23.3),
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
    endpoint="lvesv",
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


# --------------------------------------------------------------------------
# Ejection fraction
# --------------------------------------------------------------------------

EMPRESS_MI_EF = ControlAnchor(
    key="empress_mi_ef",
    trial="EMPRESS-MI",
    year=2025,
    citation=EMPRESS_MI.citation,
    population="MI with LV systolic dysfunction, LVEF <45% by CMR",
    phase="acute",
    timing="Randomized >=12 h and <=14 days after acute MI",
    endpoint="ef",
    measure="LVEF",
    indexed=False,
    change=8.5,
    sd=7.4,
    ci=None,
    n=52,
    n_approximate=True,
    note=(
        "Placebo arm. Ejection fraction rose 8.5 points over 24 weeks with no "
        "treatment, which is the scale of spontaneous functional recovery in "
        "acute post-MI patients on contemporary care. Against this comparator "
        "VentriGel's early-stratum -3.8% is a large apparent deficit, and the "
        "pooled ejection-fraction signal the trial reported cannot be read as a "
        "treatment effect without it."
    ),
)

FOCUS_CCTRN_EF = ControlAnchor(
    key="focus_cctrn_ef",
    trial="FOCUS-CCTRN",
    year=2012,
    citation=FOCUS_CCTRN.citation,
    population="Chronic ischemic heart disease with LV dysfunction, LVEF <=45%",
    phase="chronic",
    timing="Chronic; not indexed to a recent infarction",
    endpoint="ef",
    measure="LVEF",
    indexed=False,
    change=-1.3,
    sd=5.1,
    ci=None,
    n=28,
    note=(
        "Placebo arm; LVEF fell 1.3 points (SD 5.1) over 6 months from a 32.3% "
        "baseline. This matters more than it looks: VentriGel's late stratum "
        "fell 0.6 points, which against a comparator falling 1.3 is a small "
        "*benefit* rather than the null the trial reported."
    ),
)

# --------------------------------------------------------------------------
# Functional capacity
# --------------------------------------------------------------------------

KHAN_6MWT = ControlAnchor(
    key="khan_6mwt",
    trial="Khan et al. systematic review",
    year=2022,
    citation=(
        "Khan MA, et al. Placebo effects on 6-minute walk test and peak oxygen "
        "consumption in patients with heart failure: a systematic review of "
        "contemporary randomized controlled trials. J Card Fail. "
        "2022;28(5):S90. doi:10.1016/j.cardfail.2022.03.226."
    ),
    population="Heart failure, 38 double-blind placebo-controlled RCTs, 2,713 patients",
    phase="chronic",
    timing="Trials conducted 2015-2020; median duration 24.5 weeks",
    endpoint="six_min_walk",
    measure="6MWT",
    indexed=False,
    change=4.2,
    sd=12.9,
    ci=None,
    n=2713,
    percent=True,
    percent_reference=418.0,  # VentriGel late-stratum baseline, Online Table 5
    evidence="abstract",
    interval_months=5.6,
    note=(
        "PLACEBO arms only, pooled across 38 trials: +4.2% (SD 12.9) at a median "
        "24.5 weeks, with 28 of 38 trials increasing and 10 decreasing. Applied "
        "to VentriGel's 418.0 m late-stratum baseline this is about +17.6 m of "
        "expected placebo improvement, which is a third of the 50.9 m the late "
        "stratum gained. NOTE: this is a conference abstract, not a "
        "peer-reviewed full report, and carries correspondingly less weight; it "
        "is used because it is the only pooled estimate of placebo-arm walk "
        "distance located, and because leaving the endpoint unanchored would "
        "flatter it relative to the volume endpoints."
    ),
)


ANCHORS: dict[str, ControlAnchor] = {
    a.key: a
    for a in (
        TIME,
        PRESERVATION_I,
        EMPRESS_MI,
        FOCUS_CCTRN,
        FOCUS_HF,
        EMPRESS_MI_EF,
        FOCUS_CCTRN_EF,
        KHAN_6MWT,
    )
}


# --------------------------------------------------------------------------
# Deriving control-arm assumptions for the two VentriGel strata
# --------------------------------------------------------------------------


def anchors_for(phase: str, endpoint: str = "lvesv") -> list[ControlAnchor]:
    """Anchors constraining one VentriGel endpoint in one remodeling phase."""
    return [a for a in ANCHORS.values() if a.phase == phase and a.endpoint == endpoint]


#: Which VentriGel stratum each remodeling phase stands in for. The early
#: stratum is 60 days to 12 months post-MI and the acute anchors are days to
#: two weeks, so the match is imperfect and generous to the anchors; this is
#: stated in the manuscript rather than papered over.
PHASE_FOR_STRATUM = {"early": "acute", "late": "chronic"}


def anchored_control(
    endpoint: str,
    stratum: str,
    bsa: float = BSA_CENTRAL,
    retest_r: float = DEFAULT_RETEST_R,
) -> tuple[float, float] | None:
    """Best available ``(change, standard_error)`` for one endpoint and stratum.

    Returns ``None`` when no anchor covers that combination, which callers must
    handle explicitly rather than silently substituting zero. Assuming no
    control drift is itself a strong assumption, and it flatters the treatment
    whenever the untreated course is favourable -- which the ejection-fraction
    anchors show it can be, dramatically.

    Where several anchors cover the same cell the choice is made by
    :data:`PREFERRED_ANCHOR`, explicitly and with a stated reason, not by an
    automatic rule. An earlier version selected by sample size, which silently
    chose EMPRESS-MI over TIME for the early stratum and thereby flipped the
    sign of the comparator from +8.2 to -14.8 mL. A rule that can reverse a
    conclusion without anyone deciding to is worse than no rule.
    """
    phase = PHASE_FOR_STRATUM.get(stratum)
    if phase is None:
        return None
    candidates = anchors_for(phase, endpoint)
    if not candidates:
        return None

    preferred = PREFERRED_ANCHOR.get((endpoint, stratum))
    best = next((a for a in candidates if a.key == preferred), None)
    if best is None:
        # No documented preference: only safe when the cell is unambiguous.
        if len(candidates) > 1:
            raise KeyError(
                f"{endpoint}/{stratum} has {len(candidates)} candidate anchors and no "
                f"entry in PREFERRED_ANCHOR. Choose one explicitly."
            )
        best = candidates[0]
    se = best.standard_error(bsa, retest_r)
    return best.absolute_change(bsa), (0.0 if se is None else se)


#: Explicit anchor choice for each contested endpoint-stratum cell, with the
#: reason recorded in :data:`ANCHOR_CHOICE_RATIONALE`.
PREFERRED_ANCHOR: dict[tuple[str, str], str] = {
    ("lvesv", "early"): "time",
    ("lvesv", "late"): "focus_cctrn",
}

ANCHOR_CHOICE_RATIONALE: dict[tuple[str, str], str] = {
    ("lvesv", "early"): (
        "TIME over EMPRESS-MI despite the smaller n. The two disagree in sign "
        "(+4.3 vs -7.8 mL/m2) and the disagreement is real, not noise: "
        "EMPRESS-MI enrolled within 14 days on 2022-2024 therapy and captured "
        "spontaneous recovery, while VentriGel's early stratum was 60 days to "
        "12 months post-MI, past the window in which most of that recovery "
        "occurs. TIME also shares the VentriGel trial's lead author and imaging "
        "core laboratory. EMPRESS-MI is carried as the opposing bound in every "
        "sensitivity analysis rather than discarded."
    ),
    ("lvesv", "late"): (
        "FOCUS-CCTRN over FOCUS-HF on sample size (28 vs 10) and on match: "
        "chronic, LVEF <=45%, transendocardial delivery. FOCUS-HF is carried as "
        "the pessimistic bound, and it is the bound that would nullify the "
        "project, so it is reported alongside every headline number."
    ),
}


def anchor_coverage() -> dict[str, dict[str, str | None]]:
    """Which endpoint-stratum cells have an anchor, for honest reporting."""
    out: dict[str, dict[str, str | None]] = {}
    endpoints = sorted({a.endpoint for a in ANCHORS.values()})
    for ep in endpoints:
        out[ep] = {}
        for stratum, phase in PHASE_FOR_STRATUM.items():
            hits = anchors_for(phase, ep)
            out[ep][stratum] = hits[0].trial if hits else None
    return out


def bsa_sensitivity(
    endpoint: str = "lvesv", stratum: str = "late", n_points: int = 7
) -> list[tuple[float, float, float]]:
    """How much the body-surface-area assumption moves an anchored control.

    Returns ``(bsa, change, standard_error)`` across :data:`BSA_RANGE`. Only
    indexed anchors move at all, and for the late-stratum LVESV anchor the
    change is exactly zero at every BSA, so the assumption turns out not to
    matter where it matters most. That is worth demonstrating rather than
    asserting, which is why this function exists.
    """
    out = []
    for bsa in np.linspace(BSA_RANGE[0], BSA_RANGE[1], n_points):
        got = anchored_control(endpoint, stratum, float(bsa))
        if got is None:
            continue
        out.append((float(bsa), got[0], got[1]))
    return out


def retest_sensitivity(
    endpoint: str = "lvesv", stratum: str = "late"
) -> list[tuple[float, float]]:
    """How much the unpublished test-retest correlation moves an anchor's SE.

    Returns ``(r, standard_error)``. The correlation is never published, so the
    resulting standard error is itself uncertain; this makes the size of that
    second-order uncertainty visible instead of hiding it behind a default.
    """
    out = []
    for r in (0.95, 0.90, 0.85, 0.80, 0.70):
        got = anchored_control(endpoint, stratum, BSA_CENTRAL, r)
        if got is not None:
            out.append((r, got[1]))
    return out


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
