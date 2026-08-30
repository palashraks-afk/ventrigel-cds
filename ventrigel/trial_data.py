"""
Published results of the VentriGel first-in-man trial (NCT02305602).

Every value in this module is transcribed directly from:

    Traverse JH, Henry TD, Dib N, Patel AN, Pepine C, Schaer GL, DeQuach JA,
    Kinsey AM, Chamberlin P, Christman KL. First-in-Man Study of a Cardiac
    Extracellular Matrix Hydrogel in Early and Late Myocardial Infarction
    Patients. JACC Basic Transl Sci. 2019;4(6):659-669.
    doi:10.1016/j.jacbts.2019.07.012

Values come from the Online Tables of the Supplemental Appendix, which report
each endpoint as ``mean (SEM)`` with the contributing n, separately for the
total cohort and for the two prespecified temporal subgroups. Each entry below
carries the exact table it came from so a reviewer can check it against the
source in a few seconds.

No value in this project is simulated, imputed, or digitized from a figure.

Two transcription notes, recorded rather than silently corrected:

1.  The main text reports the 3-month 6-min walk change as "+35.6 m
    (p = 0.033)" while Online Table 5 reports "35.5 (14.9), p = 0.03". The
    difference is rounding. This module uses the Online Table value, because
    the table also supplies the SEM and n needed for the variance recovery.

2.  Online Table 7 renders the 3-month LVESV change for the late subgroup as
    "-110.7 (5.3)". A change of -110.7 mL is not physically compatible with a
    142.3 mL baseline, a 5.3 mL SEM, or the reported p = 0.09; the value is
    an artifact of the "-1" list marker colliding with "-10.7" during
    typesetting. Recomputing the p-value confirms it: -10.7/5.3 gives
    p = 0.09 exactly, -110.7/5.3 gives p < 1e-9. It is recorded here as
    -10.7 and flagged in ``TRANSCRIPTION_NOTES``. The 3-month LVESV values
    are not used in any headline result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CITATION = (
    "Traverse JH, Henry TD, Dib N, et al. First-in-Man Study of a Cardiac "
    "Extracellular Matrix Hydrogel in Early and Late Myocardial Infarction "
    "Patients. JACC Basic Transl Sci. 2019;4(6):659-669."
)
TRIAL_ID = "NCT02305602"

# --------------------------------------------------------------------------
# Cohort structure (Table 1 of the main paper)
# --------------------------------------------------------------------------

N_SCREENED = 22
N_ENROLLED = 15
N_EARLY = 7  # < 12 months post-MI at treatment
N_LATE = 8  # > 12 months post-MI at treatment

#: Months from index MI to injection, mean (SD) -- Table 1 reports SD here,
#: not SEM, unlike the Online Tables.
TIME_TO_INJECTION_MONTHS = {
    "total": (15.2, 10.6),
    "early": (6.5, 2.9),
    "late": (22.8, 8.7),
}
TIME_TO_INJECTION_RANGE_MONTHS = (3.0, 35.5)  # "delivered between 3 and 35.5 months"

AGE_YEARS = {"total": (59.6, 8.8), "early": (57.7, 10.3), "late": (61.3, 7.5)}

#: Protocol eligibility window, from the Methods and the Supplemental Appendix.
EF_WINDOW_PERCENT = (25.0, 45.0)
POST_MI_WINDOW_DAYS = (60, 1095)  # 60 days to 3 years


# --------------------------------------------------------------------------
# Endpoint measurements
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Measurement:
    """One published cell: a mean with its standard error and sample size.

    ``p_published`` is the paired t-test p-value printed alongside it, kept so
    that :mod:`ventrigel.recovery` can recompute it and confirm the
    transcription is faithful.
    """

    mean: float
    sem: float
    n: int
    p_published: float | None = None

    @property
    def sd(self) -> float:
        """Sample standard deviation, recovered as ``SEM * sqrt(n)``."""
        return self.sem * (self.n**0.5)


@dataclass(frozen=True)
class Endpoint:
    """A single trial endpoint with its subgroup breakdown."""

    key: str
    label: str
    unit: str
    source_table: str
    #: True when a *decrease* in the measure is the clinically good direction.
    lower_is_better: bool
    #: Change from baseline at 6 months, by subgroup.
    change_6mo: dict[str, Measurement]
    #: Baseline value, by subgroup.
    baseline: dict[str, Measurement] = field(default_factory=dict)
    #: Change from baseline at 3 months, by subgroup, where reported.
    change_3mo: dict[str, Measurement] = field(default_factory=dict)
    notes: str = ""

    def benefit(self, group: str) -> float:
        """Signed change re-expressed so that positive always means benefit."""
        m = self.change_6mo[group].mean
        return -m if self.lower_is_better else m


# -- Online Table 5: functional exercise capacity ---------------------------

SIX_MIN_WALK = Endpoint(
    key="six_min_walk",
    label="6-minute walk distance",
    unit="m",
    source_table="Online Table 5",
    lower_is_better=False,
    baseline={
        "total": Measurement(429.4, 24.8, 15),
        "early": Measurement(442.4, 28.9, 7),
        "late": Measurement(418.0, 40.5, 8),
    },
    change_3mo={
        "total": Measurement(35.5, 14.9, 14, 0.03),
        "early": Measurement(30.4, 13.5, 7, 0.07),
        "late": Measurement(40.6, 27.8, 7, 0.19),
    },
    change_6mo={
        "total": Measurement(44.4, 13.9, 14, 0.007),
        "early": Measurement(40.5, 22.5, 7, 0.12),
        "late": Measurement(50.9, 17.7, 7, 0.04),
    },
    notes=(
        "The two subgroups contribute n=7 each at 6 months, so the total-cohort "
        "mean should equal their simple average of 45.7 m; the table prints "
        "44.4 m. The 1.3 m gap is a rounding artifact in the source and is "
        "reported, not corrected."
    ),
)

# -- Online Table 7: cardiac magnetic resonance -----------------------------

LVESV = Endpoint(
    key="lvesv",
    label="LV end-systolic volume",
    unit="mL",
    source_table="Online Table 7",
    lower_is_better=True,
    baseline={
        "total": Measurement(148.5, 16.7, 14),
        "early": Measurement(156.8, 25.2, 6),
        "late": Measurement(142.3, 23.5, 8),
    },
    change_3mo={
        "total": Measurement(-2.1, 4.6, 13, 0.66),
        "early": Measurement(8.0, 5.7, 6, 0.22),
        # See module docstring, note 2: printed as -110.7 in the source.
        "late": Measurement(-10.7, 5.3, 7, 0.09),
    },
    change_6mo={
        "total": Measurement(-0.35, 3.8, 14, 0.93),
        "early": Measurement(9.3, 5.8, 6, 0.17),
        "late": Measurement(-7.6, 3.2, 8, 0.05),
    },
)

LVEDV = Endpoint(
    key="lvedv",
    label="LV end-diastolic volume",
    unit="mL",
    source_table="Online Table 7",
    lower_is_better=True,
    baseline={
        "total": Measurement(229.6, 18.6, 14),
        "early": Measurement(245.3, 26.5, 6),
        "late": Measurement(217.8, 26.5, 8),
    },
    change_3mo={
        "total": Measurement(-4.8, 6.4, 13, 0.47),
        "early": Measurement(0.0, 10.1, 6, 0.99),
        "late": Measurement(-8.9, 8.5, 7, 0.34),
    },
    change_6mo={
        "total": Measurement(-4.9, 5.4, 14, 0.38),
        "early": Measurement(-1.2, 9.5, 6, 0.89),
        "late": Measurement(-7.6, 6.5, 8, 0.28),
    },
)

EJECTION_FRACTION = Endpoint(
    key="ef",
    label="LV ejection fraction",
    unit="%",
    source_table="Online Table 7",
    lower_is_better=False,
    baseline={
        "total": Measurement(37.1, 2.3, 14),
        "early": Measurement(37.7, 3.6, 6),
        "late": Measurement(36.6, 3.3, 8),
    },
    change_3mo={
        "total": Measurement(-0.33, 1.0, 13, 0.77),
        "early": Measurement(-3.2, 1.1, 6, 0.03),
        "late": Measurement(2.1, 1.0, 7, 0.08),
    },
    change_6mo={
        "total": Measurement(-1.3, 1.2, 14, 0.29),
        "early": Measurement(-3.8, 1.3, 6, 0.03),
        "late": Measurement(-0.6, 1.5, 8, 0.69),
    },
)

VIABLE_MASS = Endpoint(
    key="viable_mass",
    label="Viable myocardial mass",
    unit="g",
    source_table="Online Table 7",
    lower_is_better=False,
    baseline={
        "total": Measurement(114.2, 6.8, 12),
        "early": Measurement(121.6, 14.8, 5),
        "late": Measurement(108.9, 5.6, 7),
    },
    change_3mo={
        "total": Measurement(7.3, 6.8, 10, 0.31),
        "early": Measurement(-2.7, 12.7, 3, 0.85),
        "late": Measurement(11.6, 8.0, 7, 0.20),
    },
    change_6mo={
        "total": Measurement(3.4, 8.1, 12, 0.68),
        "early": Measurement(-10.5, 14.9, 5, 0.52),
        "late": Measurement(15.0, 5.9, 7, 0.05),
    },
)

SCAR_FRACTION = Endpoint(
    key="scar",
    label="Scar tissue",
    unit="% LV",
    source_table="Online Table 7",
    lower_is_better=True,
    baseline={
        "total": Measurement(27.1, 1.8, 12),
        "early": Measurement(25.9, 1.6, 5),
        "late": Measurement(27.9, 2.9, 7),
    },
    change_6mo={
        "total": Measurement(0.50, 1.9, 12, 0.58),
        "early": Measurement(2.3, 3.5, 5, 0.55),
        "late": Measurement(-1.0, 1.9, 7, 0.63),
    },
)

# -- Online Table 6: symptoms and quality of life ---------------------------

MLWHFQ = Endpoint(
    key="mlwhfq",
    label="Minnesota Living with Heart Failure score",
    unit="points",
    source_table="Online Table 6",
    lower_is_better=True,
    baseline={
        "total": Measurement(38.9, 7.4, 15),
        "early": Measurement(44.3, 10.3, 7),
        "late": Measurement(34.3, 11.0, 8),
    },
    change_6mo={
        "total": Measurement(-7.2, 7.8, 14, 0.37),
        "early": Measurement(0.7, 10.9, 7, 0.95),
        "late": Measurement(-15.1, 11.2, 7, 0.22),
    },
)

NYHA = Endpoint(
    key="nyha",
    label="NYHA functional class",
    unit="class",
    source_table="Online Table 6",
    lower_is_better=True,
    baseline={
        "total": Measurement(1.9, 0.1, 15),
        "early": Measurement(2.0, 0.2, 7),
        "late": Measurement(1.8, 0.2, 8),
    },
    change_6mo={
        "total": Measurement(-0.3, 0.2, 14, 0.17),
        "early": Measurement(-0.3, 0.3, 7, 0.36),
        "late": Measurement(-0.3, 0.3, 7, 0.36),
    },
    notes="The only endpoint with no separation between subgroups.",
)

# -- Online Table 8: BNP (percent change) -----------------------------------

BNP = Endpoint(
    key="bnp",
    label="B-type natriuretic peptide",
    unit="% change",
    source_table="Online Table 8",
    lower_is_better=True,
    change_6mo={
        "total": Measurement(-12.0, 9.1, 14, 0.63),
        "early": Measurement(-6.7, 13.6, 7, 0.43),
        "late": Measurement(-18.3, 12.4, 7, 0.41),
    },
    notes=(
        "Reported as percent change, so the recovered SD is on a percent scale "
        "and is not comparable to the absolute-change endpoints."
    ),
)


ENDPOINTS: dict[str, Endpoint] = {
    e.key: e
    for e in (
        SIX_MIN_WALK,
        LVESV,
        LVEDV,
        EJECTION_FRACTION,
        VIABLE_MASS,
        SCAR_FRACTION,
        MLWHFQ,
        NYHA,
        BNP,
    )
}

#: Endpoints plausible as a Phase II primary endpoint. BNP is excluded because
#: it is on a percent scale; NYHA and scar because the trial showed no
#: subgroup separation worth enriching for.
CANDIDATE_PRIMARY_ENDPOINTS = ("lvesv", "six_min_walk", "viable_mass", "ef", "lvedv", "mlwhfq")

TRANSCRIPTION_NOTES = [
    (
        "lvesv",
        "change_3mo",
        "late",
        "Source prints -110.7; recorded as -10.7. The published p = 0.09 is "
        "reproduced by -10.7/5.3 (p = 0.087) and contradicted by -110.7/5.3 "
        "(p < 1e-9). Not used in any headline result.",
    ),
    (
        "six_min_walk",
        "change_3mo",
        "total",
        "Main text says +35.6 m, p = 0.033; Online Table 5 says 35.5, p = 0.03. "
        "The table value is used because it carries the SEM and n.",
    ),
]
