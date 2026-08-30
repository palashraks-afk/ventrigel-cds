"""
Enrichment analysis for a Phase II trial of a cardiac extracellular matrix
hydrogel, built entirely from the published results of the VentriGel
first-in-man trial (NCT02305602).

The question: given what Phase I actually showed, how large must Phase II be,
and how much smaller does it get if enrollment is restricted to the subgroup
in which the effect was seen?

Modules
-------
``trial_data``   Published means, SEMs and n, transcribed with sources.
``recovery``     Recovers SDs and proves the transcription is faithful.
``inference``    Tests whether the subgroup effect exists, and corrects for
                 the number of endpoints examined. Read this first: everything
                 else is conditional on what it finds.
``literature``   External control-arm anchors from published trials, replacing
                 the single-arm design's missing comparator.
``power``        Sample size as a function of enrichment.
``assurance``    Probability of success, integrating over effect uncertainty.
``economics``    Cost model, including the screening penalty enrichment incurs.
``sensitivity``  Bootstrap and assumption sweeps. The only simulation here.
"""

__version__ = "3.0.0"

from . import (
    assurance,
    economics,
    inference,
    literature,
    power,
    recovery,
    sensitivity,
    trial_data,
)

__all__ = [
    "trial_data",
    "recovery",
    "inference",
    "literature",
    "power",
    "assurance",
    "economics",
    "sensitivity",
    "__version__",
]
