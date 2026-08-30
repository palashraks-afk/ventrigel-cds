# Source documents in this repository

`Ventri PDF/JACC VentriGel.pdf` and `Supplemental Material.pdf` are the primary source for every
number in `ventrigel/trial_data.py`. They are included so the transcription can be checked against
the original without hunting for it.

> Traverse JH, Henry TD, Dib N, Patel AN, Pepine C, Schaer GL, DeQuach JA, Kinsey AM, Chamberlin P,
> Christman KL. First-in-man study of a cardiac extracellular matrix hydrogel in early and late
> myocardial infarction patients. *JACC Basic Transl Sci.* 2019;4(6):659–669.
> doi:[10.1016/j.jacbts.2019.07.012](https://doi.org/10.1016/j.jacbts.2019.07.012)

That article is published open access under
[CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/), which permits redistribution
of the unmodified work for non-commercial purposes with attribution. The PDFs here are unmodified
and are attributed above. They are **not** covered by this repository's MIT license, which applies
only to the code and manuscript authored here.

The five external control-arm anchors in `ventrigel/literature.py` are **not** redistributed. Each
carries a full citation and PubMed ID so the values can be checked at source:

| Trial | Citation | PMID |
|---|---|---|
| TIME | Traverse JH, et al. *JAMA.* 2012;308(22):2380–2389 | 23129008 |
| PRESERVATION-I | Rao SV, et al. *J Am Coll Cardiol.* 2016;68(7):715–723 | 27515331 |
| EMPRESS-MI | Carberry J, et al. *Eur J Heart Fail.* 2025;27(3):566–576 | 39675781 |
| FOCUS-CCTRN | Perin EC, et al. *JAMA.* 2012;307(16):1717–1726 | 22447880 |
| FOCUS-HF | Perin EC, et al. *Am Heart J.* 2011;161(6):1078–1087 | 21641354 |
| Khan et al.\* | Khan MA, et al. *J Card Fail.* 2022;28(5):S90 | not indexed |

\* **Conference abstract, not a peer-reviewed full report.** Identified via Crossref (doi:10.1016/j.cardfail.2022.03.226); the S-prefixed page number places it in a meeting supplement, and it is absent from PubMed. It is used because it is the only pooled estimate of placebo-arm walk distance located, and because leaving that endpoint unanchored would flatter it relative to the anchored volume endpoints. It is labelled `evidence="abstract"` in the code and in every table it appears in.

Author lists, journals, volumes and pages for all five were verified against PubMed via the NCBI
E-utilities API rather than transcribed from memory or from a search-result snippet.
