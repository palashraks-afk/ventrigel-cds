# Sizing a Phase II trial of VentriGel from published data

How large would a Phase II trial of an injectable cardiac extracellular matrix hydrogel have to be,
how likely is it to succeed, and which assumption is that answer most sensitive to?

Everything is derived from published summary statistics: the VentriGel first-in-man trial
([NCT02305602](https://clinicaltrials.gov/study/NCT02305602), Traverse et al., *JACC Basic Transl
Sci* 2019) plus the control arms of five other randomized trials.
**No synthetic patients. No patient-level prediction.**

```bash
pip install -r requirements.txt
python run_analysis.py      # every number in the paper, ~1 min
python make_figures.py      # all nine figures
python test_ventrigel.py    # 57 tests
streamlit run app.py        # interactive design calculator
```

---

## Read this first: how strong is the evidence?

**Weak.** The project is built on one post-hoc subgroup finding, and the analysis says so up front
rather than burying it.

The trial reported that remodeling improved mainly in patients treated >12 months post-MI. But it
only ever compared each stratum against *its own baseline* — it never tested the strata against
**each other**. That test was missing from the literature and from the first version of this
project. Here it is:

| endpoint | early | late | interaction p |
|---|---|---|---|
| **LVESV** | +9.3 | −7.6 | **0.034** |
| ejection fraction | −3.8 | −0.6 | 0.133 |
| viable mass | −10.5 | +15.0 | 0.169 |
| MLWHFQ | +0.7 | −15.1 | 0.332 |
| 6-min walk | +40.5 | +50.9 | 0.723 |

One of nine endpoints is nominally significant, and it **survives no multiplicity correction**.
Bonferroni needs p < 0.0056; Benjamini–Hochberg rejects everything. Even collapsing the
algebraically linked endpoints to five independent families (ejection fraction is determined by the
two volumes) gives a threshold of 0.010, which p = 0.034 still fails.

Two checks come back favourably, and both are worth stating:

- **The strata are balanced at baseline** — all eight measures, minimum p = 0.46. The contrast is
  not confounded by differing severity at entry.
- **It is not regression to the mean.** The early stratum began with the *higher* LVESV and got
  worse; regression predicts the opposite direction.

So: a suggestive, fragile signal — enough to design a trial around, not enough to assert.
**Everything below is conditional on it.**

---

## The design consequence

For LVESV the two strata move in *opposite* directions, so the pooled −0.35 mL is not a weak
effect. It is **+9.3 and −7.6 averaging to nothing**:

| | unselected | enriched to late |
|---|---|---|
| Effect vs. control | +2.64 mL | +5.70 mL |
| SD of change | 14.3 mL | 9.1 mL |
| **N for 80% power** | **1,046** | **92** |

Enrichment works on both terms at once — it raises the effect *and* removes the between-stratum
variance — so the gains compound in σ²/Δ².

### Enrichment is not universally good

The transferable finding is that the gain depends on **the sign of the effect in the stratum being
excluded**, not on the size of the pooled effect:

| endpoint | N unselected | N enriched | verdict |
|---|---|---|---|
| LVESV | 1,046 | 92 | enrichment is decisive |
| viable mass | 5,230 | 70 | large, but interaction p = 0.17 |
| MLWHFQ | 958 | 242 | helps |
| 6-min walk | 86 | 56 | **barely worth it** — both strata improve |
| ejection fraction | no benefit | no benefit | **enrichment is harmful** |

Ejection fraction is the instructive failure: its pooled signal exists *only* because the early
stratum declined significantly (−3.8%, p = 0.03). Enriching erases it — and since that is a possible
safety signal, erasing it is a governance question, not just a statistical one.

---

## Power is not probability of success

The 92-patient design was solved for 80% power **at a point estimate** that rests on eight patients.
Integrating power over the uncertainty in the effect — *assurance* — gives the real number:

| N | assurance |
|---|---|
| 52 | 53% |
| **92** | **68%** |
| 174 | 80% |
| 442 | 90% |

Assurance also **plateaus at 97.5%**. No sample size does better, because that is the share of
plausible effects that point toward benefit at all. Enrollment cannot fix an effect that is not
there.

The recommended design is therefore **174 patients**, not 92 — and at six sites it would take 82
months to enroll. Finishing in 30–36 months needs **17–21 sites**. That is a recruitment finding,
and it is the constraint a sponsor hits first.

---

## The number that decides everything

The Phase I was single-arm, so the comparator is missing. Rather than sweeping an arbitrary range,
this analysis anchors it to published control arms:

| trial | population | control-arm 6-month change | n |
|---|---|---|---|
| PRESERVATION-I (2016) | large STEMI, 2–5 days | LVEDVI **+11.7** mL/m² | 102 |
| TIME (2012) | anterior STEMI, 3–7 days | LVESVI **+4.3** mL/m² | 37 |
| EMPRESS-MI (2025) | MI, LVEF<45%, modern GDMT | LVESVI **−7.8** mL/m² | 52 |
| FOCUS-CCTRN (2012) | chronic ischemic, LVEF≤45% | LVESVI **0.0** mL/m² | 28 |
| FOCUS-HF (2011) | chronic ischemic HF | LVESV **−9.9** mL | 10 |

**The acute anchors disagree in sign, and that is the finding.** Older cohorts dilated; the
2022–2024 cohort on contemporary therapy underwent *reverse* remodeling. Post-MI natural history is
era-dependent.

Chronic populations are stable — which matters, because the enriched trial enrolls no early
patients. Its size is unaffected by the early assumption and depends **entirely** on the late one:

- **FOCUS-CCTRN says 0.0 mL** → the design stands at 92–174 patients.
- **FOCUS-HF says −9.9 mL** → VentriGel's −7.6 mL is *smaller than natural history*, there is no
  effect, and no trial is worth running.

One number decides the project, and the two available estimates disagree. Measuring the six-month
LVESV change in untreated chronic post-MI patients with LVEF ≤ 45% would settle it — obtainable from
existing observational cohorts with serial CMR, with no new trial.

---

## Why you can trust the arithmetic

The Supplemental Appendix reports every endpoint as `mean (SEM)` with `n`, so `SD = SEM × √n`
recovers the variance structure exactly. Two independent checks run on **every execution**:

**1. All 38 checkable published p-values reproduce** from the transcribed mean, SEM and n via
`t = mean/SEM` on `n−1` df — testing transcription, the SD identity, and the stated test at once.

**2. Each pooled cohort reconstructs from its two strata** by the laws of total expectation and
total variance. The published totals were never inputs, so agreement is a real test:

| | reconstructed | published | error |
|---|---|---|---|
| LVESV mean | −0.36 mL | −0.35 mL | 0.01 mL |
| LVESV SD | 14.26 mL | 14.22 mL | **0.3%** |
| median SD error, 9 endpoints | | | **3.3%** |

Four cells are excluded with documented reasons rather than counted as passes: three BNP cells (the
table reports percent change but its t-test column tests absolute values) and total-cohort scar
fraction (the printed statistics imply p = 0.80, not the printed p = 0.58). Both are recorded in
`ventrigel/trial_data.py`, not silently corrected.

---

## What changed from v1, and why

An earlier version ([Zenodo 10.5281/zenodo.21516443](https://doi.org/10.5281/zenodo.21516443))
generated 2,000 synthetic patients from a hand-written scoring rule, trained a classifier, and
reported **ROC-AUC = 1.00**.

That result was circular. The label was a deterministic threshold function of the same features the
model was given, so the classifier recovered an `if` statement its author had written. Dropping the
intermediate score column removed the shortcut but not the leakage. Reported feature importances
reproduced hand-chosen penalty weights; a described `GridSearchCV` does not appear in the code.

Fifteen single-arm patients cannot support individual response prediction, and simulation does not
manufacture the missing information. See [`CORRECTION.md`](CORRECTION.md) and `deprecated/`.

Simulation appears in this version only in `sensitivity.py` and `assurance.py`, and only to
propagate uncertainty already present in the published estimates. It never creates observations.

---

## Layout

```
ventrigel/
  trial_data.py    published means, SEMs, n — each with its source table
  recovery.py      SD recovery + the two validation checks
  inference.py     interaction tests, multiplicity, balance, RTM  ← read first
  literature.py    external control-arm anchors, with citations
  power.py         sample size vs. enrichment (noncentral t)
  assurance.py     probability of success, integrating effect uncertainty
  economics.py     cost model, including the screening penalty
  sensitivity.py   bootstrap and assumption sweeps
run_analysis.py    reproduces every number (9 sections)
make_figures.py    reproduces all nine figures
test_ventrigel.py  57 tests; power math checked against textbook values
app.py             Streamlit design calculator
paper/             manuscript source
deprecated/        the retired v1 classifier, with a written post-mortem
```

## Citation

Source trial:

> Traverse JH, Henry TD, Dib N, Patel AN, Pepine C, Schaer GL, DeQuach JA, Kinsey AM, Chamberlin P,
> Christman KL. First-in-man study of a cardiac extracellular matrix hydrogel in early and late
> myocardial infarction patients. *JACC Basic Transl Sci.* 2019;4(6):659–669.
> doi:10.1016/j.jacbts.2019.07.012

External anchors are cited in full in `ventrigel/literature.py`.

This is an independent secondary analysis of published summary statistics. It was not conducted
with, funded by, or endorsed by Ventrix, Inc., the Christman Laboratory, or any trial investigator.
No patient-level data were accessed. MIT licensed.
