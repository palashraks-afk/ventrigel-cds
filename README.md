# Sizing a Phase II trial of VentriGel from published data

How large would a Phase II trial of an injectable cardiac extracellular matrix hydrogel have to be,
how likely is it to succeed, and which assumption is that answer most sensitive to?

Everything is derived from published summary statistics: the VentriGel first-in-man trial
([NCT02305602](https://clinicaltrials.gov/study/NCT02305602), Traverse et al., *JACC Basic Transl
Sci* 2019) plus eight published control and placebo arms.
**No synthetic patients. No patient-level prediction.**

**Live calculator:** <https://ventrigelcds.streamlit.app/>

```bash
pip install -r requirements.txt
python run_analysis.py      # every number in the paper, ~1 min
python make_figures.py      # all eight figures
python test_ventrigel.py    # 75 tests
streamlit run app.py        # interactive design calculator
```

---

## Read this first: how strong is the evidence?

**Weak.** The project is built on one post-hoc subgroup finding, and the analysis says so up front
rather than burying it.

The trial reported that remodeling improved mainly in patients treated >12 months post-MI. But it
only ever compared each stratum against *its own baseline*. It never tested the strata against
**each other**. Here is that test:

| endpoint | early | late | interaction p |
|---|---|---|---|
| **LVESV** | +9.3 | −7.6 | **0.034** |
| ejection fraction | −3.8 | −0.6 | 0.133 |
| viable mass | −10.5 | +15.0 | 0.169 |
| MLWHFQ | +0.7 | −15.1 | 0.332 |
| 6-min walk | +40.5 | +50.9 | 0.723 |

One of nine endpoints is nominally significant, and it **survives no multiplicity correction**.
Bonferroni needs p < 0.0056; Benjamini-Hochberg rejects everything. Even collapsing the
algebraically linked endpoints to five families (EF is determined by the two volumes) gives a
threshold of 0.010, which p = 0.034 still fails.

Two caveats on the test itself, both stated in the paper: it compares **change scores rather than
ANCOVA** (so p = 0.034 is probably conservative, but ANCOVA needs patient-level data), and the family
is nine endpoints **at the 6-month visit**, and counting the 1- and 3-month visits would enlarge it.

Two checks come back favourably:

- **The strata are balanced at baseline**, all eight measures, minimum p = 0.46.
- **It is not regression to the mean.** The early stratum began with the *higher* LVESV and got
  worse; regression predicts the opposite.

So it is a suggestive but fragile signal. Enough to design a trial around, not enough to assert.
**Everything below is conditional on it, and the last section prices that condition.**

---

## What untreated patients actually do

The Phase I was single-arm, so its comparator is missing. Rather than sweeping an arbitrary range,
this analysis anchors it to published control and placebo arms:

| source | population | control-arm 6-month change | n |
|---|---|---|---|
| PRESERVATION-I (2016) | large STEMI, 2-5 days | LVEDVI **+11.7** mL/m² | 102 |
| TIME (2012) | anterior STEMI, 3-7 days | LVESVI **+4.3** mL/m² | 37 |
| EMPRESS-MI (2025) | MI, LVEF<45%, modern GDMT | LVESVI **−7.8** mL/m² | 52 |
| FOCUS-CCTRN (2012) | chronic ischemic, LVEF≤45% | LVESVI **0.0** mL/m² | 28 |
| FOCUS-HF (2011) | chronic ischemic HF | LVESV **−9.9** mL | 10 |
| EMPRESS-MI (2025) | acute | LVEF **+8.5** points | 52 |
| FOCUS-CCTRN (2012) | chronic | LVEF **−1.3** points | 28 |
| Khan et al. (2022)\* | HF, 38 placebo arms | 6MWT **+4.2%** ≈ +17.6 m | 2,713 |

\* conference abstract, not a peer-reviewed full report, and labelled as such wherever used.

**The acute anchors disagree in sign, and that is the finding.** Older cohorts dilated; the
2022-2024 cohort on contemporary therapy underwent *reverse* remodeling. Post-MI natural history is
era-dependent.

Anchoring changes two endpoints qualitatively. Against a placebo arm whose EF *falls* 1.3 points,
VentriGel's late-stratum −0.6 is a small **benefit**, not the null the trial reported. And the
6-minute walk loses a third of its effect to the placebo response.

Where two anchors compete the choice is **explicit and documented**, not automatic. Selecting by
sample size would have picked EMPRESS-MI over TIME and reversed the early comparator's sign.

The comparators actually fed to the model, after de-indexing at BSA 1.9 m²:

| endpoint | early stratum | late stratum |
|---|---|---|
| LVESV | **+8.2 mL** (TIME) | **+0.0 mL** (FOCUS-CCTRN) |
| LVEDV | +22.2 mL (PRESERVATION-I) | *none* |
| ejection fraction | +8.5 pts (EMPRESS-MI) | −1.3 pts (FOCUS-CCTRN) |
| 6-min walk | *none* | +17.6 m (Khan) |
| viable mass, MLWHFQ | *none* | *none* |

---

## The comparator is an estimate, and that roughly doubles the trial

An anchor is an *estimate*, not a constant. FOCUS-CCTRN's "zero" comes from 28 patients, with a
standard error of **4.4 mL**, the same order as the 7.6 mL effect measured against it:

| | assurance at N=174 | ceiling | N for 80% success |
|---|---|---|---|
| comparator treated as exact | 80% | 97.5% | 174 |
| **comparator uncertainty propagated** | **72%** | **90.9%** | **406** |

Recovering that SE isn't direct: the trial publishes SDs of the *levels*, not of the change, so it
depends on a test-retest correlation nobody reports
(`SD_Δ = √(s₁² + s₂² − 2r·s₁s₂)`). At r = 0.95 the SE is 2.7 mL; at r = 0.70 it is 6.1 mL. Both that
correlation and the body-surface-area assumption are swept, and usefully the key anchor's point
estimate is exactly zero at *every* plausible BSA, so only its uncertainty moves.

---

## The design consequence

For LVESV the two strata move in *opposite* directions, so the pooled −0.35 mL is not a weak effect.
It is **+9.3 and −7.6 averaging to nothing**. Enrichment raises the effect *and* removes the
between-stratum variance, so gains compound in σ²/Δ²:

| endpoint | N unselected | N enriched | verdict |
|---|---|---|---|
| **LVESV** | 1,046 | **92** | enrichment is decisive (11.4×) |
| viable mass † | 5,230 | 70 | large, but interaction p = 0.17 |
| MLWHFQ † | 958 | 242 | helps |
| 6-min walk | 134 | 126 | **barely worth it** (1.1×), placebo response hits both strata |
| LVEDV | 126 | 366 | **enrichment is worse** here |
| ejection fraction | no benefit | 2,282 | technically detectable, not practically |

† no published anchor; assumes no control drift, so these are optimistic.

The governing quantity is the effect in the stratum being **excluded**, relative to its own
comparator, not the size of the pooled effect.

---

## Power is not probability of success

The 92-patient design was solved for 80% power **at a point estimate** resting on eight patients.
Assurance integrates power over uncertainty in both the effect and the comparator:

| N | nominal power | assurance |
|---|---|---|
| 92 | 85% | **63%** |
| 174 | 99% | 72% |
| **406** | 100% | **80%** |
| 1,000 | 100% | 85% |

Assurance **plateaus at 90.9%**, which is the share of plausible effects pointing toward benefit at all.
Enrollment cannot fix an effect that isn't there.

### And the number a sponsor actually needs

All of that is conditional on the subgroup effect being real. Unconditional probability of success is
that prior × assurance:

| N | prior 30% | prior 50% | prior 80% |
|---|---|---|---|
| 92 | 19% | 32% | 50% |
| 406 | 24% | **40%** | 64% |
| 800 | 25% | 42% | 67% |

**The prior enters multiplicatively, so no sample size lifts the programme above it.** Beyond ~400
patients the marginal return on enrollment is near zero. That is the quantitative case for
spending the next increment of money on the comparator rather than on patients.

---

## The enriched trial can't confirm the claim it rests on

A trial enrolling only late patients shows the therapy works *in that stratum*. It can never show
that **timing matters**, because there are no early patients to compare against. But "treat late, not early"
is the actual claim.

A 2×2 design powered on the interaction:

| control assumption | interaction contrast | N (2×2) | vs. enriched 2-arm |
|---|---|---|---|
| no control drift | 12.7 mL | 124 | 1.3× |
| **published control arms** | **6.5 mL** | **440** | 4.8× |

Anchoring halves the contrast, because most of the early stratum's apparent harm is natural history
rather than a failure of treatment. But **440 is close to the 406 an 80%-assurance enriched trial
needs**, so for roughly the same money you can answer the question you actually have.

---

## Why you can trust the arithmetic

`SD = SEM × √n` recovers the variance structure exactly from the published tables. Two independent
checks run on **every execution**:

**1. All 38 checkable published p-values reproduce** from the transcribed mean, SEM and n, testing
transcription, the SD identity, and the stated test at once.

**2. Each pooled cohort reconstructs from its two strata** by the laws of total expectation and total
variance. The published totals were never inputs:

| | reconstructed | published | error |
|---|---|---|---|
| LVESV mean | −0.36 mL | −0.35 mL | 0.01 mL |
| LVESV SD | 14.26 mL | 14.22 mL | **0.3%** |
| median SD error, 9 endpoints | | | **3.3%** |

Four cells are excluded with documented reasons rather than counted as passes: three BNP cells (the
table reports percent change but its t-test column tests absolute values) and total-cohort scar
fraction (the printed statistics imply p = 0.80, not the printed p = 0.58).

---

## What changed from v1, and why

An earlier version ([Zenodo 10.5281/zenodo.21516443](https://doi.org/10.5281/zenodo.21516443))
generated 2,000 synthetic patients from a hand-written scoring rule, trained a classifier, and
reported **ROC-AUC = 1.00**.

That result was circular: the label was a deterministic threshold function of the same features the
model was given, so the classifier recovered an `if` statement its author had written. See
[`CORRECTION.md`](CORRECTION.md) and `deprecated/`.

Simulation appears in this version only to propagate uncertainty already present in the published
estimates. It never creates observations.

---

## Layout

```
ventrigel/
  trial_data.py    published means, SEMs, n, each with its source table
  recovery.py      SD recovery + the two validation checks
  inference.py     interaction tests, multiplicity, balance, RTM  ← read first
  literature.py    external control anchors, their SEs, and BSA/retest sweeps
  power.py         sample size vs. enrichment, plus the 2×2 interaction design
  assurance.py     probability of success, and prior × assurance
  economics.py     cost model, including the screening penalty
  sensitivity.py   bootstrap and assumption sweeps
run_analysis.py    reproduces every number (10 sections)
make_figures.py    reproduces all eight figures
test_ventrigel.py  75 tests; power math checked against textbook values
check_manuscript.py verifies every number quoted in the paper against results/
app.py             Streamlit design calculator
paper/             manuscript source
deprecated/        the retired v1 classifier, with a written post-mortem
```

## Citation

Source trial:

> Traverse JH, Henry TD, Dib N, Patel AN, Pepine C, Schaer GL, DeQuach JA, Kinsey AM, Chamberlin P,
> Christman KL. First-in-man study of a cardiac extracellular matrix hydrogel in early and late
> myocardial infarction patients. *JACC Basic Transl Sci.* 2019;4(6):659-669.
> doi:10.1016/j.jacbts.2019.07.012

External anchors are cited in full in `ventrigel/literature.py` and listed with PMIDs in
[`SOURCES.md`](SOURCES.md).

This is an independent secondary analysis of published summary statistics. It was not conducted
with, funded by, or endorsed by Ventrix, Inc., the Christman Laboratory, or any trial investigator.
No patient-level data were accessed. MIT licensed.
