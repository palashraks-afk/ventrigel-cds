# Sizing a Phase II trial of VentriGel from published Phase I data

How large would a Phase II trial of an injectable cardiac extracellular matrix hydrogel have to be,
and how much smaller does it get if enrollment is restricted to the subgroup where the Phase I
actually saw an effect?

Everything here is derived from the published summary statistics of the VentriGel first-in-man
trial ([NCT02305602](https://clinicaltrials.gov/study/NCT02305602), Traverse et al.,
*JACC Basic Transl Sci* 2019;4(6):659–669). **No synthetic patients. No patient-level prediction.**

```bash
pip install -r requirements.txt
python run_analysis.py      # every number in the paper, < 1 min
python make_figures.py      # every figure
python test_ventrigel.py    # 29 tests
streamlit run app.py        # interactive trial design calculator
```

---

## The finding

The Phase I reported that improvements in left ventricular remodeling appeared mainly in patients
treated more than twelve months after their infarction. Reading the subgroup tables makes the
consequence concrete:

| LV end-systolic volume, 6-month change | value |
|---|---|
| Early stratum (<12 mo, n=6) | **+9.3 mL** (worse) |
| Late stratum (>12 mo, n=8) | **−7.6 mL** (better) |
| Pooled cohort (n=14) | **−0.35 mL** |

The pooled figure is not a weak treatment effect. It is two effects of comparable magnitude and
**opposite sign averaging to nothing**. That distinction decides the trial:

| | unselected | enriched to late |
|---|---|---|
| Effect vs. control | −0.29 mL (points at harm) | +7.60 mL |
| SD of change | 14.3 mL | 9.1 mL |
| **Randomized patients** | **no benefit to detect at any N** | **52** |

After a 25% winner's-curse discount the enriched trial needs 92 patients; propagating the
uncertainty in the 6–8 patients per stratum gives a bootstrap median of 108 with an 80% interval of
34–635.

### Enrichment is not universally good

This is the part worth keeping. What governs the gain is **the sign of the effect in the stratum
being excluded**, not the size of the pooled effect.

| endpoint | N unselected | N enriched | verdict |
|---|---|---|---|
| LV end-systolic volume | no benefit to detect | 52 | enrichment is the whole trial |
| Viable myocardial mass | 2,942 | 42 | enrichment decisive |
| MLWHFQ | 540 | 138 | enrichment helps |
| 6-minute walk | 50 | 32 | **barely worth it** — both strata improve |
| Ejection fraction | no benefit to detect | 1,748 | **enrichment is harmful** |

Ejection fraction is the instructive failure. Its pooled signal exists *only* because the early
stratum declined significantly (−3.8%, p=0.03) while the late stratum did not (−0.6%, p=0.69).
Enriching removes it. That signal is a possible safety observation, so erasing it is a governance
question, not just a statistical one.

---

## The caveat that drives everything

**The Phase I was single-arm.** It shows what happened to treated patients, not what would have
happened untreated. The early-vs-late gap is equally consistent with:

1. **Effect modification** — the therapy really works better late; or
2. **Differential natural history** — untreated early patients dilate on their own while late
   patients are stable.

These are not distinguishable from single-arm data, so control-arm drift is a **swept parameter**
throughout, not a footnote. Under the most conservative attribution — the early stratum's entire
+9.3 mL is natural history — the advantage falls from unbounded to about 4×, while the enriched
trial stays near 52–92 patients.

**The robust conclusion is the size of the enriched design, not the size of its advantage.**

One measurement would collapse most of this uncertainty: the 6-month LVESV change in *untreated*
patients matched on time since infarction. That is obtainable from existing observational cohorts
with serial CMR, with no new trial.

---

## Why you can trust the numbers

The Supplemental Appendix reports every endpoint as `mean (SEM)` with `n`, so
`SD = SEM × √n` recovers the variance structure exactly. Two independent checks confirm the
transcription, and both run on every execution:

**1. All 38 checkable published p-values reproduce** from the transcribed mean, SEM and n via
`t = mean/SEM` on `n−1` df. This tests transcription, the SD identity, and the stated test at once.

**2. Each pooled cohort reconstructs from its two strata** by the laws of total expectation and
total variance. The published totals were never inputs, so agreement is a real test — and it is
demanding, because the between-stratum term is driven by exactly the separation the argument rests
on.

| | reconstructed | published | error |
|---|---|---|---|
| LVESV mean | −0.36 mL | −0.35 mL | 0.01 mL |
| LVESV SD | 14.26 mL | 14.22 mL | **0.3%** |
| median SD error, all 9 endpoints | | | **3.3%** |

Four cells are excluded from check 1 with documented reasons rather than counted as passes: three
BNP cells (the table reports percent change but its t-test column tests absolute values) and
total-cohort scar fraction (an unresolved discrepancy in the source — the printed statistics imply
p=0.80, not the printed p=0.58). Both are recorded in `ventrigel/trial_data.py`, not silently
corrected.

---

## What changed from v1, and why

An earlier version of this project ([Zenodo 10.5281/zenodo.21516443](https://doi.org/10.5281/zenodo.21516443))
generated 2,000 synthetic patients from a hand-written eligibility scoring rule, trained a
classifier on them, and reported **ROC-AUC = 1.00**.

That result was circular. The label `Optimal_Candidate` was a deterministic threshold function of
the same features the model was given, so the classifier was recovering an `if` statement its
author had written. Dropping the intermediate `Suitability_Score` column removed the shortcut but
not the leakage, because the score is fully recoverable from the remaining columns. The reported
feature importances likewise reproduced the penalty weights chosen by hand, and the partial
dependence "thresholds" were the rule boundaries.

Fifteen single-arm patients cannot support individual response prediction, and simulation does not
manufacture the missing information. The v1 code is kept in `deprecated/` for comparison.

Simulation appears in v2 only in `sensitivity.py`, and only to propagate uncertainty already
present in the published estimates or to map assumptions. It never creates observations.

---

## Layout

```
ventrigel/
  trial_data.py    published means, SEMs, n — each with its source table
  recovery.py      SD recovery + the two validation checks
  power.py         sample size vs. enrichment (noncentral t)
  economics.py     cost model, including the screening penalty
  sensitivity.py   bootstrap and assumption sweeps — the only simulation
run_analysis.py    reproduces every number
make_figures.py    reproduces every figure
test_ventrigel.py  29 tests, power math checked against textbook values
app.py             Streamlit trial design calculator
paper/             manuscript source
deprecated/        the retired v1 classifier
```

## Citation

The source trial:

> Traverse JH, Henry TD, Dib N, Patel AN, Pepine C, Schaer GL, DeQuach JA, Kinsey AM, Chamberlin P,
> Christman KL. First-in-man study of a cardiac extracellular matrix hydrogel in early and late
> myocardial infarction patients. *JACC Basic Transl Sci.* 2019;4(6):659–669.
> doi:10.1016/j.jacbts.2019.07.012

This is an independent secondary analysis of published summary statistics. It was not conducted
with, funded by, or endorsed by Ventrix, Inc., the Christman Laboratory, or any trial investigator.
No patient-level data were accessed.
