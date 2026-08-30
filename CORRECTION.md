# Correction notice

**Concerning:** *Computational Patient Selection for Extracellular Matrix Therapies: A Machine
Learning Framework for Phase II Trial Optimization*, Zenodo, DOI
[10.5281/zenodo.21516443](https://doi.org/10.5281/zenodo.21516443) (v1).

**Status:** superseded. The headline result of that version is invalid, and this document explains
why in enough detail that a reader can verify the diagnosis independently.

---

## The claim that was wrong

Version 1 reported a Random Forest classifier achieving **ROC-AUC 1.00, accuracy 99.75%, and F1
99.83%** on the task of identifying optimal candidates for VentriGel therapy, and presented this as
evidence that machine learning could optimize Phase II patient selection.

That performance was an artifact of the study's own construction and carries no information about
patients, biology, or trial design.

## Why it is invalid

The training data were 2,000 synthetic records produced by `synthetic_generator.py`. For each
record the script computed a `Suitability_Score`, starting at 100 and deducting points for
violating trial criteria (ejection fraction outside 25-45%, time since infarction outside 60-1095
days) and for soft risk factors, then thresholded that score at 60 to produce the binary label
`Optimal_Candidate`.

`train_model.py` then dropped `Suitability_Score` and trained classifiers to predict
`Optimal_Candidate` from the remaining features.

**The label is a deterministic function of those remaining features.** The classifier was not
learning a relationship in data; it was recovering a conditional statement written a few dozen lines
earlier in the same repository. Near-perfect accuracy was the arithmetic consequence of a tree
ensemble recovering axis-aligned thresholds from noiseless labels. A materially lower score would
have indicated a bug.

The v1 manuscript described dropping `Suitability_Score` as "data leakage prevention." That removed
the *shortcut* but not the *leakage*: the score is fully recoverable from the columns that remained,
so the target stayed determined by the inputs.

## Specific claims that must be withdrawn

1. **The performance metrics.** ROC-AUC 1.00, accuracy 99.75%, precision 100%, recall 99.66% and
   F1 99.83% do not describe predictive ability. They describe the recoverability of a rule from its
   own outputs.

2. **The feature importances.** "Ejection fraction and days post-MI dominate over 80% of the
   classification decision space" was not a finding. Those two variables carried the heaviest
   hand-written penalties in the generator.

3. **The partial-dependence thresholds.** The reported probability cliff below 28% ejection fraction
   is the 25% rule boundary plus tree granularity, not a biological transition. The reported
   efficacy drop-off beyond 24 months post-MI corresponds to nothing in the generator at all. Its
   rule window ran to 1,095 days (36 months). That figure should not have appeared.

4. **The hyperparameter search.** The manuscript described a `GridSearchCV` over 100-500 estimators
   and depths 5-20 "converging at 300 estimators with a maximum depth of 12." The training script in
   the repository contains no grid search.

5. **The commercial estimate.** "A 30 percent reduction in cohort size through optimized screening
   can save upwards of $10 million per trial phase" was attributed to Moore et al. (2018) and an FDA
   web page. Neither source supports that figure.

## The underlying methodological error

Fifteen patients in a single-arm trial cannot support individual response prediction. Generating
2,000 synthetic records does not add information; it adds rows. Whatever structure the generator was
given is the only structure any model can find in its output. The error was not a coding mistake but
a category mistake about what simulation can do.

## What replaces it

Version 2 of this project asks a question the published data can actually answer: given the subgroup
effects the Phase I reported, how large must a Phase II be, how likely is it to succeed, and which
assumption is that answer most sensitive to?

It uses no synthetic patients. It also reaches a considerably more cautious conclusion than v1: the
subgroup finding that motivates the whole analysis is nominally significant for exactly one of nine
endpoints (p = 0.034) and survives no correction for multiplicity, and the analysis is presented as
conditional on it throughout.

Simulation survives only for propagating uncertainty already present in the published estimates.

- Repository: https://github.com/palashraks-afk/ventrigel-cds
- The v1 code is retained under `deprecated/` with a per-file post-mortem.

## Acknowledgement

The error was mine. It was found by re-reading the generator and the training script together and
noticing that the target variable was computed from the features, which is checkable in under a
minute by anyone with the repository open. The v1 record is left online rather than deleted so the
correction is discoverable by anyone who encounters it.

*Palash Rakshit*
