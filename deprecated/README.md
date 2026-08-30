# Retired: the synthetic-cohort classifier (v1)

This directory holds the first version of the project. It is kept for comparison and for honesty
about what changed. **Do not use it.** Its headline result is an artifact of its own construction.

## What it did

`synthetic_generator.py` created 2,000 synthetic patient records. For each one it computed a
`Suitability_Score` starting at 100 and deducting points for violating trial criteria (ejection
fraction outside 25–45%, time since infarction outside 60–1095 days) and for soft risk factors.
It then thresholded that score at 60 to produce a binary label, `Optimal_Candidate`.

`train_model.py` dropped `Suitability_Score` and `Patient_ID`, trained five classifiers on the
remaining features to predict `Optimal_Candidate`, and selected a Random Forest reporting
**ROC-AUC 1.00, accuracy 99.75%, F1 99.83%**.

## Why the result is meaningless

`Optimal_Candidate` is a deterministic function of the same features the model was given. The
classifier was not learning a biological relationship; it was recovering a conditional statement
the author had written a few dozen lines earlier. Near-perfect accuracy was the arithmetic
consequence of the setup — a tree ensemble recovering axis-aligned thresholds from noiseless
labels. Anything less would have indicated a bug.

Dropping `Suitability_Score` was described in the v1 manuscript as "data leakage prevention." It
removed the *shortcut* but not the *leakage*: the score is fully recoverable from the columns that
remained, so the target was still determined by the inputs.

Three downstream claims inherited the problem:

- **Feature importances.** Ejection fraction and days post-MI dominating the decision space was not
  a discovery. Those two variables carried the heaviest hand-written penalties.
- **Partial dependence thresholds.** The reported probability cliff below 28% ejection fraction is
  the 25% rule boundary plus tree granularity, not a biological transition.
- **The claimed drop-off past 24 months.** No such boundary exists anywhere in the generator; the
  rule window ran to 1,095 days (36 months). The v1 manuscript presented it as confirming a
  biological hypothesis. It should not have appeared at all.

The v1 manuscript also described a `GridSearchCV` hyperparameter search converging at 300 estimators
and depth 12. The training script in this directory contains no grid search.

## The deeper problem

Fifteen patients in a single-arm trial cannot support individual response prediction. Generating
2,000 synthetic records does not add information; it adds rows. Whatever structure the generator
was given is the only structure any model can find in its output.

## What replaced it

The current version asks a question the published data can actually answer: given the subgroup
effects the Phase I reported, how large must a Phase II be, and how much does enrichment change
that? It uses no synthetic patients. Simulation survives only in `ventrigel/sensitivity.py`, where
it propagates uncertainty already present in the published estimates rather than inventing
observations.

See the root `README.md`.

## Files

| file | what it was |
|---|---|
| `synthetic_generator.py` | built the 2,000-record synthetic cohort |
| `train_model.py` | the five-model "arena" |
| `app_classifier_v1.py` | Streamlit patient-candidacy scorer |
| `synthetic_ventrigel_cohort*.csv` | generated cohorts |
| `*.joblib` | serialized pipelines |
