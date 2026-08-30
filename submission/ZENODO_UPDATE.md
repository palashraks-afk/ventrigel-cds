# Zenodo: publishing the corrected version

**I cannot do this step.** It requires signing in to your Zenodo account, and I will not
authenticate as you. Everything you need is prepared below; the actual upload is about five minutes
of your time.

Do this **before** emailing anyone, so that whoever follows the DOI lands on the corrected record
rather than on v1.

---

## Why a new version, not a new record

Zenodo's "New version" keeps the concept DOI (`10.5281/zenodo.21516443`) resolving to the newest
version while preserving v1 at its own versioned DOI. That is exactly what you want: the old record
stays citable and visible, which is honest, but anyone arriving at the concept DOI sees the
correction.

**Do not delete v1.** A deleted record with a live DOI is worse than a corrected one, and the
correction is only meaningful if the thing being corrected remains visible.

---

## Steps

1. Go to <https://doi.org/10.5281/zenodo.21516443> and sign in.
2. Click **New version**.
3. **Remove** all five old files: `train_model.py`, `synthetic_generator.py`, `app.py`,
   `Rakshit_VentriGel_CDS_Paper.pdf.pdf` and the `.zip`.
4. **Upload** these three files, all of which are in `Desktop\ZENODO UPLOAD`:
   - `ventrigel_enrichment.pdf`, the rewritten manuscript
   - `correction_notice.pdf`, the retraction notice
   - `ventrigel-cds-v2.0-source.zip`, the complete source, verified reproducible from a clean extract
5. Change the **Title** to:

   > What One Fragile Subgroup Is Worth: Enrichment, Comparator Uncertainty, and Probability of
   > Success for a Phase II Trial of a Cardiac Extracellular Matrix Hydrogel

6. Replace the **Description/Abstract** with the block below.
7. Under **Related identifiers**, add `https://github.com/palashraks-afk/ventrigel-cds`
   as *"is supplemented by"*.
8. Set **Version** to `2.0`.
9. Publish.

---

## Abstract to paste

> **Correction notice.** This version supersedes v1 (*Computational Patient Selection for
> Extracellular Matrix Therapies: A Machine Learning Framework for Phase II Trial Optimization*),
> whose headline result, a Random Forest classifier reporting ROC-AUC 1.00, was invalid. The
> classifier's target variable was a deterministic threshold function of the same features it was
> given, so the reported accuracy measured the recoverability of a hand-written rule from its own
> output rather than any predictive ability. Feature importances reproduced hand-chosen penalty
> weights, a described hyperparameter search does not appear in the code, and a reported temporal
> threshold corresponds to nothing in the data generator. Full details in the accompanying correction notice.
>
> **Background.** The VentriGel first-in-man trial (NCT02305602) treated 15 post-infarction patients
> with an injectable decellularized porcine myocardial matrix and reported, post hoc, that
> improvements in left ventricular remodeling appeared mainly in patients treated more than twelve
> months after infarction. This study asks what that observation implies for a Phase II design, and
> how much of it survives scrutiny.
>
> **Methods.** No simulated patients were used. The trial's Supplemental Appendix reports every
> endpoint as mean with standard error and n by stratum, which determines the standard deviation
> exactly. Transcription was validated by recomputing all 38 checkable published p-values and by
> reconstructing each pooled moment from its strata. The between-stratum interaction, which the
> trial did not test, was evaluated and corrected for multiplicity. The missing control arm was
> anchored to eight published control and placebo arms, and those anchors' own standard errors were
> propagated. Results are reported as assurance, meaning power integrated over effect and comparator
> uncertainty, and then multiplied by the probability that the subgroup effect is real at all.
>
> **Results.** All 38 p-values reproduced; median SD reconstruction error 3.3%. Exactly one of nine
> endpoints shows a nominally significant interaction (end-systolic volume, −16.9 mL, 95% CI −32.2
> to −1.6, p = 0.034); it survives neither Bonferroni nor Benjamini-Hochberg correction, though the
> strata are balanced on all eight baseline measures and the pattern runs opposite to regression to
> the mean. Post-infarction natural history proved era-dependent and sign-discordant for acute
> patients, while chronic populations are stable. Under anchored comparators an unselected trial
> requires 1,046 patients against 92 enriched. However the chronic comparator carries a standard
> error of 4.4 mL, itself comparable to the 7.6 mL effect; propagating it raises the enrollment for
> 80% probability of success from 174 to 406, and caps assurance at 90.9%. At an even-odds prior on
> the subgroup effect the unconditional probability of success is 40%.
>
> **Conclusions.** Enrichment converts an unpowerable trial into a feasible one, but the conclusion
> rests on a fragile interaction and on a comparator estimated from 28 patients. A 2×2 design
> powered on the interaction costs 440 patients, close to the 406 an 80%-assurance enriched trial
> needs, and answers the claim actually being made. The highest-value next expenditure is not more
> patients but a better estimate of the six-month volume change in untreated chronic
> post-infarction patients.

---

## Keywords to set

`clinical trial design`, `enrichment`, `sample size`, `assurance`, `probability of success`,
`subgroup analysis`, `multiplicity`, `myocardial infarction`, `extracellular matrix`,
`cardiac remodeling`, `secondary analysis`

## After publishing

Check that the concept DOI resolves to the new version, then update the DOI wherever you have
already cited it.
