# Draft email to Dr. Christman

**This is a draft. I have not sent it and will not — sending mail on your behalf, to a named person,
is yours to do.** Read it, change anything that does not sound like you, and send it from your own
account.

**Send it only after the Zenodo correction is live.** If she follows the DOI and lands on the v1
record with "ROC-AUC 1.00" in the abstract, the correction becomes something she discovered rather
than something you disclosed, and that is a materially worse first impression.

---

## A note on what changed from your original plan

Your earlier draft asked whether timing is a stronger predictor than baseline ejection fraction.
That question no longer fits the work, because the analysis now answers something adjacent and more
specific — and because the honest version of the finding is weaker than the version that question
presumes.

The email below leads with the correction, states the finding at its actual strength, and asks one
thing she is uniquely positioned to answer and that the published record does not contain. It does
not ask for endorsement, collaboration, or a review, which is what makes it answerable in two
minutes.

---

## Subject

`Interaction test on the VentriGel Phase I subgroups — one question about the chronic comparator`

## Body

> Dear Dr. Christman,
>
> I'm a high school student in Redlands, California. I've spent the last several months doing a
> secondary analysis of the 2019 VentriGel first-in-man trial, and I have one question that I don't
> think the published literature can answer. I want to be upfront about a correction first.
>
> An earlier version of this project, which I posted to Zenodo, trained a classifier on synthetic
> patients and reported a ROC-AUC of 1.00. That result was circular — the label I was predicting was
> a deterministic function of the features I was feeding the model, so it was recovering a rule I had
> written myself. I have published a correction notice and replaced the analysis entirely. I mention
> it because the old DOI is still findable and I would rather you hear it from me.
>
> What replaced it uses only the published summary statistics. Because your Supplemental Appendix
> reports each endpoint as mean with SEM and n by stratum, the standard deviations are recoverable
> exactly, and the whole variance structure follows without needing patient-level data. Two things
> came out of it that I thought might interest you:
>
> First, the early-versus-late contrast has never been tested directly — the trial compared each
> stratum against its own baseline. Testing the strata against each other gives a significant
> interaction for end-systolic volume only (−16.9 mL, 95% CI −32.2 to −1.6, p = 0.034), and it does
> not survive correction across the nine endpoints. The strata are balanced on every reported
> baseline and the pattern runs opposite to regression to the mean, so I don't think it's an
> artifact — but it is one nominal result out of nine, and I've been careful to present it that way.
>
> Second, the design consequence is larger than I expected. Because the strata move in opposite
> directions, the pooled −0.35 mL is a cancellation rather than a small effect, and an unselected
> Phase II on that endpoint has essentially nothing to detect.
>
> **My question.** Everything downstream turns on one number I cannot obtain: the six-month change in
> end-systolic volume in *untreated* chronic post-infarction patients with LVEF ≤ 45%. The two
> published estimates I could find disagree in a way that decides the project — FOCUS-CCTRN's placebo
> arm shows no change (n = 28), while FOCUS-HF's control arm shows roughly −10 mL (n = 10). Under the
> first, VentriGel's −7.6 mL in the late stratum is a real effect and an enriched Phase II is
> feasible. Under the second it is smaller than natural history and there is nothing to test.
>
> Is there a better estimate of that quantity than the two I found — from your preclinical work, from
> the trial's screening data, or from a cohort I've missed?
>
> The analysis, code and manuscript are at
> https://github.com/palashraks-afk/ventrigel-cds, and there's an interactive version at
> https://ventrigelcds.streamlit.app. I'd be glad to be told I've got something wrong.
>
> Thank you for your time.
>
> Palash Rakshit
> palash.raks@gmail.com

---

## Why it is built this way

- **The correction comes second, before any claim.** Disclosed, briefly, without over-apologising.
- **One question, and it is genuinely unanswerable from the literature.** She can reply in two
  sentences, which is the main determinant of whether a cold email gets answered.
- **The finding is stated at its real strength**, including that it fails multiplicity correction.
  Overstating it to a person who knows this data better than anyone is the fastest way to lose her.
- **No ask for endorsement, mentorship, or review.** Those come later, if at all.
- **Links last**, so the email reads as a question rather than a promotion.

## Before you send

- [ ] Zenodo correction published and the concept DOI resolving to v2
- [ ] The app loads (it does — verified after the last deploy)
- [ ] An adult sponsor or teacher has read the manuscript
- [ ] Read it aloud once; if a sentence doesn't sound like you, change it
