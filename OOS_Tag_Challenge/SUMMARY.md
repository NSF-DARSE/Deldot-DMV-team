# Out-of-State Tag Holder Review — one-page summary

**Deliverable:** `case_predictions.csv` — 24,000 rows (12,000 cases x T0/T1) in the
template's exact schema. Full write-up in `METHOD.md`.

## The problem behind the problem

The package gives no key linking `candidate_records.csv` to the six evidence feeds — only
names, and the names are deliberately corrupted. Getting the identity layer right mattered
more than the choice of classifier, so that is where the work went.

Using `vehicle_ref` on the title feed as an independent check, the corruption pattern is
**3 good rows plus 1 replaced outright**: cluster the owner names inside each vehicle's
four rows and 98% split three-and-one, with the fourth name unrelated to the other three.
Grouping the licence feed by date of birth reproduces the same pattern, so it is the
generator's noise model, not a quirk of one table.

Then the part that mattered most: **the replaced name is a persistent alias, not fresh
noise.** The same fake name recurs across feeds at 43% against a 3.8% chance baseline, and
4,906 of 5,278 aliases point at exactly one candidate. So the alias is learned on the
title feed, where `vehicle_ref` reveals whose row it is, and then used to claim the
matching rows in the feeds that have no key of their own — 8,067 aliases, recovering 7,833
records that name matching drops.

What is *not* recoverable is left alone. Forcing unclaimed rows onto candidates made the
signal worse, because an unrelated name carries an unrelated state.

## Two findings that shaped the model

**The classes are ordered.** In the development labels no case ever jumps from
`review_not_warranted` to `review_warranted` between phases — every move passes through
`insufficient_evidence`. That is one latent score with two thresholds, so the model is a
**proportional-odds ordinal logistic**: one direction vector and two cut-points instead of
three independent classifiers, which matters at 300 labelled candidates.

Every family was benchmarked on identical folds with a sweep each, and the boosters were
given both a real categorical and the same ordinal decomposition: ordinal logistic 0.907,
multinomial logistic 0.924, random forest 0.951, CatBoost 0.964, XGBoost 0.983, LightGBM
1.047, and an LLM classifier 1.151 — worse than guessing the base rates. Every tree
ensemble loses to plain logistic regression here, because 600 rows against 41 features is
far less data than boosting needs and the signal is a smooth monotone function that trees
can only approximate as a staircase.

**The current address is the single best record in the data.** The address row with a
blank end date is where the person lives now, and on its own it out-predicts the entire
address feed (rho +0.41 against +0.32). The direction is recency generally: a 150-day
exponential half-life beats every alternative aggregation tried, including a change-point
fit. The work-location feed is complete, uncorrupted and correlates -0.07 with the answer
— it is excluded.

## Results

Grouped 5-fold cross-validation, 20 repeats, 600 labelled rows:

| | model | baseline |
|---|---:|---:|
| log loss | **0.906** | 1.096 |
| accuracy | **0.558** | 0.355 |
| macro F1 | **0.560** | 0.175 |
| macro AUC | **0.740** | 0.500 |
| priority ranking AUC | **0.811** | 0.500 |

T1 scores better than T0 (0.880 vs 0.931) — the model moves on the later evidence.
Probabilities are calibrated with no post-hoc fitting. Errors sit on the adjacent class:
only 31 of 600 rows cross the whole scale.

`review_priority = p(warranted) + 0.5 x p(insufficient)` — a case that warrants review
takes full priority, a case the evidence cannot decide still needs a person and takes half.

## Is that number honest?

Those figures come from the same 300 labels that drove roughly forty design comparisons,
which inflates them. **Nested selection** — choosing the feature set and regularisation
inside a training split, then scoring once on unseen candidates — gives **0.913**, so the
selection bias is 0.008. A paired bootstrap puts the improvement over our first version at
0.022 log loss, 95% interval [+0.003, +0.040], better in 98.8% of resamples. Treat 0.913
as the number to expect on held-out cases.

## Honest ceiling

Nine ideas were tested and rejected, including semi-supervised self-training (worse at
every setting) and two further routes to recovering the unattributable rows. Each
candidate has roughly fourteen state-bearing records split between two states, so the
latent direction can only be estimated so precisely.

## For the reviewer

`case_audit.csv` carries the same 24,000 rows plus the evidence behind each score — newest
record per feed with its date, evidence freshness, T1 update direction, and the signed
contribution of each feature block — so a case can be checked without reading the code.
`diagnostics.py`, `model_selection.py`, `audit_*.py` reproduce every number quoted above.

The output supports staff review. It is not a residency, fee, or enforcement determination.
