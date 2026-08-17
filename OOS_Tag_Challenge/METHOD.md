# Potential Out-of-State Tag Holder Review — Method

A decision-support scorer for the 12,000 challenge cases. For each case it produces a
review classification, three class probabilities, and a review-priority value, at both
T0 and T1.

Everything below is reproducible from the code in this folder:
`diagnostics.py` prints the evidence for each structural claim,
`model_selection.py` and `audit_*.py` print the comparisons behind each modelling choice,
`audit_honest.py` prints the unbiased estimate, and `predict.py` runs the whole thing.

---

## 1. What the data actually is

Nothing in the package names a join key between `candidate_records.csv` and the six
evidence feeds. The feeds carry only names (and a date of birth on the licence feed).
Three properties of the data drive every design decision that follows.

**Each feed holds one record set per candidate.** The address, licence, title and
external feeds each hold 4.01 rows per candidate; the work-location feed and the T1
update stream each hold ~2.0. These are not coincidences across five tables — the
evidence was generated per candidate, so linking is an *assignment* problem, not an
independent per-row lookup.

**Names are corrupted on a 3-good-plus-1-replaced pattern.** The title feed is the one
place this can be verified without assuming an answer, because `vehicle_ref` groups a
vehicle's history independently of who is named on it. Clustering owner names inside
each 4-row vehicle gives a 3 + 1 split for 98% of vehicles: three rows carrying one
name, lightly corrupted (truncated to an initial, one substituted character, an
adjacent transposition, a stray space), and one row carrying a name with no
relationship to the other three. Grouping the licence feed by date of birth reproduces
the same 3 + 1 pattern, so it is the generator's noise model rather than a quirk of one
table.

**The replaced name is a persistent alias, not fresh noise.** That is what makes the
fourth row recoverable in feeds beyond titles; see section 2.

**The three classes are ordered.** In the 300 development cases, no case moves from
`review_not_warranted` to `review_warranted` between phases (0 cases), and one moves the
other way. Every other move passes through `insufficient_evidence`. That is the
signature of a single latent score with two thresholds, not three unrelated categories.

| T0 \ T1 | insufficient | not warranted | warranted |
|---|---:|---:|---:|
| insufficient_evidence | 71 | 22 | 12 |
| review_not_warranted | 20 | 85 | **0** |
| review_warranted | 12 | **1** | 77 |

Reading the axis against the evidence: cases whose *recent* records point to Delaware
are labelled `review_warranted`; cases whose recent records point to another state are
`review_not_warranted`; cases in between are `insufficient_evidence`. That matches the
operational question — a vehicle whose owner now appears to be operating in Delaware is
the one a staff member should look at.

---

## 2. Identity resolution

Resolution runs in three passes.

**Pass 1 — titles, grouped by `vehicle_ref`.** Rows are clustered by name inside each
vehicle; each cluster is assigned to the candidate it agrees with best; the
replaced-name singleton is attached to the vehicle's dominant owner at a reduced weight.
This recovers the fourth row that name matching cannot reach: 38,582 to 47,581 linked
rows, and the feed's Spearman rho against the ordered label rises from +0.25 to +0.37.

**Pass 2 — the alias.** The replaced name is not drawn fresh per row. Taking the
odd-one-out name from each clean 3 + 1 vehicle and looking for it in the other feeds:

| | replaced name found | recombined-name control |
|---|---:|---:|
| address feed | 43.3% | 3.8% |
| external feed | 43.6% | 3.9% |
| licence feed | 31.9% | 3.8% |

An 11x enrichment over chance, so the same fake name recurs across feeds. Two further
checks establish that it is a *per-candidate* alias rather than a shared pool: 4,906 of
5,278 distinct aliases point at exactly one vehicle, and address rows carrying an alias
are unlinked by name matching 72.6% of the time against a 19.5% baseline for the feed as
a whole. The alias rows are precisely the rows name matching drops.

So the alias is learned on the title feed — where `vehicle_ref` reveals which candidate
the odd row belongs to — then used to claim matching rows in the feeds that have no key
of their own. Aliases pointing at more than one candidate are discarded. 8,067 aliases
are recovered, covering two thirds of the candidates.

**Pass 3 — confidence-weighted name matching** for everything else. The matcher
normalises the `SYN…-` prefixes, blocks on a SymSpell-style delete neighbourhood (edit
distance <= 2, covering substitution, insertion, deletion and transposition) plus an
explicit prefix index (covering truncation to an initial), and scores first and last name
jointly. Date-of-birth agreement adds a bonus on the licence feed, never a penalty.
Matches are softened into weights summing to 1 across competing candidates, so an
ambiguous name splits its evidence rather than committing to a guess.

Rows that no pass claims are left unclaimed. An earlier version forced them on, using the
4-rows-per-candidate structure as a capacity constraint; coverage rose to ~99% but the
address, licence and external signals all got *weaker*, because an unrelated name carries
an unrelated state.

| feed | linked | of | via alias | note |
|---|---:|---:|---:|---|
| address | 41,567 | 48,121 | 2,827 | |
| licence | 40,855 | 48,124 | 2,129 | name + DOB |
| external | 41,508 | 48,116 | 2,877 | |
| work | 24,131 | 24,131 | 0 | this feed is not corrupted |
| title | 47,581 | 48,108 | — | via `vehicle_ref` |
| T1 updates | 21,949 | 24,000 | 1,856 | 4,713 via `vehicle_ref` |

---

## 3. Signal, and what carries none

Spearman rho of a recency-weighted DE share against the ordered T0 label:

| feed | rho |
|---|---:|
| **current address only** (blank end date) | **+0.41** |
| title | +0.37 |
| address, all rows | +0.32 |
| external | +0.29 |
| licence | +0.15 |
| **work location** | **-0.07** |
| four feeds pooled | **+0.53** |

Two rows in that table decided features.

The work-location feed is complete, uncorrupted, and unrelated to the labels. It is
excluded from the pooled score and kept only as its own near-zero feature block, so the
model can confirm that rather than being forced to assume it.

The address record with a blank `effective_end_date` is where the person lives *now*, and
that **single record out-predicts the entire address feed** (+0.41 against +0.32). It
therefore gets its own feature block rather than being averaged into the history.

Recency dominates. Aggregations compared on the same resolved evidence
(`model_selection.py`, section C):

| aggregation | rho |
|---|---:|
| exponential decay, 150-day half-life | **+0.52** |
| rank decay (0.5 per step back) | +0.51 |
| newest record per feed, averaged | +0.48 |
| hard window, last 180 days | +0.46 |
| change-point, post-segment mean | +0.41 |
| linear-in-time extrapolation | +0.39 |

The half-life sits on a broad plateau between 90 and 240 days; 150 days is used. That a
change-point fit does *worse* says the evidence behaves like a recency-weighted mixture
rather than a clean move date.

The record-type fields (`event_type`, `signal_type`, `source_type`) are almost entirely
recency in disguise — on titles, `title_record` averages 1,729 days old,
`ownership_change` 825, `record_update` 287 — so the date weighting already captures
them. What they do add is that some categories carry two rows per candidate instead of
one, a less noisy read of the same feed; those three categories are kept as features.

The observed tag state contributes independently and more weakly (rho +0.11): an
out-of-state tag raises the score a little at any level of residency evidence. There is
no interaction worth modelling.

---

## 4. Model

A **proportional-odds (ordinal) logistic model** over

```
review_not_warranted (0)  <  insufficient_evidence (1)  <  review_warranted (2)
P(y <= k) = sigmoid(theta_k - x.beta),  theta_0 < theta_1
```

fitted by L-BFGS with an L2 penalty (C = 0.02), implemented in `model.py`.

Why ordinal rather than a 3-way classifier: it encodes the banded transition structure
found in section 1, and it estimates one direction vector plus two thresholds instead of
three independent ones — which matters at 300 labelled candidates.

Every family was run on identical grouped folds, each with a hyperparameter sweep. The
boosters were given two extra advantages so the comparison is not a straw man: the
observed tag state was passed to CatBoost as a genuine nine-level categorical, and each
booster was also run under the same ordinal decomposition the winner uses
(`benchmark_models.py`, full table in `benchmark_output.txt`).

| model | log loss | accuracy | macro F1 | macro AUC |
|---|---:|---:|---:|---:|
| **ordinal logistic** | **0.907** | **0.554** | **0.556** | **0.741** |
| multinomial logistic | 0.924 | 0.538 | 0.532 | 0.725 |
| random forest | 0.951 | 0.527 | 0.523 | 0.703 |
| CatBoost (with categorical) | 0.964 | 0.510 | 0.508 | 0.705 |
| XGBoost | 0.983 | 0.507 | 0.504 | 0.695 |
| hist gradient boosting | 0.998 | 0.503 | 0.501 | 0.691 |
| LightGBM | 1.047 | 0.496 | 0.495 | 0.682 |
| class prior | 1.096 | 0.355 | 0.175 | 0.500 |
| LLM classifier (Claude Haiku 4.5) | 1.151 | 0.497 | — | 0.659 |

Best configuration shown per family. Three results are worth stating plainly.

**Every tree ensemble loses to plain logistic regression here**, which inverts the usual
expectation. Three reasons, and they compound: 600 training rows against 41 features is
an order of magnitude less data than boosting wants; the signal is a smooth, near-monotone
function of a single latent direction, which trees can only approximate as a staircase,
spending variance to do it; and the ordinal constraint is information a multiclass tree
cannot represent at all.

**The gap is mostly calibration, not discrimination.** CatBoost trails on accuracy by
about four points but on log loss by 0.06 — tree ensembles are poorly calibrated out of
the box, and calibration is one of the six scored criteria.

**Handing the boosters the ordinal decomposition made them worse**, not better (XGBoost
0.983 → 1.037, CatBoost 0.964 → 1.145). The decomposition trains two models where there
was one, and at this sample size the extra variance costs more than the structure gains.
The ordinal logistic gets the same structure for the price of two scalars.

**Training rows.** T0 and T1 are pooled into 600 rows with a phase indicator, so both
phases train one model. Cross-validation groups by candidate so a candidate's T0 and T1
rows never straddle a fold.

**41 features in five blocks:**

- *evidence direction* — recency-weighted DE share of the four signal feeds at
  90/150/365-day half-lives and unweighted; the state on the newest record; how many
  feeds end in DE; disagreement between feeds; per-feed direction and newest state; the
  three two-row record categories
- *current address* — direction, count, newest state and age of the open-ended address
  records
- *evidence sufficiency* — volume of linked evidence, records in the last 6 and 12
  months, age of the newest record, share of records with no state, share of
  limited-quality records, share of expired or superseded records, identity-match
  confidence, number of distinct states seen
- *T1 updates* — volume, direction, direction of the newest update, direction of the
  title and address updates, and the change against prior evidence. Encoded as signed
  deviations that are exactly zero in the T0 phase, so the block contributes nothing
  before the updates exist rather than relying on imputation.
- *observed tag* — whether the tag on the candidate record is out of state

The four largest standardised coefficients are `ttl_recupd_de` (+0.28), `t1_adr_de`
(+0.26), `lic_credupd_de` (+0.22) and `adr_open_last_de` (+0.21). Three of the four come
from blocks the first version of this model did not have.

---

## 5. Results

Grouped 5-fold cross-validation, 20 repeats, on all 600 labelled rows:

| metric | model | baseline | first version |
|---|---:|---:|---:|
| multi-class log loss | **0.906** | 1.096 (class prior) | 0.934 |
| accuracy | **0.558** | 0.355 (majority class) | 0.527 |
| macro F1 | **0.560** | 0.175 | 0.528 |
| macro AUC | **0.740** | 0.500 | 0.719 |
| priority AUC for `review_warranted` | **0.811** | 0.500 | 0.781 |

By phase — the model responds to the later evidence, as intended:

| | log loss | accuracy | macro AUC | priority AUC |
|---|---:|---:|---:|---:|
| T0 | 0.931 | 0.520 | 0.718 | 0.794 |
| T1 | **0.880** | **0.597** | **0.762** | **0.829** |

Confusion matrix (rows true, columns predicted, order not-warranted / insufficient /
warranted):

```
[[136  61  16]
 [ 61  96  51]
 [ 15  61 103]]
```

The off-diagonal mass sits where the ordinal structure says it should: confusion is
mostly with the adjacent class, and only 31 of 600 rows cross the whole scale.

**Calibration.** Predicted vs observed frequency in quintiles of predicted probability:

| quintile | p(warranted) pred / obs | p(not warranted) pred / obs |
|---|---|---|
| 1 | 0.05 / 0.01 | 0.07 / 0.11 |
| 2 | 0.13 / 0.17 | 0.17 / 0.14 |
| 3 | 0.24 / 0.22 | 0.30 / 0.33 |
| 4 | 0.41 / 0.42 | 0.47 / 0.42 |
| 5 | 0.65 / 0.67 | 0.73 / 0.78 |

No post-hoc recalibration is applied; the ordinal likelihood is already well calibrated
on held-out folds, and with 300 labelled candidates a second fitted calibration layer
would cost more in variance than it gains.

**Class prior.** A Kolmogorov-Smirnov test of the strongest feature, labelled subset vs
all 12,000 candidates, gives D = 0.047, p = 0.52 — the development set behaves like a
random sample, so its 35/35/30 class mix is used as the population prior rather than
being reweighted.

---

## 6. Is the number honest?

Every figure above comes from cross-validation on the same 300 labelled candidates that
also drove roughly forty design comparisons. That inflates them. Two checks quantify by
how much.

**Nested selection** (`audit_honest.py`). The choice of feature blocks and regularisation
is made *inside* a training split, and the chosen model is scored once on candidates it
never saw. This is the unbiased estimate for the procedure, not just for its winner.

| | log loss | accuracy | macro AUC | priority AUC |
|---|---:|---:|---:|---:|
| ordinary cross-validation | 0.906 | 0.558 | 0.740 | 0.811 |
| **nested selection** | **0.913** | **0.549** | **0.737** | **0.809** |

The selection bias is 0.008 of log loss. The inner loop chose the same configuration in
27 of 30 folds, so the choice is stable rather than a coin flip. The unbiased new figure
(0.913) still beats the first version's *optimistic* figure (0.934).

**Paired bootstrap over candidates.** Resampling the 300 candidates 4,000 times, the
current feature set beats the first version by 0.022 of log loss, 95% interval
[+0.003, +0.040], better in 98.8% of resamples.

Treat 0.913 as the number to expect on the held-out evaluation cases.

---

## 7. Why accuracy is 56%, and what the ceiling is

56% on a three-way choice invites the question, so here is the evidence rather than an
excuse. Three probes, none of which depends on our model being good (`audit_ceiling.py`).

**The features are expressive enough.** A decision tree allowed to memorise reaches 100%
training accuracy. The limit is not model capacity.

**The ground truth is only 77.7% self-consistent.** The same candidate is labelled twice,
three months apart, with two extra records arriving in between — and the official verdict
changes for 22.3% of them. An oracle that knew the T0 answer *perfectly* would therefore
score 77.7% predicting T1. **No model can be 90% accurate against a target that moves
22% of the time.** If a solution reports 90% on this data, it has leaked labels or
memorised the development set.

**Cases the model cannot tell apart get different labels.** Nearest neighbours in the
41-dimensional feature space agree on their label only 49% of the time. Caveat worth
stating: those neighbours sit at a median distance of 4.3 against 8.8 for random pairs, so
they are about twice as close as chance rather than identical — this is suggestive of high
intrinsic noise, not a clean Bayes-error proof.

**Tuning the decision rule for accuracy is not the answer either.** Re-weighting the class
probabilities purely to maximise accuracy buys +0.008, and that was measured on the same
300 cases that chose the weights, so the honest gain is smaller. It would cost calibration,
which is separately scored.

### The figures that are in the 80s and 70s

56% is the hardest possible framing: a three-way split in which one class is *defined* as
"the evidence does not decide". Asking for 90% there is asking a model to be certain about
which cases are uncertain. Other framings of the same predictions:

| question | result |
|---|---:|
| both the model and the truth commit to a decided verdict | **88.5%** (239 of 270) |
| warranted vs everything else | 77.7% |
| predictions landing within one band of the truth | 94.8% |
| predictions at the opposite end of the scale from the truth | 5.2% |
| of the cases a reviewer opens in the top 10% of the queue, share genuinely warranted | 77% |

The last row is the one that matters operationally, and it is the one to quote to somebody
deciding whether to deploy this. The 88.5% figure is only honest when the exclusion is
stated in the same breath — quoted bare, it is the same number a careless reader would
call misleading.

---

## 8. Review priority

```
review_priority = p(review_warranted) + 0.5 x p(insufficient_evidence)
```

A case that warrants review takes full priority. A case the evidence cannot decide still
needs a person to look at it — that is what `insufficient_evidence` means operationally —
so it takes half. A case that clearly does not warrant review takes almost none. The
value lies in [0, 1] by construction, and it ranks marginally better than
`p(review_warranted)` alone; the definition is chosen for what it means to a queue, not
for the third decimal place.

---

## 9. Using the output

`case_predictions.csv` is the submission file, in the template's exact schema and row
order.

`case_audit.csv` is a supporting file with the same 24,000 rows plus the evidence behind
each score, so a reviewer can see why a case is where it is without reading the code:

- `latest_{address,licence,title,external}_{state,date}` — the newest record per feed
- `de_share_recent`, `de_share_all_time` — the direction of the evidence
- `evidence_records_linked`, `days_since_newest_record` — how much there is and how fresh
- `t1_update_direction` — `toward DE` / `away from DE` / `mixed` / `no update`
- `driver_summary` — signed contribution of each feature block, largest first
- `top_drivers` — the three individual features with the largest contribution

Signs read as: **positive pushes toward `review_warranted`, negative toward
`review_not_warranted`.** Lead with `driver_summary`; individual coefficients are
ridge-shrunk across collinear direction features, so a single feature's sign should not
be read on its own, whereas the block sums are stable.

The output supports staff review. It is not a residency, fee, or enforcement
determination, and `review_warranted` means "a person should look at this", nothing more.

---

## 10. What was tried and rejected

Each was judged on the same grouped cross-validation as the shipped model. Recording the
failures is part of the answer: it is evidence the model sits on a plateau rather than at
the first thing that worked.

| idea | result |
|---|---|
| per-feed 90- and 365-day half-lives | log loss +0.001, no gain |
| splitting T1 updates by `record_action` | +0.000, noise |
| non-linear terms on the direction index | +0.002, worse |
| a "T1 title changed state" flag | -0.002, inside the noise |
| observed tag x direction interaction | +0.002, worse; the tag effect is additive |
| explicit old-versus-new trend per feed | +0.006, worse alone and in combination |
| self-training on the ~11,700 unlabelled cases | worse at **every** confidence threshold (0.55/0.65/0.75) and pseudo-label weight (0.02/0.05/0.15), from +0.002 to +0.165 log loss. Pseudo-labels sharpen the model's existing beliefs and wreck the calibration the scoring rewards. |
| forcing unclaimed rows on via the capacity constraint | coverage rose to ~99% but address, licence and external signal all got weaker |
| recovering the replaced address row from the gap it leaves in the timeline | not possible: consecutive address records chain end-to-start in 3 of 8,322 cases, and gaps run from -136 to +413 days. The records do not tile time. |

---

## 11. Reproducibility

`predict.py` is deterministic. Run from a clean directory with the data at an arbitrary
path (`OOS_DATA_DIR`), it reproduces `case_predictions.csv` byte for byte, and
`validate.py` passes all eleven format checks.

---

## 12. Limitations

- **300 labelled candidates.** Every metric carries roughly plus or minus 0.03 of
  sampling noise, and the selection bias on top of that is quantified in section 6. The
  half-life and feature blocks were chosen on the flat part of their curves rather than
  at the cross-validation optimum. That said, more labels would not buy much: the
  learning curve (`audit_learning_curve.py`, held-out set held at 75 candidates
  throughout) has already flattened.

  | training candidates | 40 | 70 | 105 | 145 | 190 | 225 |
  |---|---:|---:|---:|---:|---:|---:|
  | log loss | 0.976 | 0.943 | 0.928 | 0.922 | 0.916 | 0.914 |

  The marginal value of 50 extra labelled candidates falls from +0.054 log loss early on
  to +0.003 at the top of the range. The binding constraint is the noise in the evidence
  itself, not the size of the label set.
- **About 13% of evidence rows remain unattributable** after all three resolution passes.
  The alias route recovers the replaced row only where the candidate's own vehicle gave a
  clean 3 + 1 split, which is two thirds of candidates. For the rest, nothing in the
  package identifies the row.
- **The signal itself is noisy.** Even with perfect resolution a candidate has roughly
  fourteen state-bearing records split between two states, so the latent direction can
  only be estimated to limited precision.
- **About 350 vehicles carry two candidates' histories** (343 have eight title rows, six
  have nine). They are split by name cluster, but where both clusters are weak the split
  is uncertain.
- **The noise model is inferred, not documented.** It is verified three ways — vehicle
  grouping, date-of-birth grouping, and the cross-feed alias enrichment — but a fourth
  corruption mode that none of those views exposes would not have been caught.
