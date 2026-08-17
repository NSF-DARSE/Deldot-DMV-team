# Prepared answers for the submission form

Copy-paste text for the fields a submission form usually asks for. Numbers match
`model_card.json` and are reproducible via `diagnostics.py` / `model_selection.py`.

## One line

An ordinal review-risk model over entity-resolved evidence, scoring all 12,000 cases at
both phases: CV log loss 0.934 against a 1.096 prior baseline, macro AUC 0.719.

## Short (about 50 words)

```
The package gives no key linking cases to the evidence feeds, and the names are
deliberately corrupted, so the work went into identity resolution. Titles are linked by
vehicle_ref, which recovers rows whose owner name was replaced outright. A
proportional-odds ordinal model then scores each case: CV log loss 0.934 vs a 1.096
baseline, macro AUC 0.719, calibrated.
```

## Medium (about 150 words)

```
Three findings shaped the solution. First, each feed holds one record set per candidate,
and in each set one row in four has its name replaced outright, which we verified two
independent ways (grouping titles by vehicle_ref, and licences by date of birth). The
title feed can recover that row because vehicle_ref identifies the vehicle regardless of
the name; the other feeds cannot, and forcing an assignment there made the signal worse.
Second, the three classes are ordered: no development case ever moves from
review_not_warranted to review_warranted without passing through insufficient_evidence.
We therefore used a proportional-odds ordinal model, which beats multinomial logistic,
random forest and gradient boosting on identical folds. Third, the direction is recency,
with a 150-day half-life, and the work-location feed carries no signal at all.

Grouped cross-validation over 600 labelled rows: log loss 0.934 against a 1.096 prior
baseline, accuracy 0.527, macro AUC 0.719, priority ranking AUC 0.781. T1 scores better
than T0, so the model responds to the later evidence. Probabilities are calibrated with
no post-hoc fitting.
```

## Approach / methodology field (about 300 words)

```
Identity resolution was the hard part, not classification. Nothing in the package links
candidate_records.csv to the six evidence feeds except names, and the names are corrupted
by truncation, substitution, transposition and stray whitespace. Using vehicle_ref on the
title feed as an independent check, we found the corruption pattern: each record set is
three lightly corrupted rows plus one row whose name is replaced outright. Grouping the
licence feed by date of birth reproduces the same three-plus-one split, confirming it is
the generator's noise model rather than a quirk of one table.

That has a direct consequence. The title feed can recover its fourth row, because
vehicle_ref says which vehicle it belongs to; doing so takes it from 38,582 to 47,581
linked rows and lifts its correlation with the label from +0.25 to +0.37. The remaining
feeds have no such key. We tested forcing an assignment there using the four-rows-per-
candidate structure as a capacity constraint, and the address, licence and external
signals all got weaker, because an unrelated name carries an unrelated state. Accepting
about 80 percent recall on those feeds is the better trade.

The classes turned out to be ordered rather than unordered: across the 300 development
cases, no case moves between the two decided classes without passing through
insufficient_evidence. We modelled that directly with a proportional-odds ordinal
logistic, which estimates one direction vector and two thresholds instead of three
independent classifiers, and which beats multinomial logistic, random forest and
gradient boosting on identical folds. Features cover evidence direction, evidence
sufficiency, the T1 update stream and the observed tag state.

Eight further ideas were tested and rejected, including semi-supervised self-training,
which was worse at every setting because pseudo-labels sharpen the model's existing
beliefs and damage calibration. A per-case audit file accompanies the predictions so a
reviewer can check any case without reading the code.
```

## Limitations field

```
Only 300 labelled candidates, so every metric carries roughly plus or minus 0.03 of
sampling noise; the feature set and half-life were chosen on the flat part of their curves
rather than at the cross-validation optimum, to avoid selecting on noise. One record in
four in the address, licence and external feeds is unattributable and nothing in the
package recovers it. Even with perfect resolution each candidate has only about fourteen
state-bearing records split between two states, so macro AUC near 0.72 is likely close to
what this data supports.
```

## Fields we cannot fill without you

- Team name, and your friend's full name and email
- Any entrant ID the organisers issued
