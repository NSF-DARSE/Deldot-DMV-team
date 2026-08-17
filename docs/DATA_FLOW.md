# Data flow and matching logic

This is the living design note for the solution. Update it when a stage
is added or a matching rule changes.

## Goal

Produce `case_predictions.csv` with one T0 row and one T1 row per candidate.

```text
candidate_records.csv
        |
        v
   PersonIndex (exact family, optional DOB)
        |
        +--> linked T0 sources + T1 stream     [stage 1]
        |    outputs/linked/
        v
   case feature table (T0 and T1)              [stage 2]
        |    outputs/features/case_features.csv
        v
   transparent recency-vote rule               [stage 2]
        |    outputs/baseline/case_predictions.csv
        v
   grouped nested-CV HGB model                 [stage 3]
        |    outputs/model/case_predictions.csv
        |    case_predictions.csv
        v
   submit case_predictions.csv
```

## Stage 1 — linkage

Source files have no `candidate_record_id`. Joining on raw name strings
fails because of mixed case and truncated given names. Quietly prefix-matching
family names would also fail: `ALCV` and `ALCVD` are different people.

The linker writes `match_rule` and `match_score` onto every row so a reviewer
can see *why* a record was attached.

### Matching rules

Family name is always an exact match after uppercasing and stripping
`SYNFAM-`. Given names may be truncated. Date of birth, when present, is a
hard filter.

| Rule | Given name | Family | DOB | Typical score |
|---|---|---|---|---|
| `identity` | exact | exact | exact | 1.00 |
| `dob_prefix` | prefix, overlap ≥ 3 | exact | exact | 0.93 |
| `dob_initial` | prefix, overlap 1–2 | exact | exact | 0.88 |
| `name_exact` | exact | exact | source has none | 0.86 |
| `name_prefix` | prefix, overlap ≥ 3 | exact | source has none | 0.72 |
| `vehicle_ref` | compatible name on a uniquely owned vehicle | | | 0.80 |
| `unlinked` | no unique candidate | | | 0 |
| `ambiguous_unassigned` | two or more survivors | | | 0 |

Rejected on purpose:

- Family-name prefixes (`ALCV` ↛ `ALCVD`)
- DOB mismatch, even if the names are exact
- 1–2 character given names when the source has no DOB

`vehicle_ref` only fires when the vehicle is uniquely tied to one candidate
by a name match, the leftover family name matches that owner, and the leftover
given name is the same person (exact or truncated).

## Stage 2 — features and rule baseline

Row counts and overall DE-share do **not** separate the labels. Recency does:

- `review_warranted` — Delaware address / license / title facts are newer
- `review_not_warranted` — out-of-state facts are newer
- `insufficient_evidence` — mixed or thin file

### Recency vote

For each source, compare the latest DE date to the latest non-DE date:

- DE newer → `+1`
- OOS newer → `-1`
- missing or exact timestamp tie → `0`

T1 appends `evidence_update_stream` rows into address, license, title, and
external. Work has no T1 domain. If a case has any T1 address row, that state
replaces the T0 open address as current.

### Score

```text
de_oos_score =
    2.0 * address_vote
  + 2.0 * title_vote
  + 1.5 * license_vote
  + 1.0 * work_vote
  + 0.5 * external_vote
  + 1.5 if current address is DE
  - 1.5 if current address is OOS
```

Weights follow how directly the source speaks to residency or the vehicle,
not a fit on the 300 labels.

### Class rule

| Condition | Class |
|---|---|
| fewer than 2 sources present, or no recency votes | `insufficient_evidence` |
| score ≥ 2.0 | `review_warranted` |
| score ≤ -2.0 | `review_not_warranted` |
| otherwise | `insufficient_evidence` |

2.0 is one strong DE-newer source (title or address), or two weaker aligned
signals. A current DE address alone (+1.5) is not enough.

Probabilities are a softmax over distance from score centers +2.5 / 0 / -2.5.
Thin files get an extra push toward insufficient. Priority is
`0.75 * p_warranted + 0.25 * p_insufficient`.

Every prediction carries `rule_reason`, e.g.
`title DE-newer (+2.0); current address DE (+1.5); score=+3.5; decision=review_warranted`.

### Observed on this package (development labels)

| Phase | Accuracy | Macro-F1 | Mean score by true class (W / I / N) |
|---|---:|---:|---|
| T0 | 0.47 | 0.47 | +2.21 / -0.07 / -2.27 |
| T1 | 0.45 | 0.45 | +2.26 / +0.02 / -1.45 |

Random is ~0.33. The score axis is in the right order. About 31% of cases
change class from T0 to T1 under the rule.

## Stage 3 — grouped nested-CV model

A shallow histogram-gradient booster is trained on the feature table,
including `de_oos_score`. It is allowed to use interactions the additive
score cannot (for example a DE-newer title with an out-of-state observation).

Validation is grouped by `candidate_record_id` so a person's T0 row cannot
train a model that is then tested on that same person's T1 row. Outer 5-fold
GroupKFold is the reported number. Inner 3-fold GroupKFold selects depth,
learning rate, leaf size, and L2. After that estimate is recorded, one model
is fit on all 300 people and applied to all 12,000 cases.

The rule is kept on every row (`rule_predicted_class`, `rule_reason`) so a
reviewer can see whether the model agreed.

### Observed nested-CV on this package

| Estimator | Accuracy | Macro-F1 | Log-loss |
|---|---:|---:|---:|
| Recency-vote rule | 0.46 | 0.46 | 1.34 |
| HGB (outer-fold mean) | 0.47 | 0.47 | 1.20 |

The lift is real and small, which is what 300 labels should produce. T0 OOF
accuracy is 0.50; T1 is 0.45. About 29% of cases change class from T0 to T1.
The model agrees with the rule on 61% of all 24,000 rows.

A logistic regression is fit only as an explanation table
(`outputs/model/logistic_coefficients.csv`). Sparse state dummies are noisy
there; `open_address_is_de` and `de_oos_score` have the expected signs.

Submit `case_predictions.csv` at the package root (also copied under
`outputs/model/`).

## Code map

| Path | Role |
|---|---|
| `oos_review/names.py` | Parse and compare synthetic tokens |
| `oos_review/linker.py` | `PersonIndex`, `link_frame`, T0/T1 wrappers |
| `oos_review/features.py` | Recency votes, current snapshot, `de_oos_score` |
| `oos_review/baseline.py` | Rule, probabilities, priority, `rule_reason` |
| `oos_review/evaluate.py` | Development-set accuracy / F1 / confusion |
| `oos_review/model.py` | Grouped nested CV, HGB fit, submission mapping |
| `oos_review/pipeline.py` | `run_linkage`, `run_features_and_baseline`, `run_model` |
| `notebooks/01_labeled_case_explorer.ipynb` | Linkage dossiers |
| `notebooks/02_features_and_baseline.ipynb` | Score, confusion, example reasons |
| `notebooks/03_model_cv.ipynb` | Nested CV vs rule, disagreements |

## How to run

```bash
python -m pytest tests
```

```python
from oos_review.pipeline import run_pipeline
bundle, features, baseline, preds, cv = run_pipeline(save=True)
```

If linkage and features already exist:

```python
from oos_review.pipeline import run_model
preds, cv = run_model(save=True)
```

## What this package does not do

The model is a staff ranking aid. It is not a residency, fee, or enforcement
determination. With 300 labels, nested-CV accuracy near 0.47 is the honest
number; in-sample accuracy will look better and should not be quoted.
