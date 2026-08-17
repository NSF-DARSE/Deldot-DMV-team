# Out-of-State Tag Holder Challenge — submission

Decision-support scorer for the 12,000 challenge cases: a review classification, three
class probabilities, and a review-priority value, at T0 and T1.

**Start with [SUMMARY.md](SUMMARY.md)** — one page.
**[METHOD.md](METHOD.md)** is the full write-up: what the data turned out to be, how
identities are resolved, why the model is ordinal, the validation results, and the eight
ideas tested and rejected, and whether the headline number is honest.

## Where to start

| you want | open |
|---|---|
| every number in one place | `RESULTS.md` |
| the one-page story | `SUMMARY.md` (or `dist/OOS_Tag_Challenge_Summary.pdf`) |
| the full method | `METHOD.md` (or `dist/OOS_Tag_Challenge_Method.pdf`) |
| the live demo | `dist/console.html` — double-click it, no install needed |
| text for a submission form | `FORM_ANSWERS.md` |
| putting this on GitHub | `REPO_SETUP.md` |

## Submission file

`case_predictions.csv` — 24,000 rows in the template's exact schema and row order. Every
row's three probabilities sum to exactly 1 in decimal; `review_priority` is in [0, 1].

## Supporting files

| file | what it is |
|---|---|
| `dist/console.html` | the reviewer console: priority queue over all 12,000 cases, per-case evidence, and the real model running live in the page (self-contained, opens from a file) |
| `case_audit.csv` | the same 24,000 rows plus the evidence behind each score: newest record per feed with its date, evidence volume and freshness, direction of the T1 update, and the signed contribution of each feature block. Intended for the staff member working the queue. |
| `model_card.json` | fitted thresholds and standardised coefficients, plus the cross-validated metrics |
| `diagnostics_output.txt` | output of `diagnostics.py` — the evidence for every structural claim in METHOD.md |
| `model_selection_output.txt` | output of `model_selection.py` — model family, feature set, and aggregation comparisons |
| `FORM_ANSWERS.md` | prepared copy-paste text for the submission form (approach, limitations, at three lengths) |
| `dist/*.pdf` | the summary and full write-up as PDFs; rebuild with `./build_pdfs.sh` |

## Running it

The challenge data is not in this repo — it is the organisers' to distribute. Point the
code at your own copy.

```bash
pip install -r requirements.txt

# the challenge package must be findable: put the Identify_Out_of_State_Tag_Holders
# folder next to this code, or point OOS_DATA_DIR at it
export OOS_DATA_DIR=/path/to/Identify_Out_of_State_Tag_Holders

python predict.py       # ~35 s: resolve -> featurise -> cross-validate -> score -> write
                        # deterministic: reproduces case_predictions.csv byte for byte
python validate.py      # schema, row order, probability sums, ranges
python diagnostics.py   # reproduces the structural findings
python model_selection.py
```

## Code

| module | responsibility |
|---|---|
| `er.py` | name normalisation, edit distance, the fuzzy-lookup index |
| `alias.py` | second-identity recovery: learns each candidate's replaced name from `vehicle_ref` |
| `resolve.py` | entity resolution — name matching, and vehicle-keyed title grouping |
| `pipeline.py` | data loading; builds the tidy evidence table |
| `featurize.py` | per-candidate, per-phase features |
| `model.py` | proportional-odds ordinal logistic model |
| `predict.py` | end-to-end run; writes the submission, audit file and model card |
| `validate.py` | submission-format checks |
| `diagnostics.py`, `model_selection.py` | reproduce the claims in METHOD.md |
| `experiments.py` | ideas tested and rejected (METHOD.md section 9) |
| `export_console.py`, `console_template.html`, `build_console.sh` | build the reviewer console |
| `llm_baseline.py` | measures an LLM classifier on the same task, for the "why not AI?" question |
| `audit_unused.py`, `audit_alias.py`, `audit_features.py` | the process audit: fields never used, the alias hypothesis, the features it surfaced |
| `audit_honest.py` | nested selection — the unbiased estimate (METHOD.md section 6) |
| `audit_learning_curve.py` | what another labelled candidate is worth (METHOD.md section 11) |
| `extra_features.py` | feature builder shared by the audit scripts |

## Headline numbers

Grouped 5-fold cross-validation (20 repeats) on the 300 development candidates, scored at
both phases (600 rows):

| | model | baseline |
|---|---:|---:|
| log loss | 0.906 | 1.096 |
| accuracy | 0.558 | 0.355 |
| macro F1 | 0.560 | 0.175 |
| macro AUC | 0.740 | 0.500 |
| priority AUC for `review_warranted` | 0.811 | 0.500 |

T1 scores better than T0 (log loss 0.880 vs 0.931), i.e. the model does move on the later
evidence rather than ignoring it.

Those figures share the 300 labels with the design search. **Nested selection puts the
unbiased figure at 0.913 log loss** (METHOD.md section 6) — treat that as the expectation
on held-out cases.

The output supports staff review. It is not a residency, fee, or enforcement
determination.
