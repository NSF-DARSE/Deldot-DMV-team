# Project status

## Completed

- Profiled every supplied file and documented joins, missingness, categories, dates, identifiers, and label distributions.
- Built reproducible T0/T1 evidence aggregation with 202 explainable features per phase.
- Evaluated Logistic Regression, Random Forest, Histogram Gradient Boosting, and sigmoid-calibrated variants.
- Selected sigmoid-calibrated Logistic Regression and documented per-class metrics, confusion matrices, and calibration results.
- Generated and strictly validated all 24,000 official prediction rows.
- Tested four review-priority formulas and documented the selected transparent composite.
- Added FastAPI endpoints, deterministic explanations, linear feature associations, evidence timeline, T0/T1 changes, and SQLite audit workflow.
- Added a responsive React reviewer queue and case-detail workspace with demo-only human actions.
- Passed 7 Python tests and 2 server-rendered frontend tests; the production frontend dependency audit reports 0 vulnerabilities.
- Published the owner-only dashboard at `https://delaware-dmv-review-support.icseven.chatgpt.site` (preview fallback data is used when the local API is unavailable).

## Key discoveries

- Evidence tables omit candidate IDs but use a strongly validated repeated candidate-block layout with small unrelated T0 tails.
- The 300-label development sample is balanced across classes but small; calibrated probability quality is more reliable than raw model confidence.
- Later evidence contains exactly two records per candidate with explicit action/domain fields and no target T0 record identifier.

## Current model metrics

- Mean multiclass log loss: 1.092
- Mean multiclass Brier score: 0.607
- Mean macro F1: 0.512
- Mean 10-bin ECE: 0.094

## Unresolved risks

- Performance estimates have high variance because only 300 labeled candidates are available.
- The validated ordered package layout should be replaced by explicit source-to-candidate keys in a production data contract.
- The prototype uses synthetic data and has not been tested for real-world drift, subgroup performance, accessibility conformance, or DMV security controls.

## Next operational step

Run a DMV-led validation using a larger temporally held-out labeled sample, review false-positive/false-negative cases with staff, and establish deployment/security/data-retention requirements before any real-record pilot.
