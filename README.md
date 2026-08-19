# Delaware DMV Out-of-State Evidence Review Pipeline

This repository contains the final reproducible prototype for linking synthetic Delaware DMV evidence, preparing time-aware features, producing calibrated T0/T1 review predictions, and evaluating update behavior.

The system is decision support only. It does not determine residency, registration obligations, exemptions, violations, guilt, fees, or enforcement action.

## Final pipeline

```text
Conservative linkage + freeze
        ↓
Temporal feature preparation
        ↓
Compact feature contract
        ↓
Calibrated CatBoost T0 prediction
        ↓
Calibrated CatBoost T1 update prediction
        ↓
OOF metrics, calibration, priority, and update-stability analysis
```

The canonical submission is `outputs/final_metrics_v1/case_predictions.csv`. It contains 24,000 rows: one T0 and one T1 prediction for each of 12,000 candidates.

## Linkage v1.1

Identity linkage is deliberately conservative and portable:

- strong DOB/name and address/name anchors establish verified aliases;
- unique exact or high-margin fuzzy names may link when unambiguous;
- ambiguous and contradictory records remain unresolved;
- row order, class labels, DE/OOS state, and event dates are prohibited identity inputs;
- names, DOB, and addresses are linkage-only and are not model predictors.

T0 vehicle titles include a narrow `vehicle_ref` bridge. It propagates identity only when all high-confidence anchors agree, no independently linked owner conflicts, and both owner-name components pass strict similarity and runner-up safeguards.

- T0 title linkage: 32,707 → 32,836 rows
- Coverage: 67.9866% → 68.2548%
- Recovered rows: 129
- Internal leave-one-alias-out audit: 2,318/2,318 agreements

The audit is internal consistency evidence, not authoritative linkage truth. Real deployment requires a manually adjudicated linkage sample.

## Modeling and validation

- 300 labeled candidates
- five candidate-level outer CV folds
- tuning and early stopping inside training folds
- scalar temperature calibration from out-of-fold predictions
- CatBoost selected against a multinomial logistic-regression challenger
- T1 reuses the exact T0 outer folds and leakage-safe T0 priors
- experimental T1 anchoring was evaluated but not applied

Current leakage-safe OOF point metrics:

| Phase | Log loss | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|
| T0 | 1.034132 | 0.448905 | 0.628625 | 0.060426 |
| T1 | 1.041262 | 0.448638 | 0.630585 | 0.044436 |

Only 300 synthetic labels are available. These metrics demonstrate the workflow, not expected real DMV performance.

## Repository structure

- `Identify_Out_of_State_Tag_Holders/` — mentor-provided synthetic input package
- `configs/` — frozen linkage, feature, model, update, and metric definitions
- `linkage_v1/` — normalization, similarity, and resolver implementation
- `feature_prep_v1/` — temporal feature builder
- `modeling_v1/` — shared calibration and model utilities
- `scripts/` — reproducible pipeline entry points
- `tests/` — focused linkage, feature, model, update, and final-output tests
- `outputs/` — reports, feature matrices, models, OOF predictions, and final predictions
- `baseline_snapshot/` — compact pre-bridge artifacts used for paired comparison
- `DMV_PIPELINE_HANDOFF.md` — detailed reproduction and handoff notes

## Reproduce

Python 3.11+ is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/run_linkage_v1.py
.venv/bin/python scripts/build_temporal_features_v1.py
.venv/bin/python scripts/select_compact_features_v1.py
.venv/bin/python scripts/train_t0_model_v1.py
.venv/bin/python scripts/train_t1_model_v1.py
.venv/bin/python scripts/analyze_update_behavior_v1.py
.venv/bin/python scripts/generate_final_metrics_v1.py
.venv/bin/python scripts/compare_vehicle_bridge_v1.py
```

Validate:

```bash
.venv/bin/python -m pytest -q \
  tests/test_linkage_v1.py \
  tests/test_temporal_features_v1.py \
  tests/test_compact_features_v1.py \
  tests/test_t0_model_v1.py \
  tests/test_t1_model_v1.py \
  tests/test_update_behavior_v1.py \
  tests/test_final_metrics_v1.py

.venv/bin/python scripts/validate_submission.py \
  outputs/final_metrics_v1/case_predictions.csv
```

The validated release has 27 passing focused tests.

## Real-data readiness boundary

Before operational use, retrain and recalibrate on authoritative, representative data; manually audit linkage precision and missed links; validate temporal assumptions and exemptions; select review thresholds with DMV stakeholders; and perform subgroup fairness and error analysis. The model must remain human-reviewed decision support.
