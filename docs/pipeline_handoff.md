# Delaware DMV OOS Review Pipeline — Handoff

This archive contains the complete, reproducible final prototype built from the mentor-provided synthetic data. The pipeline links the six source files, prepares time-aware T0/T1 features, trains calibrated CatBoost classifiers, evaluates update behavior, and produces the final case-level output.

The final linkage freeze is `linkage-v1.1.0`. It adds a conservative T0 `vehicle_ref` bridge only when strong name-linked anchors unanimously agree, no independently linked owner conflicts, and the unresolved owner name passes strict similarity and runner-up safeguards. The bridge does not use row order, labels, state, or dates to assign identity.

## Final deliverable

The canonical submission is:

`outputs/final_metrics_v1/case_predictions.csv`

The final reports are:

- `outputs/final_metrics_v1/final_metrics_report.md`
- `outputs/final_metrics_v1/update_behavior_report.md`
- `outputs/final_metrics_v1/final_metrics_v1.json`
- `outputs/vehicle_ref_bridge_comparison_v1/vehicle_bridge_comparison_report.md`

## Pipeline order

1. Linkage and linkage-rule freeze
2. Temporal feature preparation
3. Compact model-feature selection
4. T0 model training and prediction
5. T1 model training and prediction
6. Update-behavior analysis and final metrics

## Reproduce from the archive root

Create a Python environment and install the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the stages in order:

```bash
.venv/bin/python scripts/run_linkage_v1.py
.venv/bin/python scripts/build_temporal_features_v1.py
.venv/bin/python scripts/select_compact_features_v1.py
.venv/bin/python scripts/train_t0_model_v1.py
.venv/bin/python scripts/train_t1_model_v1.py
.venv/bin/python scripts/analyze_update_behavior_v1.py
.venv/bin/python scripts/generate_final_metrics_v1.py
.venv/bin/python scripts/compare_vehicle_bridge_v1.py
```

Validate the final submission and run the focused tests:

```bash
.venv/bin/python scripts/validate_submission.py outputs/final_metrics_v1/case_predictions.csv
.venv/bin/python -m pytest -q \
  tests/test_linkage_v1.py \
  tests/test_temporal_features_v1.py \
  tests/test_compact_features_v1.py \
  tests/test_t0_model_v1.py \
  tests/test_t1_model_v1.py \
  tests/test_update_behavior_v1.py \
  tests/test_final_metrics_v1.py
```

## Modeling decisions

- Linkage rules and temporal-feature definitions are versioned and frozen under `configs/` and copied into their output folders.
- Candidate-level five-fold out-of-fold evaluation prevents records from the same candidate from leaking between training and validation.
- T0 and T1 use CatBoost, with probability calibration evaluated from out-of-fold predictions.
- T1 conservative anchoring was tested on the same folds. It was not applied because it did not clearly improve held-out behavior; the unanchored T1 model remains canonical.
- The T0 vehicle-reference bridge recovered 129 title rows, increasing title linkage from 67.9866% to 68.2548%. Its leave-one-alias-out internal audit accepted 2,318 reference rows with no disagreements. This is an internal consistency result, not authoritative linkage truth.
- `baseline_snapshot/` contains only the compact pre-bridge linkage and OOF artifacts required to reproduce the before/after comparison without bundling a second full pipeline.
- The temporal features treat OOS records as review signals, not proof of a violation. They account for recency, migration direction, the 60-day grace window, source agreement, and evidence quality where supported by the available columns.

## Important limitation

The supplied data is synthetic. The package demonstrates a production-oriented, leakage-controlled workflow, but its measured accuracy is not evidence of real DMV performance. Before operational use, rerun linkage validation, feature-contract checks, calibration, threshold selection, and fairness/error analysis on authoritative real data.
