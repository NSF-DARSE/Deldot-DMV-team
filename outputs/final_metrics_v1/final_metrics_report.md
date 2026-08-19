# Final Metrics and Release-Readiness Report v1

- Status: `synthetic_prototype_ready_real_data_validation_required`
- T0/T1 production predictions retained without experimental anchoring.
- Feature drift warnings: 0 of 49 checked predictors.

## Leakage-safe OOF metrics

| Phase | Log loss | Macro-F1 | Accuracy | Multiclass Brier | ECE | AP warranted | Precision@10% | Recall@20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 1.034132 | 0.448905 | 0.450000 | 0.628625 | 0.060426 | 0.459584 | 0.500000 | 0.333333 |
| T1 | 1.041262 | 0.448638 | 0.450000 | 0.630585 | 0.044436 | 0.482781 | 0.600000 | 0.348315 |

Bootstrap confidence intervals are provided separately. They should be emphasized over point differences because only 300 labeled candidates are available.

## T0→T1 transition behavior on labeled OOF candidates

- Actual label changes: 67 (22.33%)
- Predicted class changes: 134 (44.67%)
- Change-detection precision: 0.2388
- Change-detection recall: 0.4776
- Change-detection F1: 0.3184

The model changes class more often than labels do. The experimental anchor improved point metrics but failed the clear-improvement and minimum-support gates, so the current T1 model remains unchanged and update stability is a release warning.

## Real-data readiness

The pipeline is suitable for shadow validation on real DMV data, not direct production enforcement. The feature contract records required columns, types, missingness, and reference ranges. Real labeled cases are still required for retraining, calibration, linkage precision review, update-stability approval, and operational threshold selection.

No demographic attributes are supplied, so demographic fairness cannot be evaluated from this package.

The canonical `case_predictions.csv` follows the exact submission-template row and column order.
