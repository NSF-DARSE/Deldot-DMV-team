# T1 Model Breakpoint v1

- Selected model: `catboost`
- Selection reason: CatBoost improved calibrated log loss by 0.016976.
- Labeled candidates: 300
- Predictors: 25 (3 T0 probabilities + 22 T1 update/delta features)
- Outer folds: reused from T0 so outer-validation priors are OOF for the same candidates
- Inner tuning/calibration folds: 3
- Final deployment temperature: 1.172037
- `priority_t0` is retained in the update matrix but excluded from modeling because it duplicates `p_review_warranted_t0`.

## Nested OOF selection metrics

| Model | Log loss | Macro-F1 | Accuracy | Multiclass Brier | Macro OVR Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|---:|
| catboost | 1.041262 | 0.448638 | 0.450000 | 0.630585 | 0.210195 | 0.044436 |
| logistic | 1.058238 | 0.466881 | 0.466667 | 0.639993 | 0.213331 | 0.073728 |

## Final-model feature importance

- 1. `p_review_warranted_t0`: 25.262420
- 2. `p_review_not_warranted_t0`: 14.366211
- 3. `delta_days_since_latest_oos`: 8.191407
- 4. `delta_oos_decay_90d`: 6.032477
- 5. `delta_de_decay_90d`: 5.775320
- 6. `delta_days_since_latest_de`: 5.377723
- 7. `delta_active_address_oos_count`: 5.252479
- 8. `delta_t1_update_observation_lag_mean_days`: 4.839822
- 9. `delta_link_confidence_mean`: 4.015930
- 10. `delta_active_address_de_count`: 3.878515

## Generalization and stacking boundary

The validation candidate's T0 prior is always generated without that candidate because T1 reuses the T0 outer folds. All labeled training priors are candidate-wise OOF. This is a practical cross-fitted stack for 300 labels; synthetic validation still cannot establish production DMV performance.

## Breakpoint

T1 predictions are complete. The standalone final metrics and update-behavior analysis stage has not started.
