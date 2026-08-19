# T0 Model Breakpoint v1

- Selected model: `catboost`
- Selection reason: CatBoost improved calibrated log loss by 0.022398.
- Labeled candidates: 300
- Compact predictors: 24
- Validation: 5-fold outer CV with 3-fold tuning/calibration inside each outer training fold
- Final deployment temperature: 0.869815, fit on nested OOF raw probabilities
- Review priority: `p_review_warranted`

## Nested OOF metrics

| Model | Log loss | Macro-F1 | Accuracy | Multiclass Brier | Macro OVR Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|---:|
| catboost | 1.034132 | 0.448905 | 0.450000 | 0.628625 | 0.209542 | 0.060426 |
| logistic | 1.056531 | 0.462161 | 0.463333 | 0.639581 | 0.213194 | 0.058929 |

## Final-model feature importance

- 1. `active_address_de_count`: 14.652688
- 2. `active_address_oos_count`: 11.000350
- 3. `oos_decay_share_365d`: 7.844481
- 4. `latest_oos_recency_advantage_days`: 7.617932
- 5. `de_count_90d`: 7.123703
- 6. `days_since_latest_oos`: 6.711273
- 7. `linked_record_count`: 6.144014
- 8. `oos_share_known_state`: 6.065306
- 9. `state_missing_ratio`: 5.914073
- 10. `recent_latest_source_de_count_365d`: 4.522247

Accuracy is reported but did not control model selection. Log loss is primary because calibrated probabilities and safe uncertainty handling matter operationally.

## Leakage safeguards

- Candidate IDs are unique, so no candidate crosses an outer train/validation boundary.
- Hyperparameter choice and early stopping occur only inside each outer training partition.
- Each outer-fold temperature is fit on inner OOF probabilities, never on in-fold predictions.
- The 300 labeled T0 priors supplied to the later T1 training stage are outer-OOF probabilities, not predictions from a model trained on those candidates.

## Generalization boundary

Synthetic validation cannot prove real-DMV performance. The portable safeguards are frozen linkage, stable feature definitions, small model capacity, nested validation, calibrated uncertainty, and audit-only treatment of sparse legal-window signals. Real deployment still requires schema checks plus retraining and calibration on representative real labeled cases.

## Breakpoint

T0 is complete. T1 training has not started.
