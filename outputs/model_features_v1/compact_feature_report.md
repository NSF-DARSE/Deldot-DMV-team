# Compact Model Feature Selection v1

- Candidates: 12,000
- T0 predictors: 24 (reduced from 148 model-candidate columns)
- T1 update predictors: 22 (reduced from 54)
- T0 audit-only review signals: 3
- T1 audit-only review signals: 3
- Labels used to choose this manifest: no
- PII or raw identity fields included: no

## Leakage policy

The compact manifest is fixed from domain and data-definition reasoning before reading labels. Any later supervised feature selection, constant removal, imputation, hyperparameter tuning, early stopping, and probability calibration must be fit using training-fold data only.
Development candidate IDs, but not label values, were used for an unsupervised sparsity check. Extremely sparse legal-window interactions remain available as audit-only review signals and are excluded from CatBoost inputs.

## T1 prior policy

The T1 model matrix will append calibrated out-of-fold T0 class probabilities and T0 review priority during cross-validation. In-sample T0 predictions must never be used as T1 priors.

## Runtime note

This selection reads the frozen feature matrices directly and does not rerun linkage or per-candidate event aggregation.
