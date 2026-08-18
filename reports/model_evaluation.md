# Model evaluation

All metrics are five-fold stratified out-of-fold estimates, evaluated separately at T0 and T1 so the same candidate never appears twice in a fold. Preprocessing is fitted inside each fold. Model selection emphasizes probability quality and generalization.

## Aggregate leaderboard

| model_name | mean_log_loss | mean_brier | mean_macro_f1 | mean_ece | selection_score |
| --- | --- | --- | --- | --- | --- |
| random_forest_sigmoid | 0.9250 | 0.5571 | 0.5420 | 0.0673 | 1.0161 |
| random_forest | 0.9440 | 0.5665 | 0.5384 | 0.0767 | 1.0381 |
| hist_gradient_boosting_sigmoid | 0.9679 | 0.5832 | 0.5143 | 0.0736 | 1.0662 |
| logistic_regression_sigmoid | 1.0059 | 0.6070 | 0.5119 | 0.0940 | 1.1111 |
| hist_gradient_boosting | 1.2459 | 0.6794 | 0.5246 | 0.2249 | 1.3781 |
| logistic_regression | 1.7295 | 0.7824 | 0.4959 | 0.2996 | 1.8911 |

Selected model: **random_forest_sigmoid**.

The selection score combines mean multiclass log loss, multiclass Brier score, 10-bin expected calibration error, and a small macro-F1 reward. Lower is better.

Unavailable optional baselines: XGBoost (package not installed), CatBoost (package not installed). The reproducible core uses scikit-learn models present in the runtime.

## logistic_regression

| phase | log_loss | accuracy | macro_f1 | brier | ece |
| --- | --- | --- | --- | --- | --- |
| T0 | 1.6692 | 0.5100 | 0.5109 | 0.7580 | 0.2845 |
| T1 | 1.7898 | 0.4767 | 0.4808 | 0.8067 | 0.3147 |

### T0 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5253 | 0.5778 | 0.5503 | 90.0 |
| review_not_warranted | 0.6040 | 0.5810 | 0.5922 | 105.0 |
| insufficient_evidence | 0.4000 | 0.3810 | 0.3902 | 105.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[52, 7, 31]
[15, 61, 29]
[32, 33, 40]
```

### T1 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5455 | 0.5393 | 0.5424 | 89.0 |
| review_not_warranted | 0.5200 | 0.4815 | 0.5000 | 108.0 |
| insufficient_evidence | 0.3839 | 0.4175 | 0.4000 | 103.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[48, 13, 28]
[15, 52, 41]
[25, 35, 43]
```

## logistic_regression_sigmoid

| phase | log_loss | accuracy | macro_f1 | brier | ece |
| --- | --- | --- | --- | --- | --- |
| T0 | 0.9904 | 0.5467 | 0.5429 | 0.5968 | 0.1107 |
| T1 | 1.0213 | 0.4933 | 0.4809 | 0.6172 | 0.0774 |

### T0 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5714 | 0.6222 | 0.5957 | 90.0 |
| review_not_warranted | 0.5948 | 0.6571 | 0.6244 | 105.0 |
| insufficient_evidence | 0.4535 | 0.3714 | 0.4084 | 105.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[56, 10, 24]
[13, 69, 23]
[29, 37, 39]
```

### T1 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5765 | 0.5506 | 0.5632 | 89.0 |
| review_not_warranted | 0.4800 | 0.6667 | 0.5581 | 108.0 |
| insufficient_evidence | 0.4154 | 0.2621 | 0.3214 | 103.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[49, 25, 15]
[13, 72, 23]
[23, 53, 27]
```

## random_forest

| phase | log_loss | accuracy | macro_f1 | brier | ece |
| --- | --- | --- | --- | --- | --- |
| T0 | 0.9066 | 0.5933 | 0.5895 | 0.5427 | 0.0876 |
| T1 | 0.9814 | 0.4933 | 0.4873 | 0.5903 | 0.0659 |

### T0 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.6214 | 0.7111 | 0.6632 | 90.0 |
| review_not_warranted | 0.6174 | 0.6762 | 0.6455 | 105.0 |
| insufficient_evidence | 0.5244 | 0.4095 | 0.4599 | 105.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[64, 10, 16]
[11, 71, 23]
[28, 34, 43]
```

### T1 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5686 | 0.6517 | 0.6073 | 89.0 |
| review_not_warranted | 0.5254 | 0.5741 | 0.5487 | 108.0 |
| insufficient_evidence | 0.3500 | 0.2718 | 0.3060 | 103.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[58, 13, 18]
[12, 62, 34]
[32, 43, 28]
```

## random_forest_sigmoid

| phase | log_loss | accuracy | macro_f1 | brier | ece |
| --- | --- | --- | --- | --- | --- |
| T0 | 0.8872 | 0.5767 | 0.5737 | 0.5326 | 0.0695 |
| T1 | 0.9628 | 0.5267 | 0.5104 | 0.5815 | 0.0651 |

### T0 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.6162 | 0.6778 | 0.6455 | 90.0 |
| review_not_warranted | 0.6228 | 0.6762 | 0.6484 | 105.0 |
| insufficient_evidence | 0.4713 | 0.3905 | 0.4271 | 105.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[61, 8, 21]
[9, 71, 25]
[29, 35, 41]
```

### T1 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5800 | 0.6517 | 0.6138 | 89.0 |
| review_not_warranted | 0.5401 | 0.6852 | 0.6041 | 108.0 |
| insufficient_evidence | 0.4127 | 0.2524 | 0.3133 | 103.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[58, 15, 16]
[13, 74, 21]
[29, 48, 26]
```

## hist_gradient_boosting

| phase | log_loss | accuracy | macro_f1 | brier | ece |
| --- | --- | --- | --- | --- | --- |
| T0 | 1.1778 | 0.5367 | 0.5407 | 0.6480 | 0.2131 |
| T1 | 1.3141 | 0.5067 | 0.5085 | 0.7108 | 0.2367 |

### T0 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.6000 | 0.5667 | 0.5829 | 90.0 |
| review_not_warranted | 0.5941 | 0.5714 | 0.5825 | 105.0 |
| insufficient_evidence | 0.4386 | 0.4762 | 0.4566 | 105.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[51, 11, 28]
[9, 60, 36]
[25, 30, 50]
```

### T1 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.6000 | 0.5393 | 0.5680 | 89.0 |
| review_not_warranted | 0.5339 | 0.5833 | 0.5575 | 108.0 |
| insufficient_evidence | 0.4020 | 0.3981 | 0.4000 | 103.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[48, 17, 24]
[8, 63, 37]
[24, 38, 41]
```

## hist_gradient_boosting_sigmoid

| phase | log_loss | accuracy | macro_f1 | brier | ece |
| --- | --- | --- | --- | --- | --- |
| T0 | 0.9466 | 0.5333 | 0.5348 | 0.5708 | 0.0913 |
| T1 | 0.9893 | 0.5033 | 0.4939 | 0.5956 | 0.0559 |

### T0 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5934 | 0.6000 | 0.5967 | 90.0 |
| review_not_warranted | 0.5727 | 0.6000 | 0.5860 | 105.0 |
| insufficient_evidence | 0.4343 | 0.4095 | 0.4216 | 105.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[54, 13, 23]
[9, 63, 33]
[28, 34, 43]
```

### T1 per-class metrics

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| review_warranted | 0.5952 | 0.5618 | 0.5780 | 89.0 |
| review_not_warranted | 0.5070 | 0.6667 | 0.5760 | 108.0 |
| insufficient_evidence | 0.3919 | 0.2816 | 0.3277 | 103.0 |

Confusion matrix (rows=true, columns=predicted; class order: review_warranted, review_not_warranted, insufficient_evidence):

```text
[50, 19, 20]
[11, 72, 25]
[23, 51, 29]
```

## Calibration conclusion

Sigmoid (Platt-style) calibration was compared with each available baseline. Isotonic calibration was not selected because 300 labels per phase are too few for stable three-class isotonic fits. Every production probability vector is reordered to the official class schema and renormalized before output.
