# T0 Vehicle-Reference Bridge: Before/After Comparison

## Decision

The naive vehicle-reference propagation is unsafe. The implemented linkage-v1.1.0 bridge is deliberately narrow: it requires unanimous high-confidence name anchors, no conflicting independently linked owner, strong agreement on both owner-name parts, and a positive runner-up margin.

- Identical OOF candidate/fold assignments: `true`
- T0 title linked rows: 32,707 -> 32,836 (+129)
- T0 title coverage: 67.9866% -> 68.2548% (+0.2682%)
- Strong name-linked anchor rows: 28,595
- Vehicle refs with strong anchors: 11,289
- Conflicting strong-anchor refs abstained: 676
- Refs with any independently linked owner conflict abstained: 809
- Rows rejected by name safeguards: 13,243
- Leave-one-alias-out precision: 100.00% (2,318/2,318; internal reference, not authoritative truth)
- Identity assignment uses only vehicle_ref and owner names. Row order, labels, state, and dates are not used.

## OOF point metrics

| Phase | Metric | Baseline | Bridge | Delta | Direction |
|---|---|---:|---:|---:|---|
| T0 | log_loss | 1.034991 | 1.034132 | -0.000859 | improved |
| T0 | macro_f1 | 0.445658 | 0.448905 | +0.003247 | improved |
| T0 | brier_multiclass | 0.629321 | 0.628625 | -0.000697 | improved |
| T0 | ece_10bin | 0.055895 | 0.060426 | +0.004531 | worsened |
| T0 | average_precision_review_warranted | 0.459163 | 0.459584 | +0.000421 | improved |
| T0 | ndcg_review_warranted | 0.831967 | 0.829667 | -0.002300 | worsened |
| T0 | precision_at_top_10pct | 0.500000 | 0.500000 | +0.000000 | unchanged |
| T0 | recall_at_top_20pct | 0.333333 | 0.333333 | +0.000000 | unchanged |
| T1 | log_loss | 1.042540 | 1.041262 | -0.001279 | improved |
| T1 | macro_f1 | 0.445538 | 0.448638 | +0.003100 | improved |
| T1 | brier_multiclass | 0.630618 | 0.630585 | -0.000034 | improved |
| T1 | ece_10bin | 0.063105 | 0.044436 | -0.018670 | improved |
| T1 | average_precision_review_warranted | 0.492134 | 0.482781 | -0.009352 | worsened |
| T1 | ndcg_review_warranted | 0.840714 | 0.817185 | -0.023530 | worsened |
| T1 | precision_at_top_10pct | 0.633333 | 0.600000 | -0.033333 | worsened |
| T1 | recall_at_top_20pct | 0.337079 | 0.348315 | +0.011236 | improved |

## Interpretation

Log loss, macro-F1, and multiclass Brier improve slightly at both phases. T1 ECE improves, while T0 ECE worsens. Priority behavior is mixed: T0 AP rises slightly, T1 AP/NDCG/precision@10% fall, and T1 recall@20% rises. The paired bootstrap intervals in `vehicle_bridge_paired_bootstrap.csv` should be used to judge uncertainty; with only 300 labeled candidates, these point changes are not evidence of a broad predictive improvement.

The bridge is retained because its linkage rule is conservative and its internal precision audit is clean, not because synthetic-label model metrics clearly dominate the baseline.
