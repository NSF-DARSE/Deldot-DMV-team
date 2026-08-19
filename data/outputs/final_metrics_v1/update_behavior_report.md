# Final T0→T1 Update-Behavior Analysis v1

- Current T1 class changes: 4,837 of 12,000
- Decision: `retain_current_unanchored_t1`
- Production T1 predictions modified: no

## Evidence groups and probability movement

| Group | Candidates | Current class changes | Change rate | Mean total-variation change | Mean absolute probability change |
|---|---:|---:|---:|---:|---:|
| meaningful_new_evidence | 11,447 | 4,669 | 40.788% | 0.096298 | 0.064199 |
| effectively_no_meaningful_update | 502 | 147 | 29.283% | 0.101276 | 0.067517 |
| weak_minor_update | 51 | 21 | 41.176% | 0.084383 | 0.056255 |

The effectively-no-meaningful-update group changes class too frequently operationally: its current 30.2% full-population flip rate is not lower than the meaningful-evidence group by enough, and its mean probability movement is slightly larger. This is a stability warning.

## Same-fold OOF anchoring comparison

| Approach | Log loss | Macro-F1 | Brier | ECE | Class-change rate |
|---|---:|---:|---:|---:|---:|
| current_unanchored | 1.041262 | 0.448638 | 0.630585 | 0.044436 | 44.667% |
| experimental_anchored | 1.035989 | 0.452697 | 0.627245 | 0.022836 | 43.667% |

Anchoring reduces OOF log loss by 0.005273, below the predeclared 0.01 clear-improvement threshold. Only 14 labeled candidates are in the effectively-no-meaningful-update group, below the required 30. Paired bootstrap intervals are saved separately and include substantial uncertainty because the affected labeled subgroup is small.

## Release decision

The current unanchored T1 model is retained, exactly as requested when anchoring does not clearly improve held-out behavior. The experimental anchor is not applied to `t1_predictions.csv` or the combined case predictions. The stability warning should be revisited once representative real labels provide at least 30 effectively-no-update cases.
