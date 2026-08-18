# Review-priority scoring

`review_priority` answers which cases staff should inspect first. It is not a probability of guilt, violation, liability, residency, or an enforcement decision.

## Alternatives tested on selected-model out-of-fold predictions

| formula | average_precision_review_warranted | precision_in_top_20_pct | mean_priority |
| --- | --- | --- | --- |
| recency_focused | 0.6529 | 0.7083 | 0.2893 |
| probability_only | 0.6521 | 0.7083 | 0.2902 |
| evidence_supported | 0.6518 | 0.7167 | 0.2915 |
| conflict_sensitive | 0.6517 | 0.7167 | 0.2883 |

## Production formula

The selected `evidence_supported` formula is:

```text
support = 0.45 × evidence_strength + 0.35 × evidence_recency + 0.20 × cross_source_agreement
priority = 0.78 × p_review_warranted
         + 0.14 × p_review_warranted × support
         + 0.08 × p_insufficient_evidence × (0.40 + 0.60 × normalized_entropy)
priority = clip(priority, 0, 1)
```

This keeps warranted probability dominant, promotes well-supported/recent cases, and reserves a small queue share for uncertain cases that may specifically benefit from human evidence gathering. The probability-only alternative was retained as a benchmark rather than silently equating probability with operational priority.
