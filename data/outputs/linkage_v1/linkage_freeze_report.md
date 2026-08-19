# Frozen Linkage v1 Report

- Rule version: `linkage-v1.1.0`
- Frozen rule SHA-256: `f17367cd773c9e379a8e0ff6236f1e64246c35e9e3640d625fdd44b678264c44`
- Frozen implementation SHA-256: `f04c013cb60d9fbc1ba9501d7e3f30f39cb1b34742a8ef74bdf054b3262d8ecd`
- Candidate records: 12,000
- Evidence/update records evaluated: 240,600
- Linked: 172,148 (71.5%)
- Unresolved: 68,452

## Transferability safeguards

- No row-order or repeated-block assumptions.
- No labels, predictions, DE/OOS state, or model metrics used in linkage.
- Dates are preserved for later features but are not identity evidence.
- Aliases are learned only from strong DOB/name or address/name anchors.
- Ambiguous and contradictory records remain unresolved instead of being forced.
- Vehicle references propagate identity only when T0 ownership is unambiguous and the name is noncontradictory.
- Link confidence is a deterministic rule-strength score, not an empirical probability.

## T0 vehicle-reference bridge

- High-confidence name-linked anchor rows: 28,595
- Vehicle refs with at least one strong anchor: 11,289
- Conflicting strong-anchor vehicle refs (abstained): 676
- Vehicle refs with any independently linked owner conflict (abstained): 809
- Previously unresolved T0 title rows recovered: 129
- Leave-one-alias-out accepted reference rows: 2,318
- Precision against held-out strong-name reference: 100.0%
- The bridge uses only vehicle_ref and owner names; row order, labels, state, and dates are prohibited identity inputs.
- This is an internal consistency audit, not authoritative ground truth.

## Source results

| phase | source | total_records | linked_records | unresolved_records | link_rate | tier_a_records | tier_b_records | tier_c_records | linked_candidate_coverage | mean_link_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | address_history | 48121 | 32747 | 15374 | 0.680514 | 0 | 29019 | 3728 | 11828 | 0.946765 |
| T0 | license_id_events | 48124 | 35911 | 12213 | 0.746218 | 34721 | 968 | 222 | 11997 | 0.988401 |
| T0 | external_context_signals | 48116 | 32732 | 15384 | 0.680273 | 0 | 29049 | 3683 | 11832 | 0.946863 |
| T0 | vehicle_title_events | 48108 | 32836 | 15272 | 0.682548 | 129 | 29035 | 3672 | 11822 | 0.946963 |
| T0 | work_location_signals | 24131 | 21633 | 2498 | 0.896482 | 0 | 19201 | 2432 | 11683 | 0.946882 |
| T1 | evidence_update_stream | 24000 | 16289 | 7711 | 0.678708 | 3224 | 11703 | 1362 | 10639 | 0.955857 |

## Strong-anchor holdout audit

- Holdout reference rows: 7,159
- Accepted holdout rows: 6,266
- Precision against DOB-anchor reference: 100.0%
- Coverage on DOB-anchor holdout: 87.5%
- This is an internal consistency audit, not authoritative link accuracy; no ground-truth links were supplied.

## Real DMV deployment requirements

- Revalidate normalization and thresholds on a manually adjudicated, representative linkage sample.
- Monitor false-link and missed-link rates separately by source, name pattern, and missingness pattern.
- Keep raw PII in governed source systems; downstream feature tables should use candidate/source identifiers and linkage diagnostics.
- Treat low coverage as uncertainty, not evidence that a candidate has no out-of-state activity.
- Version and re-freeze the rules whenever source schemas or data-quality patterns change.

## Freeze boundary

Any threshold or rule change requires a new rule version, a new hash, and regeneration of every downstream artifact.
