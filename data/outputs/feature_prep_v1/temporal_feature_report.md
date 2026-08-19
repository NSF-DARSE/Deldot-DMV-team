# Temporal Feature Preparation v1

- Feature rule version: `temporal-features-v1.1.0`
- Frozen rule SHA-256: `f852f130f638b934ee6a8c2c0347750673be1b469f079812d0b7d20b0c3f5fe1`
- Frozen implementation SHA-256: `45867e853376e615ee1504dc79a0f413d9a2c92bb228c479c1cdace0c4360ab4`
- Candidate rows per phase: 12,000
- T0/T1 cumulative columns: 153
- T1 delta columns: 55

## Temporal safeguards

- T0 uses candidate_observed_date as the candidate-specific as-of date.
- T1 uses the complete release batch cutoff: 2026-08-11.
- 11,671 future-effective T0 records are excluded from T0 current-state features and retained as a data-quality count.
- Old OOS evidence is retained as history but separated from recent windows and recency-decayed evidence.
- DE-after-OOS and OOS-after-DE direction, per-source transitions, active addresses, and latest-source conflicts are explicit features.

## Delaware 60-day window

Delaware's 60-day new-resident vehicle registration requirement is represented only through proxy features. The clock uses the start of the latest uninterrupted Delaware address-evidence run. OOS title-domain signals are split into pre-move, within-window, and post-window counts.

Official sources:

- https://delcode.delaware.gov/title21/c021/sc01/index.html
- https://dmv.de.gov/VehicleServices/inspections/index.shtml?dc=v_equipment
- https://delcode.delaware.gov/title21/c027/sc01/index.html
- https://dmv.de.gov/DriverServices/drivers_license/index.shtml?dc=dr_lic_gen_req

The data cannot establish legal residency, actual registration status, or exemptions. No feature is a violation flag.

## Supported credential, persistence, and title-event additions

- Active OOS credential events are separated into within-60-day and post-60-day counts. Because credential type is absent, these are not labeled as authoritative driver-license records.
- T1 license-domain updates without credential_status are counted separately and never assumed active.
- Combined conflict requires both a post-window OOS title signal and an explicitly active post-window OOS credential signal.
- Prior-state persistence compares later title/active-credential states with the last OOS address state immediately preceding the latest DE address-evidence run.
- Meaningful title evidence distinguishes title_record, ownership_change, and T1 new_record from older/pre-move or generic record_update evidence.
- T1 observation lag and recent 90-day independent-source corroboration are included as data-supported context.

## Breakpoint

No model has been trained. These feature tables must be reviewed before CatBoost T0 training.
