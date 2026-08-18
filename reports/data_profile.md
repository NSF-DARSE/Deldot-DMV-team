# Delaware DMV challenge data profile

Generated deterministically by `scripts/profile_data.py`. Percentages are based on the supplied synthetic package; no external enrichment was used.

## Executive findings

- Candidate population: **12,000** unique candidate records.
- Required prediction rows: **24,000** (one T0 and one T1 row per candidate).
- Development labels: **300** candidates, with separate T0 and T1 labels.
- Evidence tables do not contain `candidate_record_id`. The package nevertheless has a strongly validated repeated-block layout: the first 4 source rows per candidate (2 for work and T1) follow candidate order, with small unrelated tails in T0 files.
- Candidate names, birth dates, and street identifiers are used only for record linkage. They are excluded from predictive features.
- T1 is an append-only evidence update stream. `record_action` describes a new record, correction, or status update; it supplements the T0 snapshot and does not silently overwrite source rows because the stream does not provide a target T0 record identifier.
- Unresolved/low-confidence evidence is not forced onto a candidate. Match method and confidence are retained for auditability.

## Data dictionary

| phase_folder | file | column | inferred_type | description |
| --- | --- | --- | --- | --- |
| Data_T0 | address_history.csv | source_record_id | string | Identifier for a source record within the synthetic challenge data. |
| Data_T0 | address_history.csv | first_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | address_history.csv | last_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | address_history.csv | street_address | string | Synthetic location identifier or address-related value recorded by the source. |
| Data_T0 | address_history.csv | state | string | State/jurisdiction code recorded by the source. |
| Data_T0 | address_history.csv | effective_start_date | date/datetime | Date/time value associated with this source field. |
| Data_T0 | address_history.csv | effective_end_date | date/datetime | Date/time value associated with this source field. |
| Data_T0 | address_history.csv | source_type | string | Source-system/context field in the synthetic challenge data. |
| Data_T0 | candidate_records.csv | candidate_record_id | string | Synthetic identifier used in the challenge dataset. |
| Data_T0 | candidate_records.csv | first_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | candidate_records.csv | last_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | candidate_records.csv | date_of_birth | date/datetime | Synthetic date-of-birth representation using the reserved SYNDOB- namespace. |
| Data_T0 | candidate_records.csv | observed_street_address | string | Synthetic location identifier or address-related value recorded by the source. |
| Data_T0 | candidate_records.csv | observed_state | string | State/jurisdiction value recorded by the source. |
| Data_T0 | candidate_records.csv | candidate_observed_date | date/datetime | Date/time value associated with this source field. |
| Data_T0 | candidate_records.csv | review_status | string | Status/category value recorded by the source. |
| Data_T0 | external_context_signals.csv | source_record_id | string | Identifier for a source record within the synthetic challenge data. |
| Data_T0 | external_context_signals.csv | first_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | external_context_signals.csv | last_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | external_context_signals.csv | signal_type | string | Field supplied in the synthetic challenge dataset. |
| Data_T0 | external_context_signals.csv | signal_state | string | State/jurisdiction value recorded by the source. |
| Data_T0 | external_context_signals.csv | effective_date | date/datetime | Date the recorded information becomes effective. |
| Data_T0 | external_context_signals.csv | evidence_quality | string | Field supplied in the synthetic challenge dataset. |
| Data_T0 | external_context_signals.csv | source_description | string | Source-system/context field in the synthetic challenge data. |
| Data_T0 | license_id_events.csv | source_record_id | string | Identifier for a source record within the synthetic challenge data. |
| Data_T0 | license_id_events.csv | first_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | license_id_events.csv | last_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | license_id_events.csv | date_of_birth | date/datetime | Synthetic date-of-birth representation using the reserved SYNDOB- namespace. |
| Data_T0 | license_id_events.csv | credential_state | string | State/jurisdiction value recorded by the source. |
| Data_T0 | license_id_events.csv | event_type | string | Type/category of the recorded event. |
| Data_T0 | license_id_events.csv | event_date | date/datetime | Date associated with the recorded event. |
| Data_T0 | license_id_events.csv | credential_status | string | Status/category value recorded by the source. |
| Data_T0 | vehicle_title_events.csv | source_record_id | string | Identifier for a source record within the synthetic challenge data. |
| Data_T0 | vehicle_title_events.csv | vehicle_ref | string | Field supplied in the synthetic challenge dataset. |
| Data_T0 | vehicle_title_events.csv | owner_first_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | vehicle_title_events.csv | owner_last_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | vehicle_title_events.csv | event_type | string | Type/category of the recorded event. |
| Data_T0 | vehicle_title_events.csv | event_state | string | State/jurisdiction value recorded by the source. |
| Data_T0 | vehicle_title_events.csv | event_date | date/datetime | Date associated with the recorded event. |
| Data_T0 | work_location_signals.csv | source_record_id | string | Identifier for a source record within the synthetic challenge data. |
| Data_T0 | work_location_signals.csv | first_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | work_location_signals.csv | last_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T0 | work_location_signals.csv | work_state | string | State/jurisdiction value recorded by the source. |
| Data_T0 | work_location_signals.csv | observed_date | date/datetime | Date/time value associated with this source field. |
| Data_T0 | work_location_signals.csv | source_type | string | Source-system/context field in the synthetic challenge data. |
| Data_T1 | evidence_update_stream.csv | source_record_id | string | Identifier for a source record within the synthetic challenge data. |
| Data_T1 | evidence_update_stream.csv | release_batch | date/datetime | Field supplied in the synthetic challenge dataset. |
| Data_T1 | evidence_update_stream.csv | source_domain | string | Source-system/context field in the synthetic challenge data. |
| Data_T1 | evidence_update_stream.csv | record_action | string | Field supplied in the synthetic challenge dataset. |
| Data_T1 | evidence_update_stream.csv | first_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T1 | evidence_update_stream.csv | last_name | string | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace. |
| Data_T1 | evidence_update_stream.csv | state | string | State/jurisdiction code recorded by the source. |
| Data_T1 | evidence_update_stream.csv | vehicle_ref | string | Field supplied in the synthetic challenge dataset. |
| Data_T1 | evidence_update_stream.csv | effective_date | date/datetime | Date the recorded information becomes effective. |
| Data_T1 | evidence_update_stream.csv | observed_date | date/datetime | Date/time value associated with this source field. |
| Data_T1 | evidence_update_stream.csv | source_description | string | Source-system/context field in the synthetic challenge data. |

## File-level profiles

### `Data_T0/candidate_records.csv`

Shape: **12,000 rows × 8 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| candidate_record_id | object | 0.0 | 12000 |
| first_name | object | 0.0 | 3450 |
| last_name | object | 0.0 | 5793 |
| date_of_birth | object | 0.0 | 9534 |
| observed_street_address | object | 0.0 | 11546 |
| observed_state | object | 0.0 | 9 |
| candidate_observed_date | datetime64[ns] | 0.0 | 176 |
| review_status | object | 0.0 | 1 |

Categorical values:

| column | values (count) |
| --- | --- |
| observed_state | DE: 5,956; PA: 2,039; MD: 1,610; NJ: 935; NY: 433; VA: 367; FL: 245; NC: 230; SC: 185 |
| review_status | unreviewed: 12,000 |

Date ranges:

| column | minimum | maximum | unparseable_or_missing |
| --- | --- | --- | --- |
| candidate_observed_date | 2026-02-02 | 2026-07-27 | 0 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| candidate_record_id | 12000 | 0 |
| first_name | 3450 | 8550 |
| last_name | 5793 | 6207 |
| date_of_birth | 9534 | 2466 |
| observed_street_address | 11546 | 454 |

### `Data_T0/address_history.csv`

Shape: **48,121 rows × 8 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| source_record_id | object | 0.0 | 48121 |
| first_name | object | 0.0 | 7576 |
| last_name | object | 0.0 | 10500 |
| street_address | object | 0.0 | 41800 |
| state | object | 2.502 | 9 |
| effective_start_date | datetime64[ns] | 0.0 | 1857 |
| effective_end_date | datetime64[ns] | 28.744 | 1324 |
| source_type | object | 0.0 | 4 |

Categorical values:

| column | values (count) |
| --- | --- |
| state | DE: 23,458; PA: 7,936; MD: 6,255; NJ: 3,617; NY: 1,651; VA: 1,455; nan: 1,204; FL: 941; NC: 911; SC: 693 |
| source_type | customer_update: 12,083; service_record: 12,052; account_profile: 11,994; correspondence_record: 11,992 |

Date ranges:

| column | minimum | maximum | unparseable_or_missing |
| --- | --- | --- | --- |
| effective_start_date | 2020-08-02 | 2026-08-01 | 0 |
| effective_end_date | 2023-05-16 | 2027-01-12 | 13832 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| source_record_id | 48121 | 0 |
| first_name | 7576 | 40545 |
| last_name | 10500 | 37621 |
| street_address | 41800 | 6321 |

### `Data_T0/license_id_events.csv`

Shape: **48,124 rows × 8 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| source_record_id | object | 0.0 | 48124 |
| first_name | object | 0.0 | 7536 |
| last_name | object | 0.0 | 10756 |
| date_of_birth | object | 2.435 | 15087 |
| credential_state | object | 2.548 | 9 |
| event_type | object | 0.0 | 3 |
| event_date | datetime64[ns] | 0.0 | 1853 |
| credential_status | object | 0.0 | 4 |

Categorical values:

| column | values (count) |
| --- | --- |
| credential_state | DE: 23,434; PA: 7,957; MD: 6,263; NJ: 3,598; NY: 1,648; VA: 1,455; nan: 1,226; FL: 954; NC: 906; SC: 683 |
| event_type | credential_update: 24,063; credential_record: 12,031; credential_status_change: 12,030 |
| credential_status | active: 18,112; expired: 10,692; superseded: 10,664; unknown: 8,656 |

Date ranges:

| column | minimum | maximum | unparseable_or_missing |
| --- | --- | --- | --- |
| event_date | 2020-08-02 | 2026-08-01 | 0 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| source_record_id | 48124 | 0 |
| first_name | 7536 | 40588 |
| last_name | 10756 | 37368 |
| date_of_birth | 15087 | 31865 |

### `Data_T0/vehicle_title_events.csv`

Shape: **48,108 rows × 7 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| source_record_id | object | 0.0 | 48108 |
| vehicle_ref | object | 0.0 | 11651 |
| owner_first_name | object | 0.0 | 7585 |
| owner_last_name | object | 0.0 | 10466 |
| event_type | object | 0.0 | 3 |
| event_state | object | 2.536 | 9 |
| event_date | datetime64[ns] | 0.0 | 1856 |

Categorical values:

| column | values (count) |
| --- | --- |
| event_type | record_update: 24,043; title_record: 12,035; ownership_change: 12,030 |
| event_state | DE: 23,465; PA: 7,926; MD: 6,249; NJ: 3,606; NY: 1,655; VA: 1,447; nan: 1,220; FL: 941; NC: 913; SC: 686 |

Date ranges:

| column | minimum | maximum | unparseable_or_missing |
| --- | --- | --- | --- |
| event_date | 2020-08-02 | 2026-08-01 | 0 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| source_record_id | 48108 | 0 |
| vehicle_ref | 11651 | 36457 |
| owner_first_name | 7585 | 40523 |
| owner_last_name | 10466 | 37642 |

### `Data_T0/work_location_signals.csv`

Shape: **24,131 rows × 6 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| source_record_id | object | 0.0 | 24131 |
| first_name | object | 0.0 | 5016 |
| last_name | object | 0.0 | 7521 |
| work_state | object | 2.511 | 9 |
| observed_date | datetime64[ns] | 0.0 | 1001 |
| source_type | object | 0.0 | 1 |

Categorical values:

| column | values (count) |
| --- | --- |
| work_state | DE: 11,764; PA: 3,982; MD: 3,157; NJ: 1,788; NY: 821; VA: 731; nan: 606; FL: 478; NC: 458; SC: 346 |
| source_type | employment_location_context: 24,131 |

Date ranges:

| column | minimum | maximum | unparseable_or_missing |
| --- | --- | --- | --- |
| observed_date | 2023-11-05 | 2026-08-01 | 0 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| source_record_id | 24131 | 0 |
| first_name | 5016 | 19115 |
| last_name | 7521 | 16610 |

### `Data_T0/external_context_signals.csv`

Shape: **48,116 rows × 8 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| source_record_id | object | 0.0 | 48116 |
| first_name | object | 0.0 | 7457 |
| last_name | object | 0.0 | 10531 |
| signal_type | object | 0.0 | 3 |
| signal_state | object | 2.494 | 9 |
| effective_date | datetime64[ns] | 0.0 | 1850 |
| evidence_quality | object | 0.0 | 2 |
| source_description | object | 0.0 | 4 |

Categorical values:

| column | values (count) |
| --- | --- |
| signal_type | reference_update: 24,068; account_context: 12,028; service_context: 12,020 |
| signal_state | DE: 23,442; PA: 7,941; MD: 6,262; NJ: 3,625; NY: 1,654; VA: 1,451; nan: 1,200; FL: 946; NC: 905; SC: 690 |
| evidence_quality | standard: 36,085; limited: 12,031 |
| source_description | service_feed_c: 12,093; reference_feed_b: 12,070; account_feed_d: 11,996; reference_feed_a: 11,957 |

Date ranges:

| column | minimum | maximum | unparseable_or_missing |
| --- | --- | --- | --- |
| effective_date | 2020-08-02 | 2026-08-01 | 0 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| source_record_id | 48116 | 0 |
| first_name | 7457 | 40659 |
| last_name | 10531 | 37585 |

### `Data_T1/evidence_update_stream.csv`

Shape: **24,000 rows × 11 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| source_record_id | object | 0.0 | 24000 |
| release_batch | object | 0.0 | 1 |
| source_domain | object | 0.0 | 4 |
| record_action | object | 0.0 | 3 |
| first_name | object | 0.0 | 5406 |
| last_name | object | 0.0 | 7653 |
| state | object | 2.512 | 9 |
| vehicle_ref | object | 80.362 | 4259 |
| effective_date | datetime64[ns] | 0.0 | 53 |
| observed_date | datetime64[ns] | 0.0 | 4 |
| source_description | object | 0.0 | 4 |

Categorical values:

| column | values (count) |
| --- | --- |
| release_batch | T1: 24,000 |
| source_domain | address: 7,244; license: 6,033; external: 6,010; title: 4,713 |
| record_action | record_correction: 8,032; status_update: 8,023; new_record: 7,945 |
| state | DE: 12,240; PA: 3,753; MD: 3,025; NJ: 1,705; NY: 803; VA: 664; nan: 603; FL: 453; NC: 412; SC: 342 |
| source_description | late_event_feed_d: 6,052; correction_feed_c: 5,995; source_refresh_b: 5,991; source_refresh_a: 5,962 |

Date ranges:

| column | minimum | maximum | unparseable_or_missing |
| --- | --- | --- | --- |
| effective_date | 2026-06-17 | 2026-08-08 | 0 |
| observed_date | 2026-08-08 | 2026-08-11 | 0 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| source_record_id | 24000 | 0 |
| first_name | 5406 | 18594 |
| last_name | 7653 | 16347 |
| vehicle_ref | 4259 | 454 |

### `Development_Labels/Development_Labels.csv`

Shape: **300 rows × 3 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| candidate_record_id | object | 0.0 | 300 |
| label_t0 | object | 0.0 | 3 |
| label_t1 | object | 0.0 | 3 |

Categorical values:

| column | values (count) |
| --- | --- |
| label_t0 | insufficient_evidence: 105; review_not_warranted: 105; review_warranted: 90 |
| label_t1 | review_not_warranted: 108; insufficient_evidence: 103; review_warranted: 89 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| candidate_record_id | 300 | 0 |

### `Submission_Template.csv`

Shape: **24,000 rows × 7 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| candidate_record_id | object | 0.0 | 12000 |
| phase | object | 0.0 | 2 |
| predicted_class | float64 | 100.0 | 0 |
| p_review_warranted | float64 | 100.0 | 0 |
| p_review_not_warranted | float64 | 100.0 | 0 |
| p_insufficient_evidence | float64 | 100.0 | 0 |
| review_priority | float64 | 100.0 | 0 |

Categorical values:

| column | values (count) |
| --- | --- |
| phase | T0: 12,000; T1: 12,000 |

Candidate/entity identifier fields:

| identifier/entity field | unique_non_null | duplicate_non_null_rows |
| --- | --- | --- |
| candidate_record_id | 12000 | 12000 |

### `Data_Dictionary.csv`

Shape: **56 rows × 5 columns**

| column | dtype | missing_pct | unique_non_null |
| --- | --- | --- | --- |
| phase_folder | object | 0.0 | 2 |
| file | object | 0.0 | 7 |
| column | object | 0.0 | 32 |
| inferred_type | object | 0.0 | 2 |
| description | object | 0.0 | 14 |

Categorical values:

| column | values (count) |
| --- | --- |
| phase_folder | Data_T0: 45; Data_T1: 11 |
| file | evidence_update_stream.csv: 11; address_history.csv: 8; candidate_records.csv: 8; external_context_signals.csv: 8; license_id_events.csv: 8; vehicle_title_events.csv: 7; work_location_signals.csv: 6 |
| inferred_type | string: 44; date/datetime: 12 |
| description | Synthetic human-name representation using a reserved SYNGIV-/SYNFAM-/SYNNAME- namespace.: 14; Identifier for a source record within the synthetic challenge data.: 6; Field supplied in the synthetic challenge dataset.: 6; Date/time value associated with this source field.: 5; Source-system/context field in the synthetic challenge data.: 5; State/jurisdiction value recorded by the source.: 5; Synthetic location identifier or address-related value recorded by the source.: 2; State/jurisdiction code recorded by the source.: 2; Synthetic date-of-birth representation using the reserved SYNDOB- namespace.: 2; Status/category value recorded by the source.: 2; Date the recorded information becomes effective.: 2; Type/category of the recorded event.: 2; Date associated with the recorded event.: 2; Synthetic identifier used in the challenge dataset.: 1 |

Candidate/entity identifier fields:

_None._

## Relationship and join analysis

The dictionary confirms that only `candidate_records.csv`, the labels, and the submission template contain `candidate_record_id`. Source evidence must therefore be linked. Inspection found a repeated candidate-block layout: the first 48,000 rows in each four-event source, first 24,000 work rows, and all 24,000 T1 rows align with candidate order. This is inferred rather than assumed: 94–99% of T0 blocks and 81.8% of T1 blocks contain at least one exact normalized candidate identity, far above chance, while the small T0 tails are unrelated noise. The resolver accepts this layout only when an 80% anchor threshold passes, and otherwise falls back to field-based entity resolution.

1. validated package block order and block-size invariants;
2. exact normalized reserved synthetic first/last identity;
3. credential birth date plus compatible name;
4. aliases learned only from high-confidence credential links;
5. unique synthetic location and vehicle-reference anchors;
6. conservative fuzzy matching only when the validated layout is unavailable.

Rows in the unrelated T0 tails remain unresolved. If block validation fails on a future package, low-confidence fallback matches also remain unresolved. This is safer than assigning unrelated evidence and is represented later as evidence incompleteness.

| domain | rows | matched_rows | match_rate | candidates_covered | mean_match_confidence | block_anchor_rate |
| --- | --- | --- | --- | --- | --- | --- |
| license | 48124 | 48000 | 0.9974 | 12000 | 0.9829 | 0.9871666666666666 |
| address | 48121 | 48000 | 0.9975 | 12000 | 0.9828 | 0.9855833333333334 |
| title | 48108 | 48000 | 0.9978 | 12000 | 0.9828 | 0.9861666666666666 |
| work | 24131 | 24000 | 0.9946 | 12000 | 0.9903 | 0.94225 |
| external | 48116 | 48000 | 0.9976 | 12000 | 0.9828 | 0.9868333333333333 |
| updates | 24000 | 24000 | 1.0 | 12000 | 0.9828 | 0.81825 |

### T1 interpretation

All update rows have `release_batch=T1`. The stream contains four source domains (`address`, `license`, `external`, and `title`) and three actions (`new_record`, `record_correction`, and `status_update`). Because no update row points to a specific T0 `source_record_id`, corrections and status updates are modeled as later evidence observations and separately counted; existing T0 records are preserved. T1 features equal the T0 evidence snapshot plus resolved update observations, along with change/delta features.

## Development-label distributions

| phase | class | count | percentage |
| --- | --- | --- | --- |
| T0 | insufficient_evidence | 105 | 35.0 |
| T0 | review_not_warranted | 105 | 35.0 |
| T0 | review_warranted | 90 | 30.0 |
| T1 | review_not_warranted | 108 | 36.0 |
| T1 | insufficient_evidence | 103 | 34.33 |
| T1 | review_warranted | 89 | 29.67 |

T0/T1 labels are outcomes for the same 300 development candidates. Evaluation must keep candidate identity grouped across phases (or evaluate each phase separately) to prevent the same candidate from leaking between train and validation folds.

## Modeling guardrails discovered from the package

- `review_status` is constant (`unreviewed`) and is not predictive.
- Names, date of birth, and street identifiers are linkage-only fields and are not model inputs.
- `record_action` is a T1 process field; it is used only in T1 update/change features.
- With only 300 labeled candidates, model selection must emphasize cross-validated log loss, calibration, and macro F1 rather than training fit.
- The three labels are operational review recommendations, not legal, residency, registration, fee, or enforcement determinations.
