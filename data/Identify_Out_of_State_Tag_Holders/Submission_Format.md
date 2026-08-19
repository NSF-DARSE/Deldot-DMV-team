# Submission Format

Submit a CSV file named `case_predictions.csv`.

Required columns, in any column order:

| Column | Requirement |
|---|---|
| `candidate_record_id` | Must match a challenge candidate identifier |
| `phase` | `T0` or `T1` |
| `predicted_class` | `review_warranted`, `review_not_warranted`, or `insufficient_evidence` |
| `p_review_warranted` | Numeric 0–1 |
| `p_review_not_warranted` | Numeric 0–1 |
| `p_insufficient_evidence` | Numeric 0–1 |
| `review_priority` | Numeric 0–1 |

For each row, the three class probabilities must sum to 1.

Every challenge case must have exactly one T0 row and one T1 row.

`submission_template.csv` contains the complete set of candidate/phase rows and the
required columns. It does not contain predictions.
