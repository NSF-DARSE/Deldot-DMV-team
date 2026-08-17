# Challenge Brief
## Potential Out-of-State Tag Holder Review

### Operational context

The Department receives and maintains records from multiple administrative
sources. Those records may contain information relevant to whether a vehicle
record warrants additional staff review for possible Delaware registration
requirements.

The challenge data does **not** contain an authoritative field that directly
states whether a vehicle is currently registered in another state.

### Challenge

Develop a decision-support system that evaluates the provided records and
produces, for each challenge case:

- a review classification;
- class probabilities;
- a review-priority value.

Three review classifications are used:

- `review_warranted`
- `review_not_warranted`
- `insufficient_evidence`

The challenge has two evaluation phases. `Data_T0` is the initial information
available for each case. `Data_T1` contains later records that must be
incorporated before producing the T1 output.

The system is intended to support staff review. The challenge outcome is not an
automatic legal, residency, fee, or enforcement determination.

### Development labels

A labeled development set is provided in
`Development_Labels/development_labels.csv`.

The remaining evaluation labels are withheld by the organizers.

### Required submission

Submit one `case_predictions.csv` file using the exact schema shown in
`submission_template.csv`.

For every case, submit one T0 row and one T1 row.

Probability fields must be numeric values from 0 to 1 and sum to 1 within each
row. `review_priority` must be a numeric value from 0 to 1.

### Evaluation

Evaluation considers:

- classification quality;
- review prioritization;
- probability calibration;
- response to later evidence;
- performance on held-out evaluation cases.

A separate organizer review considers operational usefulness, clarity, and
auditability of the submitted solution.

