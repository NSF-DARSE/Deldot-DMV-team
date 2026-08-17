# Data flow and matching logic

This is the living design note for the solution. Update it when a stage
is added or a matching rule changes.

## Goal

Produce `case_predictions.csv` with one T0 row and one T1 row per candidate.
That file is not built yet. The first stage is attaching source records to
candidates so later features are auditable.

```text
candidate_records.csv
        |
        |  parse SYNGIV / SYNFAM / SYNDOB
        v
   PersonIndex (exact family, optional DOB)
        |
        +--> address_history
        +--> license_id_events          (uses DOB)
        +--> vehicle_title_events       (name, then vehicle_ref)
        +--> work_location_signals
        +--> external_context_signals
        +--> evidence_update_stream     (T1; name, then vehicle_ref)
        |
        v
linked tables in outputs/linked/
        |
        v
notebooks/01_labeled_case_explorer.ipynb   <-- current review surface
        |
        v
(later) feature tables -> model -> case_predictions.csv
```

## Why linkage is its own stage

Source files have no `candidate_record_id`. Joining on raw name strings
fails because of mixed case and truncated given names. Quietly prefix-matching
family names would also fail: `ALCV` and `ALCVD` are different people.

The linker writes `match_rule` and `match_score` onto every row so a reviewer
can see *why* a record was attached.

## Matching rules

Family name is always an exact match after uppercasing and stripping
`SYNFAM-`. Given names may be truncated. Date of birth, when present, is a
hard filter.

| Rule | Given name | Family | DOB | Typical score |
|---|---|---|---|---|
| `identity` | exact | exact | exact | 1.00 |
| `dob_prefix` | prefix, overlap ≥ 3 | exact | exact | 0.93 |
| `dob_initial` | prefix, overlap 1–2 | exact | exact | 0.88 |
| `name_exact` | exact | exact | source has none | 0.86 |
| `name_prefix` | prefix, overlap ≥ 3 | exact | source has none | 0.72 |
| `vehicle_ref` | not used | not used | not used | 0.80 |
| `unlinked` | no unique candidate | | | 0 |
| `ambiguous_unassigned` | two or more survivors | | | 0 |

Rejected on purpose:

- Family-name prefixes (`ALCV` ↛ `ALCVD`)
- DOB mismatch, even if the names are exact (same synthetic name can be two people)
- 1–2 character given names when the source has no DOB (too many people share a last name)

``vehicle_ref`` is a second pass for title rows and T1 title updates.
It only fires when:

- the vehicle is already uniquely tied to one candidate by a *name* match, and
- the leftover row's family name matches that owner, and
- the leftover given name is the same person (exact or truncated), not a
  different given name on the same title.

That is how a truncated `SYNGIV-N` can still attach, while another family
member on the same vehicle is left unlinked.

## Code map

| Path | Role |
|---|---|
| `oos_review/names.py` | Parse and compare synthetic tokens |
| `oos_review/linker.py` | `PersonIndex`, `link_frame`, T0/T1 wrappers |
| `oos_review/load.py` | CSV loaders |
| `oos_review/pipeline.py` | Stage runner; writes `outputs/linked/` |
| `oos_review/caseview.py` | Coverage table and per-case dossier |
| `tests/` | Policy tests for the rules above |
| `notebooks/01_labeled_case_explorer.ipynb` | Inspect labeled cases through the linker |

## Observed coverage on this package

These shares are after the rules above, not after forcing every row onto a
candidate. Unlinked rows are other people in the source files.

| Source | Linked | Ambiguous | Notes |
|---|---:|---:|---|
| address_history | 59.6% | 431 | no DOB |
| license_id_events | 68.6% | 8 | DOB makes 1-letter givens safe |
| vehicle_title_events | 67.2% | 453 | includes 3,599 `vehicle_ref` fills |
| work_location_signals | 78.8% | 288 | |
| external_context_signals | 59.6% | 453 | |
| evidence_update_stream | 61.9% | 232 | includes 386 `vehicle_ref` fills |

On the 300 labeled cases, raw linked-row counts and DE-share of address/license
rows do **not** separate the three classes. The next stage has to use recency,
conflicts, and source quality, not volume.

## How to run

```bash
python -m pytest tests
```

From a notebook or REPL:

```python
from oos_review.pipeline import run_linkage
bundle = run_linkage(save=True)
```

## Next stage

Use the linked tables and the 300 development labels to define features
(DE-tie vs out-of-state-tie, recency, conflicts, missingness) and a
transparent rule baseline before fitting a 3-class model.
