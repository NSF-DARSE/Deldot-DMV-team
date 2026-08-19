# Case Study: Delaware DMV Out-of-State Tag Holder Review

## Overview
Decision-support prototype that links synthetic Delaware DMV evidence, scores whether a case warrants staff review for possible out-of-state registration, and produces T0/T1 class probabilities plus a review priority. It does not determine residency, fees, or enforcement action.

The official submission file is `case_predictions.csv` at the repository root (24,000 rows: one T0 and one T1 prediction for each of 12,000 candidates).

## Repository Structure
- `oos_review/` – source code
- `docs/` – optional documentation (Sphinx scaffold)
- `data/` – input/output data (if applicable)

`oos_review/` contains the linkage, feature, and modeling libraries, pipeline scripts, review API (`backend/`), and Hencheck dashboard (`frontend/`). `data/` holds the challenge package, pipeline outputs, and the pre-bridge baseline snapshot. `case_predictions.csv` is the required challenge submission artifact.

## Documentation
This repository includes an optional Sphinx documentation scaffold.

Build HTML docs:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html
```

Pipeline handoff notes: `docs/pipeline_handoff.md`.

## Reproduce

Python 3.11+ is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python oos_review/scripts/run_linkage_v1.py
.venv/bin/python oos_review/scripts/build_temporal_features_v1.py
.venv/bin/python oos_review/scripts/select_compact_features_v1.py
.venv/bin/python oos_review/scripts/train_t0_model_v1.py
.venv/bin/python oos_review/scripts/train_t1_model_v1.py
.venv/bin/python oos_review/scripts/analyze_update_behavior_v1.py
.venv/bin/python oos_review/scripts/generate_final_metrics_v1.py
.venv/bin/python oos_review/scripts/compare_vehicle_bridge_v1.py
.venv/bin/python oos_review/scripts/validate_submission.py case_predictions.csv
```

```bash
.venv/bin/python -m pytest -q
```

Dashboard (local):

```bash
cd oos_review/backend
python3 -m uvicorn server:app --host 127.0.0.1 --port 8000
# in another terminal
cd oos_review/frontend && yarn start
```
