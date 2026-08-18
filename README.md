# Delaware DMV Potential Out-of-State Tag Holder Review

A complete hackathon prototype that turns the supplied synthetic T0 and T1 evidence into calibrated, explainable review recommendations and an operational review queue for Delaware DMV staff.

The system is decision support only. It prioritizes cases for human review; it does not determine residency, registration obligations, fees, violations, guilt, or enforcement action.

## 1. Problem

The challenge provides 12,000 candidate records, six T0 evidence sources, a later T1 evidence stream, and 300 development labels. For every candidate at both phases the system outputs one of:

- `review_warranted`
- `review_not_warranted`
- `insufficient_evidence`

It also outputs three calibrated class probabilities and a transparent `review_priority` from 0 to 1.

## 2. Human-in-the-loop boundary

Recommendations are queue-management aids. The frontend deliberately suppresses synthetic name, birth-date, and street fields; reviewers see only candidate IDs and relevant evidence metadata. Human workflow buttons write demo audit actions and never initiate enforcement. Protected attributes are neither constructed nor used.

## 3. Architecture

```text
Synthetic challenge CSVs
        ↓
profile + validated entity/evidence linkage
        ↓
explainable T0 and T1 feature snapshots
        ↓
cross-validated baselines + calibration
        ↓
official 24,000-row submission
        ↓
FastAPI ── SQLite audit log
        ↓
React reviewer dashboard
```

The ML and scoring packages live under `src/` and do not depend on the API or UI.

## 4. Data pipeline and joins

`scripts/profile_data.py` loads the dictionary and every CSV, prints shape, columns, dtypes, missingness, low-cardinality categorical values, date ranges, identifier fields, and label distributions, then writes `reports/data_profile.md`.

The evidence tables omit `candidate_record_id`. Inspection found a repeated package layout: the first four rows per candidate in address, credential, title, and external sources; the first two work rows; and two T1 update rows follow candidate order. This is accepted only after an 80% ordered exact-identity anchor test passes. Actual anchor rates are 94–99% for T0 and 81.8% for T1. Small unrelated T0 tails remain unresolved. If the layout validation fails, the resolver falls back to conservative name, birth-date, location, and vehicle anchors.

Names, date of birth, and street identifiers are linkage-only and never become model inputs.

## 5. Feature engineering

`src/features/` creates one 202-column explainable row per candidate and phase. Features include:

- per-source record, DE/non-DE, missing-state, and unique-state counts;
- latest state and days since latest evidence by source;
- 30/90/180/365-day DE and non-DE counts;
- recency-weighted state evidence;
- active/expired/superseded credential indicators;
- title recency and distinct vehicle counts;
- work signals as supporting evidence;
- external standard/limited quality features;
- cross-source agreement, conflict, completeness, and identity consistency;
- T1 new-record/action counts, reinforcement, contradiction, uncertainty resolution, and deltas.

T1 is modeled as T0 plus later observations. Corrections and status updates are separately counted rather than overwriting a T0 row because the update stream does not identify a target T0 `source_record_id`.

## 6. Models tested

The reproducible runtime compared:

- multinomial Logistic Regression;
- Random Forest;
- Histogram Gradient Boosting;
- sigmoid-calibrated versions of all three.

XGBoost and CatBoost are detected and added automatically when installed; neither optional package was available in the supplied runtime. Evaluation uses five-fold stratified cross-validation independently for T0 and T1, with preprocessing inside every fold.

## 7. Selected model and metrics

The selected production model is sigmoid-calibrated Logistic Regression, chosen by a composite dominated by multiclass log loss, Brier score, and calibration error with a small macro-F1 reward.

Cross-validated mean performance across T0/T1:

- multiclass log loss: **1.092**
- multiclass Brier score: **0.607**
- macro F1: **0.512**
- 10-bin expected calibration error: **0.094**

The modest labeled sample and resulting discrimination are explicitly treated as limitations. The system does not present low-confidence output as certainty. Full phase/model/per-class metrics and confusion matrices are in `reports/model_evaluation.md`.

## 8. Calibration

Sigmoid (Platt-style) calibration is evaluated against each uncalibrated baseline. Isotonic calibration is intentionally not used because 300 labeled cases per phase are insufficient for stable three-class isotonic fits. Production probabilities are reordered to the official schema and renormalized to sum to one.

## 9. Review priority

Priority is not copied from `p_review_warranted`. Four alternatives are tested on out-of-fold predictions. The production formula is:

```text
support = 0.45 × evidence_strength + 0.35 × evidence_recency + 0.20 × agreement
priority = 0.78 × p_warranted
         + 0.14 × p_warranted × support
         + 0.08 × p_insufficient × (0.40 + 0.60 × normalized_entropy)
```

The warranted probability remains dominant, evidence support breaks operational ties, and a small uncertainty term keeps cases needing information in the human queue. See `reports/priority_scoring.md`.

## 10. T0/T1 updates

Both snapshots are preserved. API and UI case detail show the new evidence count, class change, warranted-probability change, priority change, evidence-strength change, and chronological T0/T1 timeline.

## 11. Explainability

Explanations are deterministic feature/evidence rules—no LLM is used. They mention recent credential/title/address/external patterns, conflicts, sparse or variable identity evidence, and T1 reinforcement/contradiction. The selected linear model also exposes coefficient × transformed-feature associations averaged across calibration folds. These are described as associations, never causes.

## 12. Auditability

`scripts/initialize_audit.py` creates `data/audit/review_audit.db`. Every prediction snapshot stores candidate ID, phase, model version, timestamp, prediction, probabilities, priority, complete feature snapshot, and deterministic evidence summary. Human demo actions are timestamped separately. This answers, “What did the system know when it generated this recommendation?”

## 13. Run the backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/initialize_audit.py
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open API documentation at `http://127.0.0.1:8000/docs`. Endpoints include `/health`, `/cases`, phase-specific case views, timeline, explanation, priority, audit, review actions, and model metrics.

## 14. Run the frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://localhost:3000`. If the API is unavailable, the interface clearly switches to a limited preview dataset and disables persistent workflow behavior.

## 15. Regenerate the submission

```bash
python scripts/profile_data.py
python scripts/build_features.py
python scripts/train_models.py
python scripts/generate_predictions.py
python scripts/validate_submission.py
python scripts/initialize_audit.py
```

The official file is `outputs/case_predictions.csv`. The validator fails on wrong columns, wrong row count, duplicates, missing/extra candidate-phase pairs, invalid classes, nonnumeric/out-of-range values, probability-sum errors, or missing phases.

## Tests

```bash
pytest -q
cd frontend && npm run build
```

Tests cover feature artifacts, T0/T1 incorporation, probability sums, class and priority validity, duplicates/missing rows, API health/priority order, and deterministic explanations.

## Reproducibility and limitations

- Random seed: `42`.
- Model version and prediction timestamp are persisted with each run.
- Development labels are never joined into the feature population.
- The inferred ordered block relationship is package-specific and guarded by an anchor-rate validation; shuffled future packages use the conservative fallback resolver.
- Only 300 labeled cases are available. Metrics are estimates with meaningful uncertainty and should be reassessed on larger, temporally representative labeled samples.
- This prototype does not use cloud services, LLMs, RAG, or agents in its prediction path.

