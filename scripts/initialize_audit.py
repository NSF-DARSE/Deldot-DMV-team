#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.database import AuditStore
from src.config import ARTIFACT_DIR, MODEL_DIR, OUTPUT_DIR
from src.explanations.rules import deterministic_explanation


def main() -> None:
    predictions = pd.read_csv(OUTPUT_DIR / "case_predictions.csv")
    metadata = json.loads((OUTPUT_DIR / "prediction_metadata.json").read_text(encoding="utf-8"))
    bundle = joblib.load(MODEL_DIR / "model_bundle.joblib")
    features = {
        "T0": pd.read_csv(ARTIFACT_DIR / "features_t0.csv"),
        "T1": pd.read_csv(ARTIFACT_DIR / "features_t1.csv"),
    }
    feature_maps = {
        phase: frame.set_index("candidate_record_id") for phase, frame in features.items()
    }
    summaries = {}
    for prediction in predictions.itertuples(index=False):
        summaries[(prediction.candidate_record_id, prediction.phase)] = deterministic_explanation(
            feature_maps[prediction.phase].loc[prediction.candidate_record_id],
            prediction._asdict(),
            prediction.phase,
        )
    store = AuditStore(PROJECT_ROOT / "data" / "audit" / "review_audit.db")
    store.initialize()
    inserted = store.seed_predictions(
        predictions,
        features,
        bundle["model_version"],
        metadata["prediction_timestamp"],
        summaries,
    )
    print(f"Audit database ready; inserted {inserted:,} new prediction snapshots at {store.path}")


if __name__ == "__main__":
    main()
