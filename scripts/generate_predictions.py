#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACT_DIR, MODEL_DIR, OUTPUT_DIR, RAW_DATA_DIR, REPORT_DIR
from src.scoring.predict import generate_predictions
from src.scoring.priority import evaluate_priority_alternatives


def _priority_report(metrics: pd.DataFrame) -> str:
    display = metrics.copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "# Review-priority scoring",
            "",
            "`review_priority` answers which cases staff should inspect first. It is not a probability of guilt, violation, liability, residency, or an enforcement decision.",
            "",
            "## Alternatives tested on selected-model out-of-fold predictions",
            "",
            header,
            separator,
            *rows,
            "",
            "## Production formula",
            "",
            "The selected `evidence_supported` formula is:",
            "",
            "```text",
            "support = 0.45 × evidence_strength + 0.35 × evidence_recency + 0.20 × cross_source_agreement",
            "priority = 0.78 × p_review_warranted",
            "         + 0.14 × p_review_warranted × support",
            "         + 0.08 × p_insufficient_evidence × (0.40 + 0.60 × normalized_entropy)",
            "priority = clip(priority, 0, 1)",
            "```",
            "",
            "This keeps warranted probability dominant, promotes well-supported/recent cases, and reserves a small queue share for uncertain cases that may specifically benefit from human evidence gathering. The probability-only alternative was retained as a benchmark rather than silently equating probability with operational priority.",
        ]
    ) + "\n"


def main() -> None:
    predictions, metadata = generate_predictions(
        MODEL_DIR / "model_bundle.joblib",
        ARTIFACT_DIR / "features_t0.csv",
        ARTIFACT_DIR / "features_t1.csv",
        RAW_DATA_DIR / "Submission_Template.csv",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "case_predictions.csv", index=False, float_format="%.10f")
    (OUTPUT_DIR / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    bundle = joblib.load(MODEL_DIR / "model_bundle.joblib")
    selected = bundle["selected_model"]
    oof = pd.read_csv(MODEL_DIR / "oof_predictions.csv")
    oof = oof.loc[oof["model_name"].eq(selected)].copy()
    feature_pieces = []
    for phase in ("T0", "T1"):
        features = pd.read_csv(ARTIFACT_DIR / f"features_{phase.lower()}.csv")
        part = oof.loc[oof["phase"].eq(phase)].merge(
            features, on="candidate_record_id", how="left", validate="one_to_one"
        )
        feature_pieces.append(part)
    priority_data = pd.concat(feature_pieces, ignore_index=True)
    probabilities = priority_data[
        ["p_review_warranted", "p_review_not_warranted", "p_insufficient_evidence"]
    ].to_numpy()
    priority_metrics = evaluate_priority_alternatives(
        probabilities, priority_data, priority_data["actual_class"].to_numpy()
    )
    priority_metrics.to_csv(MODEL_DIR / "priority_evaluation.csv", index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "priority_scoring.md").write_text(
        _priority_report(priority_metrics), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(priority_metrics.to_string(index=False))
    print(f"Wrote {OUTPUT_DIR / 'case_predictions.csv'}")


if __name__ == "__main__":
    main()

