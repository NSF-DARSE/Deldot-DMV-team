#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACT_DIR, MODEL_DIR, RAW_DATA_DIR, REPORT_DIR
from src.models.training import train_and_evaluate


def main() -> None:
    t0 = pd.read_csv(ARTIFACT_DIR / "features_t0.csv")
    t1 = pd.read_csv(ARTIFACT_DIR / "features_t1.csv")
    labels = pd.read_csv(RAW_DATA_DIR / "Development_Labels" / "Development_Labels.csv")
    bundle, leaderboard = train_and_evaluate(t0, t1, labels, MODEL_DIR, REPORT_DIR)
    print(leaderboard.to_string(index=False))
    print(f"Selected: {bundle['selected_model']}")
    print(f"Model version: {bundle['model_version']}")
    print(f"Wrote artifacts to {MODEL_DIR} and {REPORT_DIR / 'model_evaluation.md'}")


if __name__ == "__main__":
    main()

