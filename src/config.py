from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path(
    os.environ.get(
        "DMV_RAW_DATA_DIR",
        PROJECT_ROOT / "data" / "raw" / "Identify_Out_of_State_Tag_Holders",
    )
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42
DE_STATE = "DE"
VALID_CLASSES = (
    "review_warranted",
    "review_not_warranted",
    "insufficient_evidence",
)

