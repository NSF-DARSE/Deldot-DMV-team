"""Repository path helpers for the oos_review package."""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = REPO_ROOT / "data"
CHALLENGE_DATA = DATA_DIR / "Identify_Out_of_State_Tag_Holders"
OUTPUTS = DATA_DIR / "outputs"
CONFIGS = PACKAGE_ROOT / "configs"
BASELINE = DATA_DIR / "baseline_snapshot"
DASHBOARD_DATA = PACKAGE_ROOT / "backend" / "data"


def ensure_import_path() -> None:
    for path in (REPO_ROOT, PACKAGE_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
