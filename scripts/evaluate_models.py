#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR


def main() -> None:
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        raise SystemExit("No metrics found. Run scripts/train_models.py first.")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print(f"Selected model: {metrics['selection']['model_name']}")
    for row in metrics["selection"]["leaderboard"]:
        print(
            f"{row['model_name']:<38} log_loss={row['mean_log_loss']:.4f} "
            f"brier={row['mean_brier']:.4f} macro_f1={row['mean_macro_f1']:.4f} "
            f"ece={row['mean_ece']:.4f}"
        )


if __name__ == "__main__":
    main()
