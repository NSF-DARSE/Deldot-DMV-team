#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACT_DIR
from src.data.loaders import load_raw_data
from src.features.build import build_features


def main() -> None:
    data = load_raw_data()
    artifacts = build_features(data)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts.t0.to_csv(ARTIFACT_DIR / "features_t0.csv", index=False)
    artifacts.t1.to_csv(ARTIFACT_DIR / "features_t1.csv", index=False)
    artifacts.timeline.to_csv(
        ARTIFACT_DIR / "evidence_timeline.csv.gz", index=False, compression="gzip"
    )
    artifacts.resolution.to_csv(ARTIFACT_DIR / "entity_resolution_summary.csv", index=False)
    print(f"T0 features: {artifacts.t0.shape}")
    print(f"T1 features: {artifacts.t1.shape}")
    print(f"Timeline rows: {len(artifacts.timeline):,}")
    print("Entity resolution:")
    print(artifacts.resolution.to_string(index=False))
    print(f"Wrote feature artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
