"""Run pipeline stages and optionally write artifacts.

Stage 1: name/DOB linkage (``run_linkage``).
Stage 2: feature table + transparent rule baseline (``run_features_and_baseline``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oos_review import paths
from oos_review.baseline import apply_baseline, to_submission
from oos_review.features import build_t0_t1_features
from oos_review.linker import PersonIndex, link_t0_sources, link_t1_stream
from oos_review.load import (
    load_candidates,
    load_linked_bundle,
    load_t0_sources,
    load_t1_stream,
)


def run_linkage(*, save: bool = True, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Link T0 sources and the T1 update stream to candidate_record_id.

    Returns a dict that always includes ``candidates`` plus one frame per
    source. Linked frames keep original columns and add match metadata.
    """
    candidates = load_candidates()
    index = PersonIndex.from_candidates(candidates)
    t0_linked = link_t0_sources(candidates, load_t0_sources(), index=index)
    t1_linked = link_t1_stream(
        candidates,
        load_t1_stream(),
        index=index,
        title_linked=t0_linked["vehicle_title_events"],
    )

    bundle = {"candidates": candidates, **t0_linked, "evidence_update_stream": t1_linked}

    if save:
        dest = output_dir or paths.LINKED_DIR
        dest.mkdir(parents=True, exist_ok=True)
        for name, frame in bundle.items():
            frame.to_csv(dest / f"{name}.csv", index=False)

    return bundle


def run_features_and_baseline(
    bundle: dict[str, pd.DataFrame] | None = None,
    *,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build T0/T1 features and apply the rule baseline.

    If ``bundle`` is omitted, linked CSVs are loaded from ``outputs/linked/``.
    """
    if bundle is None:
        bundle = load_linked_bundle()
    features = build_t0_t1_features(bundle["candidates"], bundle)
    preds = apply_baseline(features)

    if save:
        paths.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        paths.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        features.to_csv(paths.FEATURES_DIR / "case_features.csv", index=False)
        preds.to_csv(paths.BASELINE_DIR / "case_predictions_audit.csv", index=False)
        to_submission(preds).to_csv(
            paths.BASELINE_DIR / "case_predictions.csv", index=False
        )

    return features, preds


def run_pipeline(*, save: bool = True) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Linkage then features/baseline. The usual one-call entry point."""
    bundle = run_linkage(save=save)
    features, preds = run_features_and_baseline(bundle, save=save)
    return bundle, features, preds
