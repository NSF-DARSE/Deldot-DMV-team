"""Run pipeline stages and optionally write artifacts.

Stage 1: name/DOB linkage.
Stage 2: feature table + transparent rule baseline.
Stage 3: grouped nested-CV model on those features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump

from oos_review import paths
from oos_review.baseline import apply_baseline, to_submission as baseline_to_submission
from oos_review.features import build_t0_t1_features
from oos_review.linker import PersonIndex, link_t0_sources, link_t1_stream
from oos_review.load import (
    load_candidates,
    load_labels,
    load_linked_bundle,
    load_t0_sources,
    load_t1_stream,
    read_csv,
)
from oos_review.model import (
    apply_model,
    fit_model,
    labeled_frame,
    logistic_coefficients,
    majority_params,
    nested_cv,
    to_submission as model_to_submission,
)


def run_linkage(*, save: bool = True, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Link T0 sources and the T1 update stream to candidate_record_id."""
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
    """Build T0/T1 features and apply the rule baseline."""
    if bundle is None:
        bundle = load_linked_bundle()
    features = build_t0_t1_features(bundle["candidates"], bundle)
    preds = apply_baseline(features)

    if save:
        paths.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        paths.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        features.to_csv(paths.FEATURES_DIR / "case_features.csv", index=False)
        preds.to_csv(paths.BASELINE_DIR / "case_predictions_audit.csv", index=False)
        baseline_to_submission(preds).to_csv(
            paths.BASELINE_DIR / "case_predictions.csv", index=False
        )

    return features, preds


def run_model(
    features: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    *,
    save: bool = True,
    search: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested CV, then fit on all labels and score every case.

    Returns ``(predictions, cv_fold_metrics)``.
    """
    if features is None or baseline is None:
        features, baseline = run_features_and_baseline(save=save)
    labels = load_labels()
    cv = nested_cv(features, labels, search=search)
    estimator = fit_model(features, labels, params=majority_params(cv))
    preds = apply_model(features, estimator, baseline=baseline)
    template = read_csv(paths.SUBMISSION_TEMPLATE)
    submission = model_to_submission(preds, template=template)

    if save:
        paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_csv(paths.MODEL_DIR / "case_predictions_audit.csv", index=False)
        submission.to_csv(paths.MODEL_DIR / "case_predictions.csv", index=False)
        submission.to_csv(paths.PROJECT_ROOT / "case_predictions.csv", index=False)
        cv.fold_metrics.to_csv(paths.MODEL_DIR / "cv_fold_metrics.csv", index=False)
        dump(estimator, paths.MODEL_DIR / "hgb_pipeline.joblib")
        logistic_coefficients(features, labels).to_csv(
            paths.MODEL_DIR / "logistic_coefficients.csv", index=False
        )
        oof = labeled_frame(features, labels)[["candidate_record_id", "phase", "y"]].copy()
        oof["oof_predicted_class"] = cv.oof_pred
        oof["p_review_warranted"] = cv.oof_proba[:, 0]
        oof["p_review_not_warranted"] = cv.oof_proba[:, 1]
        oof["p_insufficient_evidence"] = cv.oof_proba[:, 2]
        oof.to_csv(paths.MODEL_DIR / "oof_predictions.csv", index=False)

    return preds, cv.fold_metrics


def run_pipeline(*, save: bool = True, search: bool = True):
    """Linkage, features/baseline, then the model."""
    bundle = run_linkage(save=save)
    features, baseline = run_features_and_baseline(bundle, save=save)
    preds, cv_metrics = run_model(features, baseline, save=save, search=search)
    return bundle, features, baseline, preds, cv_metrics
