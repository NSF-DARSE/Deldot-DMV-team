"""Model matrix, submission shape, and a tiny grouped-CV smoke test."""

import pandas as pd

from oos_review.baseline import CLASSES, SUBMISSION_COLUMNS, apply_baseline
from oos_review.model import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    apply_model,
    fit_model,
    model_matrix,
    nested_cv,
    stack_labels,
    to_submission,
)


def _toy_features(n_people: int = 15) -> pd.DataFrame:
    rows = []
    states = ["DE", "PA", "MD"]
    for i in range(n_people):
        state = states[i % 3]
        score = {0: 4.0, 1: -4.0, 2: 0.0}[i % 3]
        for phase, bump in [("T0", 0.0), ("T1", 0.4 if i % 2 == 0 else -0.4)]:
            rows.append(
                {
                    "candidate_record_id": f"CAN-{i:03d}",
                    "phase": phase,
                    "observed_state": state,
                    "observed_is_de": state == "DE",
                    "candidate_observed_date": "2026-06-01",
                    "n_address": 2,
                    "latest_address_state": state,
                    "last_de_address_date": "2025-01-01",
                    "last_oos_address_date": "2024-01-01",
                    "address_recency_vote": 1 if score > 0 else -1 if score < 0 else 0,
                    "n_license": 2,
                    "latest_license_state": state,
                    "last_de_license_date": "2025-01-01",
                    "last_oos_license_date": "2024-01-01",
                    "license_recency_vote": 1 if score > 0 else -1 if score < 0 else 0,
                    "n_title": 2,
                    "latest_title_state": state,
                    "last_de_title_date": "2025-01-01",
                    "last_oos_title_date": "2024-01-01",
                    "title_recency_vote": 1 if score > 0 else -1 if score < 0 else 0,
                    "n_work": 1,
                    "latest_work_state": state,
                    "last_de_work_date": "2025-01-01",
                    "last_oos_work_date": "2024-01-01",
                    "work_recency_vote": 1 if score > 0 else -1 if score < 0 else 0,
                    "n_external": 1,
                    "latest_external_state": state,
                    "last_de_external_date": "2025-01-01",
                    "last_oos_external_date": "2024-01-01",
                    "external_recency_vote": 0,
                    "open_address_state": state,
                    "has_open_address": True,
                    "open_address_is_de": state == "DE",
                    "latest_license_status": "active",
                    "latest_license_is_de": state == "DE",
                    "has_active_de_license": state == "DE",
                    "latest_title_is_de": state == "DE",
                    "latest_work_is_de": state == "DE",
                    "latest_external_is_de": state == "DE",
                    "n_current_de_ties": 4 if state == "DE" else 0,
                    "n_current_oos_ties": 0 if state == "DE" else 3,
                    "has_state_conflict": False,
                    "n_distinct_current_states": 1,
                    "oos_observed_open_de": False,
                    "n_de_newer_sources": 4 if score > 0 else 0,
                    "n_oos_newer_sources": 4 if score < 0 else 0,
                    "n_recency_votes": 4,
                    "n_sources_present": 5,
                    "n_t1": 0 if phase == "T0" else 1,
                    "mean_match_score": 0.9,
                    "de_oos_score": score + bump,
                }
            )
    return pd.DataFrame(rows)


def _toy_labels(features: pd.DataFrame) -> pd.DataFrame:
    people = features["candidate_record_id"].drop_duplicates()
    rows = []
    cycle = list(CLASSES)
    for i, cid in enumerate(people):
        y = cycle[i % 3]
        rows.append({"candidate_record_id": cid, "label_t0": y, "label_t1": y})
    return pd.DataFrame(rows)


def test_stack_labels_has_two_rows_per_person():
    labels = _toy_labels(_toy_features())
    stacked = stack_labels(labels)
    assert stacked.groupby("candidate_record_id").size().eq(2).all()
    assert set(stacked["phase"]) == {"T0", "T1"}


def test_model_matrix_drops_ids_and_timestamps():
    X = model_matrix(_toy_features())
    assert "candidate_record_id" not in X.columns
    assert "last_de_address_date" not in X.columns
    for col in NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES:
        assert col in X.columns


def test_fit_apply_probabilities_sum_to_one_and_classes_are_valid():
    features = _toy_features()
    labels = _toy_labels(features)
    estimator = fit_model(
        features,
        labels,
        params={"min_samples_leaf": 2, "max_depth": 2, "max_iter": 20},
    )
    baseline = apply_baseline(features)
    preds = apply_model(features, estimator, baseline=baseline)
    total = (
        preds["p_review_warranted"]
        + preds["p_review_not_warranted"]
        + preds["p_insufficient_evidence"]
    )
    assert (total - 1).abs().max() < 1e-9
    assert set(preds["predicted_class"]).issubset(set(CLASSES))
    assert preds["model_agrees_with_rule"].notna().all()
    assert preds["review_priority"].between(0, 1).all()


def test_to_submission_follows_template_order():
    features = _toy_features(n_people=6)
    labels = _toy_labels(features)
    estimator = fit_model(
        features,
        labels,
        params={"min_samples_leaf": 2, "max_depth": 2, "max_iter": 20},
    )
    preds = apply_model(features, estimator)
    template = (
        features[["candidate_record_id", "phase"]].iloc[::-1].reset_index(drop=True)
    )
    sub = to_submission(preds, template=template)
    assert list(sub.columns) == list(SUBMISSION_COLUMNS)
    assert list(zip(sub.candidate_record_id, sub.phase)) == list(
        zip(template.candidate_record_id, template.phase)
    )


def test_nested_cv_without_search_runs_grouped():
    features = _toy_features(n_people=15)
    labels = _toy_labels(features)
    cv = nested_cv(features, labels, search=False)
    assert len(cv.fold_metrics) == 5
    assert cv.oof_proba.shape == (30, 3)
    assert len(cv.oof_pred) == 30
    assert cv.mean_accuracy >= 0
