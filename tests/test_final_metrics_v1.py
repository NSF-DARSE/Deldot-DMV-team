from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_final_metrics_outputs_when_present():
    output = ROOT / "data" / "outputs" / "final_metrics_v1"
    if not (output / "final_metrics_diagnostics.json").exists():
        return
    diagnostics = json.loads((output / "final_metrics_diagnostics.json").read_text())
    metrics = pd.read_csv(output / "final_oof_metrics.csv")
    per_class = pd.read_csv(output / "final_per_class_metrics.csv")
    drift = pd.read_csv(output / "labeled_unlabeled_feature_drift.csv")
    contract = json.loads((output / "real_data_feature_contract.json").read_text())
    submission = pd.read_csv(output / "case_predictions.csv")
    assert set(metrics.phase) == {"T0", "T1"}
    assert len(per_class) == 6
    assert len(drift) == 49
    assert len(contract["t0_features"]) == 24
    assert len(contract["t1_features"]) == 25
    assert submission.columns.tolist() == [
        "candidate_record_id",
        "phase",
        "predicted_class",
        "p_review_warranted",
        "p_review_not_warranted",
        "p_insufficient_evidence",
        "review_priority",
    ]
    assert len(submission) == 24_000
    assert not submission.isna().any().any()
    assert diagnostics["combined_prediction_rows"] == 24_000
    assert diagnostics["unique_candidate_phase_rows"] == 24_000
    assert diagnostics["probability_sum_max_error"] < 1e-12
    assert diagnostics["update_behavior_decision"] == "retain_current_unanchored_t1"
    assert diagnostics["anchoring_applied"] is False
    assert diagnostics["release_status"] == "synthetic_prototype_ready_real_data_validation_required"
