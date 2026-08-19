from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oos_review.scripts.train_t1_model_v1 import build_update_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_update_matrix_keeps_priority_for_audit():
    priors = pd.DataFrame(
        {
            "candidate_record_id": ["CAN-A"],
            "p_review_warranted": [0.6],
            "p_review_not_warranted": [0.2],
            "p_insufficient_evidence": [0.2],
            "review_priority": [0.6],
        }
    )
    deltas = pd.DataFrame({"candidate_record_id": ["CAN-A"], "new_t1_record_count": [2]})
    matrix = build_update_matrix(priors, deltas)
    assert matrix.loc[0, "p_review_warranted_t0"] == 0.6
    assert matrix.loc[0, "priority_t0"] == 0.6


def test_t1_outputs_are_valid_when_present():
    output = ROOT / "data" / "outputs" / "t1_model_v1"
    if not output.exists():
        return
    predictions = pd.read_csv(output / "t1_predictions.csv")
    oof = pd.read_csv(output / "t1_oof_predictions.csv")
    combined = pd.read_csv(output / "case_predictions_t0_t1.csv")
    training_matrix = pd.read_csv(output / "t1_training_update_matrix.csv")
    diagnostics = json.loads((output / "t1_model_diagnostics.json").read_text())
    probability_columns = [
        "p_review_warranted",
        "p_review_not_warranted",
        "p_insufficient_evidence",
    ]
    assert len(predictions) == 12_000
    assert len(oof) == 300
    assert len(combined) == 24_000
    assert predictions.candidate_record_id.is_unique
    assert not combined.duplicated(["candidate_record_id", "phase"]).any()
    assert combined.groupby("candidate_record_id").phase.nunique().eq(2).all()
    assert np.allclose(predictions[probability_columns].sum(axis=1), 1.0)
    assert predictions[probability_columns].ge(0).all().all()
    assert predictions[probability_columns].le(1).all().all()
    assert predictions.review_priority.equals(predictions.p_review_warranted)
    assert "priority_t0" in training_matrix.columns
    assert "priority_t0" not in diagnostics["predictor_names"]
    assert diagnostics["outer_folds_reused_from_t0"] is True
    assert diagnostics["priority_t0_excluded_as_duplicate"] is True
