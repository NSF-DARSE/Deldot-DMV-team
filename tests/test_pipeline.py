from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_submission import validate_submission
from src.config import ARTIFACT_DIR, OUTPUT_DIR, RAW_DATA_DIR, VALID_CLASSES


def test_feature_generation_outputs_one_row_per_candidate_and_phase():
    t0 = pd.read_csv(ARTIFACT_DIR / "features_t0.csv")
    t1 = pd.read_csv(ARTIFACT_DIR / "features_t1.csv")
    assert len(t0) == len(t1) == 12_000
    assert t0["candidate_record_id"].is_unique
    assert t1["candidate_record_id"].is_unique
    assert set(t0["candidate_record_id"]) == set(t1["candidate_record_id"])
    assert (t0["new_t1_record_count"] == 0).all()
    assert (t1["new_t1_record_count"] == 2).all()


def test_t1_has_change_features_and_preserves_t0():
    t0 = pd.read_csv(ARTIFACT_DIR / "features_t0.csv").set_index("candidate_record_id")
    t1 = pd.read_csv(ARTIFACT_DIR / "features_t1.csv").set_index("candidate_record_id")
    assert (t1["total_state_signal_count"] >= t0["total_state_signal_count"]).all()
    assert np.allclose(
        t1["change_in_evidence_strength"],
        t1["evidence_strength_score"] - t0["evidence_strength_score"],
    )


def test_submission_contract():
    validate_submission(
        OUTPUT_DIR / "case_predictions.csv",
        RAW_DATA_DIR / "Submission_Template.csv",
    )
    prediction = pd.read_csv(OUTPUT_DIR / "case_predictions.csv")
    assert set(prediction["predicted_class"]).issubset(set(VALID_CLASSES))
    assert not prediction.duplicated(["candidate_record_id", "phase"]).any()
    assert prediction["review_priority"].between(0, 1).all()
    assert np.allclose(
        prediction[
            ["p_review_warranted", "p_review_not_warranted", "p_insufficient_evidence"]
        ].sum(axis=1),
        1.0,
        atol=1e-8,
    )

