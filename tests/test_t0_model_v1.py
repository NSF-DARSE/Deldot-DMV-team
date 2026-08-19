from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling_v1.t0 import apply_temperature, fit_temperature, metric_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_temperature_scaling_preserves_probability_simplex():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    scaled = apply_temperature(probabilities, 1.7)
    assert np.all((scaled >= 0.0) & (scaled <= 1.0))
    assert np.allclose(scaled.sum(axis=1), 1.0)


def test_temperature_fit_is_positive_and_metrics_are_finite():
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]]
    )
    y = np.array([0, 1, 2, 0])
    temperature = fit_temperature(probabilities, y)
    metrics = metric_bundle(y, apply_temperature(probabilities, temperature))
    assert temperature > 0
    assert all(np.isfinite(value) for value in metrics.values())


def test_t0_outputs_are_valid_when_present():
    output = ROOT / "outputs" / "t0_model_v1"
    if not output.exists():
        return
    predictions = pd.read_csv(output / "t0_predictions.csv")
    oof = pd.read_csv(output / "t0_oof_predictions.csv")
    priors = pd.read_csv(output / "t0_training_priors.csv")
    diagnostics = json.loads((output / "t0_model_diagnostics.json").read_text())
    importance = pd.read_csv(output / "t0_feature_importance.csv")
    probability_columns = [
        "p_review_warranted",
        "p_review_not_warranted",
        "p_insufficient_evidence",
    ]
    assert len(predictions) == len(priors) == 12_000
    assert len(oof) == 300
    assert predictions.candidate_record_id.is_unique
    assert priors.candidate_record_id.is_unique
    assert np.allclose(predictions[probability_columns].sum(axis=1), 1.0)
    assert np.allclose(priors[probability_columns].sum(axis=1), 1.0)
    assert predictions[probability_columns].ge(0).all().all()
    assert predictions[probability_columns].le(1).all().all()
    assert predictions.review_priority.equals(predictions.p_review_warranted)
    assert (priors.prediction_origin == "outer_oof").sum() == 300
    assert diagnostics["oof_training_prior_count"] == 300
    assert len(importance) == diagnostics["feature_count"] == 24
    assert importance["feature"].is_unique
    assert importance["importance"].ge(0).all()
