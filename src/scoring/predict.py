from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import VALID_CLASSES
from src.models.training import align_probabilities
from src.scoring.priority import review_priority


def predict_phase(bundle: dict, phase: str, features: pd.DataFrame) -> pd.DataFrame:
    model_features = features[bundle["feature_columns"]]
    model = bundle["models"][phase]
    probabilities = align_probabilities(model.predict_proba(model_features), model.classes_)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted = np.asarray(VALID_CLASSES)[np.argmax(probabilities, axis=1)]
    result = pd.DataFrame(
        {
            "candidate_record_id": features["candidate_record_id"],
            "phase": phase,
            "predicted_class": predicted,
            "p_review_warranted": probabilities[:, 0],
            "p_review_not_warranted": probabilities[:, 1],
            "p_insufficient_evidence": probabilities[:, 2],
            "review_priority": review_priority(probabilities, features),
        }
    )
    return result


def generate_predictions(
    model_path: Path,
    t0_path: Path,
    t1_path: Path,
    template_path: Path,
) -> tuple[pd.DataFrame, dict]:
    bundle = joblib.load(model_path)
    t0 = pd.read_csv(t0_path)
    t1 = pd.read_csv(t1_path)
    predictions = pd.concat(
        [predict_phase(bundle, "T0", t0), predict_phase(bundle, "T1", t1)],
        ignore_index=True,
    )
    template = pd.read_csv(template_path)[["candidate_record_id", "phase"]]
    ordered = template.merge(
        predictions,
        on=["candidate_record_id", "phase"],
        how="left",
        validate="one_to_one",
    )
    metadata = {
        "model_version": bundle["model_version"],
        "selected_model": bundle["selected_model"],
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": len(ordered),
    }
    return ordered, metadata

