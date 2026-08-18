from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / np.log(clipped.shape[1])


def priority_components(
    probabilities: np.ndarray, features: pd.DataFrame
) -> dict[str, np.ndarray]:
    p_warranted = probabilities[:, 0]
    p_insufficient = probabilities[:, 2]
    strength = features["evidence_strength_score"].to_numpy(dtype=float)
    recency = features["evidence_recency_score"].to_numpy(dtype=float)
    agreement = features["cross_source_agreement_score"].to_numpy(dtype=float)
    conflict = features["conflicting_evidence_score"].to_numpy(dtype=float)
    uncertainty = normalized_entropy(probabilities)
    support = (0.45 * strength + 0.35 * recency + 0.20 * agreement).clip(0, 1)
    return {
        "probability_only": p_warranted.clip(0, 1),
        "evidence_supported": (
            0.78 * p_warranted
            + 0.14 * p_warranted * support
            + 0.08 * p_insufficient * (0.40 + 0.60 * uncertainty)
        ).clip(0, 1),
        "conflict_sensitive": (
            0.76 * p_warranted
            + 0.14 * p_warranted * support
            + 0.10 * p_insufficient * uncertainty * (0.50 + 0.50 * conflict)
        ).clip(0, 1),
        "recency_focused": (
            0.80 * p_warranted
            + 0.12 * p_warranted * recency
            + 0.08 * p_insufficient * uncertainty
        ).clip(0, 1),
    }


def review_priority(probabilities: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """Operational queue score; not a legal or violation probability."""
    return priority_components(probabilities, features)["evidence_supported"]


def evaluate_priority_alternatives(
    probabilities: np.ndarray, features: pd.DataFrame, actual: np.ndarray
) -> pd.DataFrame:
    positive = (actual == "review_warranted").astype(int)
    rows = []
    for name, scores in priority_components(probabilities, features).items():
        cutoff_count = max(1, int(np.ceil(0.20 * len(scores))))
        top = np.argsort(scores)[::-1][:cutoff_count]
        rows.append(
            {
                "formula": name,
                "average_precision_review_warranted": float(
                    average_precision_score(positive, scores)
                ),
                "precision_in_top_20_pct": float(positive[top].mean()),
                "mean_priority": float(scores.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["average_precision_review_warranted", "precision_in_top_20_pct"],
        ascending=False,
    )

