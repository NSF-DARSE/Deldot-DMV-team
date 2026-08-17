"""Transparent 3-class rule on ``de_oos_score``.

Decision
--------
The score is a weighted sum of recency votes plus a current-address bump
(see ``features.py``). Class cuts are chosen from the meaning of the weights,
not from a search on the 300 labels:

* ``n_sources_present < 2`` or ``n_recency_votes == 0`` → ``insufficient_evidence``
  (the file is too thin to take a side).
* ``de_oos_score >= 2.0`` → ``review_warranted``
  (at least one strong DE-newer source, or two weaker ones, possibly with a
  current DE address).
* ``de_oos_score <= -2.0`` → ``review_not_warranted``
  (symmetric OOS-current picture).
* otherwise → ``insufficient_evidence`` (mixed or weak).

Probabilities
-------------
Three logits sit at score centers +2.5 / 0 / -2.5. Softmax turns distance to
those centers into a distribution that sums to 1. Thin files get an extra
push toward ``insufficient_evidence``. This is not a fitted calibrator.

Priority
--------
``review_priority = 0.75 * p_review_warranted + 0.25 * p_insufficient_evidence``

Staff should see likely-warranted cases first, then uncertain ones, then
confident no-review cases last.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oos_review.features import DOMAIN_WEIGHTS, OPEN_ADDRESS_WEIGHT

CLASSES = (
    "review_warranted",
    "review_not_warranted",
    "insufficient_evidence",
)

SCORE_WARRANT_MIN = 2.0
SCORE_NOT_WARRANT_MAX = -2.0
MIN_SOURCES_PRESENT = 2
MIN_RECENCY_VOTES = 1

CENTER_WARRANTED = 2.5
CENTER_NOT_WARRANTED = -2.5
CENTER_INSUFFICIENT = 0.0
SOFTMAX_TEMPERATURE = 1.75
LOW_EVIDENCE_INSUFFICIENT_LOGIT = 2.0

PRIORITY_WARRANTED_WEIGHT = 0.75
PRIORITY_INSUFFICIENT_WEIGHT = 0.25


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits / temperature
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def _decide(row: pd.Series) -> str:
    thin = (row["n_sources_present"] < MIN_SOURCES_PRESENT) or (
        row["n_recency_votes"] < MIN_RECENCY_VOTES
    )
    if thin:
        return "insufficient_evidence"
    score = float(row["de_oos_score"])
    if score >= SCORE_WARRANT_MIN:
        return "review_warranted"
    if score <= SCORE_NOT_WARRANT_MAX:
        return "review_not_warranted"
    return "insufficient_evidence"


def explain_row(row: pd.Series) -> str:
    """One-line audit string a reviewer can read without opening code."""
    bits: list[str] = []
    for domain, weight in DOMAIN_WEIGHTS.items():
        vote = int(row.get(f"{domain}_recency_vote", 0) or 0)
        if vote == 1:
            bits.append(f"{domain} DE-newer ({weight:+.1f})")
        elif vote == -1:
            bits.append(f"{domain} OOS-newer ({-weight:+.1f})")
    if bool(row.get("open_address_is_de")):
        bits.append(f"current address DE ({OPEN_ADDRESS_WEIGHT:+.1f})")
    elif bool(row.get("has_open_address")):
        bits.append(f"current address OOS ({-OPEN_ADDRESS_WEIGHT:+.1f})")
    bits.append(f"score={float(row['de_oos_score']):+.1f}")
    bits.append(f"decision={row['predicted_class']}")
    return "; ".join(bits)


def apply_baseline(features: pd.DataFrame) -> pd.DataFrame:
    """Add class, probabilities, priority, and a reason string."""
    result = features.copy()
    result["predicted_class"] = result.apply(_decide, axis=1)

    scores = result["de_oos_score"].to_numpy(dtype=float)
    logits = np.column_stack(
        [
            -np.abs(scores - CENTER_WARRANTED),
            -np.abs(scores - CENTER_NOT_WARRANTED),
            -np.abs(scores - CENTER_INSUFFICIENT),
        ]
    )
    thin = (result["n_sources_present"] < MIN_SOURCES_PRESENT) | (
        result["n_recency_votes"] < MIN_RECENCY_VOTES
    )
    logits[thin.to_numpy(), 2] += LOW_EVIDENCE_INSUFFICIENT_LOGIT
    probs = _softmax(logits, SOFTMAX_TEMPERATURE)
    result["p_review_warranted"] = probs[:, 0]
    result["p_review_not_warranted"] = probs[:, 1]
    result["p_insufficient_evidence"] = probs[:, 2]
    # Guard against drift from rounding.
    total = (
        result["p_review_warranted"]
        + result["p_review_not_warranted"]
        + result["p_insufficient_evidence"]
    )
    result["p_review_warranted"] = result["p_review_warranted"] / total
    result["p_review_not_warranted"] = result["p_review_not_warranted"] / total
    result["p_insufficient_evidence"] = result["p_insufficient_evidence"] / total

    result["review_priority"] = (
        PRIORITY_WARRANTED_WEIGHT * result["p_review_warranted"]
        + PRIORITY_INSUFFICIENT_WEIGHT * result["p_insufficient_evidence"]
    ).clip(0, 1)

    result["rule_reason"] = result.apply(explain_row, axis=1)
    return result


SUBMISSION_COLUMNS = [
    "candidate_record_id",
    "phase",
    "predicted_class",
    "p_review_warranted",
    "p_review_not_warranted",
    "p_insufficient_evidence",
    "review_priority",
]


def to_submission(preds: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SUBMISSION_COLUMNS if c not in preds.columns]
    if missing:
        raise KeyError(f"Baseline output missing columns: {missing}")
    out = preds[SUBMISSION_COLUMNS].copy()
    out["predicted_class"] = pd.Categorical(
        out["predicted_class"], categories=list(CLASSES)
    )
    return out.sort_values(["candidate_record_id", "phase"]).reset_index(drop=True)
