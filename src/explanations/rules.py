from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _value(features: pd.Series | dict[str, Any], key: str, default: float = 0.0) -> Any:
    value = features.get(key, default)
    return default if pd.isna(value) else value


def evidence_bullets(features: pd.Series | dict[str, Any], phase: str) -> list[str]:
    bullets: list[str] = []
    if _value(features, "active_credential_latest_non_de"):
        state = _value(features, "active_credential_latest_state", "non-Delaware")
        bullets.append(f"The latest active credential signal is associated with {state}.")
    if _value(features, "title_non_de_records_180d") > 0:
        state = _value(features, "title_latest_state", "a non-Delaware state")
        bullets.append(f"Recent vehicle-title activity points to {state}.")
    if _value(features, "repeated_non_de_address_evidence"):
        bullets.append("Repeated address observations contain non-Delaware state indicators.")
    if _value(features, "external_standard_non_de_count") > 0:
        bullets.append("At least one standard-quality external signal points outside Delaware.")
    if _value(features, "max_sources_same_non_de_state") >= 2:
        state = _value(features, "dominant_evidence_state", "the same non-Delaware state")
        bullets.append(f"Multiple independent source types align on {state}.")
    if _value(features, "latest_source_states_de") >= 3:
        bullets.append("Several source types have Delaware as their latest state indicator.")
    if _value(features, "de_non_de_conflict"):
        bullets.append("The evidence contains both Delaware and non-Delaware indicators.")
    if _value(features, "total_missing_state_records") > 0:
        bullets.append("Some linked evidence records do not contain a usable state value.")
    if _value(features, "low_identity_consistency_count") > 0:
        bullets.append("Some source identities vary materially, reducing evidence confidence.")
    if phase == "T1":
        if _value(features, "t1_reinforces_t0"):
            bullets.append("New T1 evidence reinforces the dominant T0 state pattern.")
        if _value(features, "t1_contradicts_t0"):
            bullets.append("New T1 evidence also contradicts part of the T0 state pattern.")
        if _value(features, "t1_resolves_uncertainty"):
            bullets.append("The later evidence reduces a previously observed source conflict.")
    return bullets[:6]


def deterministic_explanation(
    features: pd.Series | dict[str, Any], prediction: pd.Series | dict[str, Any], phase: str
) -> dict[str, Any]:
    predicted_class = str(prediction["predicted_class"])
    bullets = evidence_bullets(features, phase)
    agreement = float(_value(features, "cross_source_agreement_score"))
    conflict = float(_value(features, "conflicting_evidence_score"))
    missing = int(_value(features, "missing_evidence_categories"))
    non_de_sources = int(_value(features, "max_sources_same_non_de_state"))
    de_latest = int(_value(features, "latest_source_states_de"))

    if predicted_class == "review_warranted":
        if non_de_sources >= 2:
            text = (
                "This case was prioritized for staff review because multiple independent, "
                "recent evidence sources are associated with the same non-Delaware state. "
                "The pattern is an association for review, not a legal or enforcement finding."
            )
        else:
            text = (
                "This case was prioritized for staff review because the model found a combination "
                "of recent non-Delaware indicators. The evidence is associative and requires human verification."
            )
    elif predicted_class == "review_not_warranted":
        if de_latest >= 3:
            text = (
                "The current evidence does not warrant elevated review priority because several "
                "source types most recently point to Delaware. Staff may still review the underlying records."
            )
        else:
            text = (
                "The current combination of evidence is more consistent with the development cases "
                "that did not warrant elevated review. This is a queue recommendation, not a legal conclusion."
            )
    else:
        if missing > 0:
            reason = "one or more evidence categories are missing"
        elif conflict >= 0.5:
            reason = "available sources contain substantial Delaware/non-Delaware conflict"
        elif agreement < 0.5:
            reason = "the available source types do not agree strongly"
        else:
            reason = "the model probabilities remain too uncertain for a stronger recommendation"
        text = (
            f"The case is classified as insufficient evidence because {reason}. "
            "Additional human verification may clarify the record."
        )
    return {"summary": text, "evidence_bullets": bullets}


def linear_feature_contributions(
    model: Any,
    feature_row: pd.DataFrame,
    predicted_class: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return averaged pre-calibration linear associations when available."""
    pipelines = []
    if hasattr(model, "calibrated_classifiers_"):
        pipelines = [item.estimator for item in model.calibrated_classifiers_]
    elif hasattr(model, "named_steps"):
        pipelines = [model]
    contributions: dict[str, list[float]] = {}
    for pipeline in pipelines:
        if not hasattr(pipeline, "named_steps"):
            continue
        classifier = pipeline.named_steps.get("classifier")
        preprocess = pipeline.named_steps.get("preprocess")
        if classifier is None or preprocess is None or not hasattr(classifier, "coef_"):
            continue
        class_positions = np.where(classifier.classes_ == predicted_class)[0]
        if not len(class_positions):
            continue
        transformed = np.asarray(preprocess.transform(feature_row))[0]
        names = preprocess.get_feature_names_out()
        values = transformed * classifier.coef_[class_positions[0]]
        for name, value in zip(names, values):
            clean_name = str(name).replace("numeric__", "").replace("categorical__", "")
            contributions.setdefault(clean_name, []).append(float(value))
    averaged = [
        {"feature": name, "contribution": float(np.mean(values))}
        for name, values in contributions.items()
    ]
    averaged.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    for item in averaged[:limit]:
        item["direction"] = "supports predicted class" if item["contribution"] >= 0 else "opposes predicted class"
        item["note"] = "base linear association before probability calibration; not causal"
    return averaged[:limit]

