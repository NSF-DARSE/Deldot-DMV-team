from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.audit.database import AuditStore, VALID_REVIEW_ACTIONS
from src.config import ARTIFACT_DIR, MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT, VALID_CLASSES
from src.explanations.rules import deterministic_explanation, linear_feature_contributions


DISCLAIMER = (
    "Decision support only. Recommendations prioritize human review and do not make "
    "legal, residency, registration, fee, or enforcement determinations."
)


class ReviewActionRequest(BaseModel):
    phase: Literal["T0", "T1"] = "T1"
    action: Literal[
        "confirm_for_further_review",
        "dismiss_from_review",
        "needs_more_information",
    ]
    notes: Optional[str] = Field(default=None, max_length=1000)


class Repository:
    def __init__(self) -> None:
        self.predictions = pd.read_csv(OUTPUT_DIR / "case_predictions.csv")
        self.features = {
            "T0": pd.read_csv(ARTIFACT_DIR / "features_t0.csv").set_index("candidate_record_id"),
            "T1": pd.read_csv(ARTIFACT_DIR / "features_t1.csv").set_index("candidate_record_id"),
        }
        self.timeline = pd.read_csv(
            ARTIFACT_DIR / "evidence_timeline.csv.gz",
            parse_dates=["event_date", "observed_date"],
        )
        self.bundle = joblib.load(MODEL_DIR / "model_bundle.joblib")
        self.metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
        metadata_path = OUTPUT_DIR / "prediction_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.prediction_lookup = self.predictions.set_index(["candidate_record_id", "phase"])
        self.candidate_ids = set(self.predictions["candidate_record_id"])
        self.audit = AuditStore(PROJECT_ROOT / "data" / "audit" / "review_audit.db")
        self.audit.initialize()

    def require_candidate(self, candidate_record_id: str) -> None:
        if candidate_record_id not in self.candidate_ids:
            raise HTTPException(status_code=404, detail="Candidate record not found")

    def phase_detail(self, candidate_record_id: str, phase: str) -> dict:
        self.require_candidate(candidate_record_id)
        prediction = self.prediction_lookup.loc[(candidate_record_id, phase)].to_dict()
        features = self.features[phase].loc[candidate_record_id]
        explanation = deterministic_explanation(features, prediction, phase)
        return {
            "candidate_record_id": candidate_record_id,
            "phase": phase,
            **{key: _native(value) for key, value in prediction.items()},
            "confidence": float(max(prediction[col] for col in (
                "p_review_warranted", "p_review_not_warranted", "p_insufficient_evidence"
            ))),
            "evidence_strength": float(features["evidence_strength_score"]),
            "evidence_recency": float(features["evidence_recency_score"]),
            "cross_source_agreement": float(features["cross_source_agreement_score"]),
            "conflict_score": float(features["conflicting_evidence_score"]),
            "dominant_evidence_state": str(features["dominant_evidence_state"]),
            "independent_source_types": int(features["independent_evidence_source_types"]),
            "new_t1_records": int(features["new_t1_record_count"]),
            "explanation": explanation,
            "disclaimer": DISCLAIMER,
        }


def _native(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


repository = Repository()
app = FastAPI(
    title="Delaware DMV Potential Out-of-State Tag Holder Review API",
    version="1.0.0",
    description=DISCLAIMER,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": repository.bundle["model_version"],
        "selected_model": repository.bundle["selected_model"],
        "candidate_count": len(repository.candidate_ids),
        "prediction_rows": len(repository.predictions),
        "decision_support_only": True,
        "disclaimer": DISCLAIMER,
    }


@app.get("/cases")
def cases(
    phase: Literal["T0", "T1"] = "T1",
    predicted_class: Optional[str] = None,
    min_priority: float = Query(0.0, ge=0, le=1),
    max_priority: float = Query(1.0, ge=0, le=1),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    if predicted_class is not None and predicted_class not in VALID_CLASSES:
        raise HTTPException(status_code=422, detail="Invalid predicted_class")
    frame = repository.predictions.loc[repository.predictions["phase"].eq(phase)].copy()
    if predicted_class:
        frame = frame.loc[frame["predicted_class"].eq(predicted_class)]
    frame = frame.loc[frame["review_priority"].between(min_priority, max_priority)]
    frame = frame.sort_values("review_priority", ascending=False)
    total = len(frame)
    page = frame.iloc[offset : offset + limit].copy()
    strengths = repository.features[phase]["evidence_strength_score"]
    page["evidence_strength"] = page["candidate_record_id"].map(strengths)
    probability_columns = [
        "p_review_warranted", "p_review_not_warranted", "p_insufficient_evidence"
    ]
    page["confidence"] = page[probability_columns].max(axis=1)
    return {
        "items": page.to_dict(orient="records"),
        "total": total,
        "limit": limit,
        "offset": offset,
        "phase": phase,
        "disclaimer": DISCLAIMER,
    }


@app.get("/priority")
def priority(
    phase: Literal["T0", "T1"] = "T1",
    limit: int = Query(100, ge=1, le=500),
    min_priority: float = Query(0.0, ge=0, le=1),
) -> dict:
    return cases(phase=phase, min_priority=min_priority, max_priority=1.0, limit=limit, offset=0)


@app.get("/cases/{candidate_record_id}")
def case(candidate_record_id: str) -> dict:
    t0 = repository.phase_detail(candidate_record_id, "T0")
    t1 = repository.phase_detail(candidate_record_id, "T1")
    return {
        "candidate_record_id": candidate_record_id,
        "t0": t0,
        "t1": t1,
        "changes": {
            "predicted_class_changed": t0["predicted_class"] != t1["predicted_class"],
            "p_review_warranted_change": t1["p_review_warranted"] - t0["p_review_warranted"],
            "priority_change": t1["review_priority"] - t0["review_priority"],
            "evidence_strength_change": t1["evidence_strength"] - t0["evidence_strength"],
            "new_evidence_records": t1["new_t1_records"],
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/cases/{candidate_record_id}/t0")
def case_t0(candidate_record_id: str) -> dict:
    return repository.phase_detail(candidate_record_id, "T0")


@app.get("/cases/{candidate_record_id}/t1")
def case_t1(candidate_record_id: str) -> dict:
    return repository.phase_detail(candidate_record_id, "T1")


@app.get("/cases/{candidate_record_id}/timeline")
def timeline(candidate_record_id: str) -> dict:
    repository.require_candidate(candidate_record_id)
    frame = repository.timeline.loc[
        repository.timeline["candidate_record_id"].eq(candidate_record_id)
    ].copy()
    safe_columns = [
        "phase_available", "effective_source_domain", "source_record_id", "event_date",
        "observed_date", "state", "event_type", "status", "quality", "vehicle_ref",
        "record_action", "match_confidence", "match_method", "identity_consistency",
    ]
    frame = frame[safe_columns].sort_values("event_date")
    frame["event_date"] = frame["event_date"].dt.strftime("%Y-%m-%d")
    frame["observed_date"] = frame["observed_date"].dt.strftime("%Y-%m-%d")
    frame = frame.where(pd.notna(frame), None)
    return {
        "candidate_record_id": candidate_record_id,
        "events": frame.to_dict(orient="records"),
        "disclaimer": DISCLAIMER,
    }


@app.get("/cases/{candidate_record_id}/explanation")
def explanation(
    candidate_record_id: str, phase: Literal["T0", "T1"] = "T1"
) -> dict:
    detail = repository.phase_detail(candidate_record_id, phase)
    feature_row = repository.features[phase].loc[[candidate_record_id]][
        repository.bundle["feature_columns"]
    ]
    contributions = linear_feature_contributions(
        repository.bundle["models"][phase], feature_row, detail["predicted_class"]
    )
    return {
        "candidate_record_id": candidate_record_id,
        "phase": phase,
        **detail["explanation"],
        "feature_contributions": contributions,
        "contribution_method": "calibrated logistic base-model coefficient × transformed feature value",
        "disclaimer": DISCLAIMER,
    }


@app.get("/cases/{candidate_record_id}/audit")
def case_audit(candidate_record_id: str) -> dict:
    repository.require_candidate(candidate_record_id)
    return repository.audit.case_audit(candidate_record_id)


@app.post("/cases/{candidate_record_id}/review-action")
def record_review_action(candidate_record_id: str, request: ReviewActionRequest) -> dict:
    repository.require_candidate(candidate_record_id)
    if request.action not in VALID_REVIEW_ACTIONS:
        raise HTTPException(status_code=422, detail="Invalid review action")
    return repository.audit.add_review_action(
        candidate_record_id, request.phase, request.action, request.notes
    )


@app.get("/model/metrics")
def model_metrics() -> dict:
    selected = repository.bundle["selected_model"]
    return {
        "model_version": repository.bundle["model_version"],
        "selected_model": selected,
        "selection": repository.metrics["selection"],
        "selected_model_metrics": repository.metrics["models"][selected],
        "calibration_note": "Sigmoid calibration was compared with uncalibrated baselines; isotonic was avoided because the labeled sample is small.",
        "decision_support_only": True,
        "disclaimer": DISCLAIMER,
    }
