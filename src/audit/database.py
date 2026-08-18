from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


VALID_REVIEW_ACTIONS = {
    "confirm_for_further_review",
    "dismiss_from_review",
    "needs_more_information",
}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class AuditStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_audit (
                    id INTEGER PRIMARY KEY,
                    candidate_record_id TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK (phase IN ('T0', 'T1')),
                    model_version TEXT NOT NULL,
                    prediction_timestamp TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    p_review_warranted REAL NOT NULL,
                    p_review_not_warranted REAL NOT NULL,
                    p_insufficient_evidence REAL NOT NULL,
                    review_priority REAL NOT NULL,
                    feature_snapshot_json TEXT NOT NULL,
                    evidence_summary_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_candidate_phase_version
                ON prediction_audit(candidate_record_id, phase, model_version)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS human_review_action (
                    id INTEGER PRIMARY KEY,
                    candidate_record_id TEXT NOT NULL,
                    phase TEXT NOT NULL CHECK (phase IN ('T0', 'T1')),
                    action TEXT NOT NULL,
                    notes TEXT,
                    action_timestamp TEXT NOT NULL,
                    demo_only INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_action_candidate_phase
                ON human_review_action(candidate_record_id, phase, action_timestamp)
                """
            )
            connection.execute("PRAGMA optimize")

    def seed_predictions(
        self,
        predictions: pd.DataFrame,
        features_by_phase: dict[str, pd.DataFrame],
        model_version: str,
        prediction_timestamp: str,
        evidence_summaries: dict[tuple[str, str], dict[str, Any]],
    ) -> int:
        feature_maps = {
            phase: frame.set_index("candidate_record_id") for phase, frame in features_by_phase.items()
        }
        rows = []
        for prediction in predictions.itertuples(index=False):
            feature_row = feature_maps[prediction.phase].loc[prediction.candidate_record_id]
            feature_json = json.dumps(
                {key: _json_value(value) for key, value in feature_row.items()},
                separators=(",", ":"),
                sort_keys=True,
            )
            summary_json = json.dumps(
                evidence_summaries[(prediction.candidate_record_id, prediction.phase)],
                separators=(",", ":"),
                sort_keys=True,
            )
            rows.append(
                (
                    prediction.candidate_record_id,
                    prediction.phase,
                    model_version,
                    prediction_timestamp,
                    prediction.predicted_class,
                    prediction.p_review_warranted,
                    prediction.p_review_not_warranted,
                    prediction.p_insufficient_evidence,
                    prediction.review_priority,
                    feature_json,
                    summary_json,
                )
            )
        with self.connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO prediction_audit (
                    candidate_record_id, phase, model_version, prediction_timestamp,
                    predicted_class, p_review_warranted, p_review_not_warranted,
                    p_insufficient_evidence, review_priority, feature_snapshot_json,
                    evidence_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return connection.total_changes - before

    def add_review_action(
        self, candidate_record_id: str, phase: str, action: str, notes: str | None = None
    ) -> dict[str, Any]:
        if action not in VALID_REVIEW_ACTIONS:
            raise ValueError(f"Invalid review action: {action}")
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO human_review_action (
                    candidate_record_id, phase, action, notes, action_timestamp, demo_only
                ) VALUES (?, ?, ?, ?, ?, 1)
                """,
                (candidate_record_id, phase, action, notes, timestamp),
            )
            action_id = cursor.lastrowid
        return {
            "id": action_id,
            "candidate_record_id": candidate_record_id,
            "phase": phase,
            "action": action,
            "notes": notes,
            "action_timestamp": timestamp,
            "demo_only": True,
            "message": "Workflow action recorded for the prototype; no enforcement action was triggered.",
        }

    def case_audit(self, candidate_record_id: str) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as connection:
            predictions = connection.execute(
                """
                SELECT candidate_record_id, phase, model_version, prediction_timestamp,
                       predicted_class, p_review_warranted, p_review_not_warranted,
                       p_insufficient_evidence, review_priority, evidence_summary_json
                FROM prediction_audit
                WHERE candidate_record_id = ?
                ORDER BY prediction_timestamp, phase
                """,
                (candidate_record_id,),
            ).fetchall()
            actions = connection.execute(
                """
                SELECT id, candidate_record_id, phase, action, notes, action_timestamp, demo_only
                FROM human_review_action
                WHERE candidate_record_id = ?
                ORDER BY action_timestamp
                """,
                (candidate_record_id,),
            ).fetchall()
        return {
            "predictions": [dict(row) for row in predictions],
            "human_review_actions": [dict(row) for row in actions],
        }

