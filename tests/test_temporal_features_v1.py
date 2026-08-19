from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from feature_prep_v1.builder import TemporalFeatureBuilder


ROOT = Path(__file__).resolve().parents[1]


def builder() -> TemporalFeatureBuilder:
    return TemporalFeatureBuilder(
        ROOT / "data" / "Identify_Out_of_State_Tag_Holders",
        ROOT / "data" / "outputs" / "linkage_v1" / "linked_events.csv",
        ROOT / "oos_review" / "configs" / "temporal_feature_rules_v1.json",
    )


def test_feature_artifacts_have_one_row_per_candidate_after_build():
    output = ROOT / "data" / "outputs" / "feature_prep_v1"
    if not output.exists():
        return
    t0 = pd.read_csv(output / "features_t0.csv")
    t1 = pd.read_csv(output / "features_t1.csv")
    delta = pd.read_csv(output / "features_t1_delta.csv")
    assert len(t0) == len(t1) == len(delta) == 12_000
    assert t0.candidate_record_id.is_unique
    assert t1.candidate_record_id.is_unique
    assert delta.candidate_record_id.is_unique


def test_old_oos_then_new_de_is_directional_not_current_oos():
    b = builder()
    candidate = b.candidates.iloc[0].copy()
    candidate["observed_state"] = "DE"
    candidate["_candidate_dt"] = pd.Timestamp("2025-06-01")
    events = b.events.iloc[:0].copy()
    rows = []
    for date, state in (("2024-01-01", "PA"), ("2025-05-01", "DE")):
        row = {column: "" for column in events.columns}
        row.update(
            {
                "_event_dt": pd.Timestamp(date), "_state_class": "DE" if state == "DE" else "OOS",
                "state": state, "_canonical_source": "address", "_confidence": 0.99,
                "link_tier": "A", "phase": "T0", "source": "address_history", "effective_end_date": "",
                "_end_dt": pd.NaT,
            }
        )
        rows.append(row)
    events = pd.DataFrame(rows, columns=events.columns)
    features = b._aggregate_candidate(candidate, "T0", events, pd.Timestamp("2025-06-01"), 0)
    assert features["de_after_oos"] == 1
    assert features["oos_after_de"] == 0
    assert features["oos_count_90d"] == 0
    assert features["recent_latest_source_oos_count_365d"] == 0
    assert features["de_evidence_after_latest_oos_count"] == 1


def test_oos_vehicle_signal_is_split_by_60_day_window():
    b = builder()
    events = b.events.iloc[:0].copy()
    rows = []
    for date in ("2025-01-20", "2025-03-15"):
        row = {column: "" for column in events.columns}
        row.update(
            {
                "_event_dt": pd.Timestamp(date), "_state_class": "OOS", "state": "NJ",
                "_canonical_source": "title", "vehicle_ref": "VH-1",
            }
        )
        rows.append(row)
    events = pd.DataFrame(rows, columns=events.columns)
    features = b._post_move_features(events, pd.Timestamp("2025-01-01"), "NJ")
    assert features["within_60d_oos_vehicle_signal_count"] == 1
    assert features["post_60d_oos_vehicle_signal_count"] == 1
    assert features["post_60d_oos_vehicle_signal_present"] == 1


def test_active_credential_and_title_combined_conflict_requires_explicit_active_status():
    b = builder()
    events = b.events.iloc[:0].copy()
    rows = []
    for source, date, state, status, event_type in (
        ("title", "2025-04-01", "NJ", "", "ownership_change"),
        ("license", "2025-04-02", "NJ", "active", "credential_update"),
        ("license", "2025-04-03", "NJ", "", "credential_update"),
    ):
        row = {column: "" for column in events.columns}
        row.update(
            {
                "_event_dt": pd.Timestamp(date), "_state_class": "OOS", "state": state,
                "_canonical_source": source, "credential_status": status,
                "event_type": event_type, "phase": "T0",
            }
        )
        rows.append(row)
    events = pd.DataFrame(rows, columns=events.columns)
    features = b._post_move_features(events, pd.Timestamp("2025-01-01"), "NJ")
    assert features["post_60d_active_oos_credential_signal_count"] == 1
    assert features["post_60d_combined_title_active_credential_conflict_present"] == 1
    assert features["post_60d_prior_state_persistence_cross_source_present"] == 1


def test_active_credential_signal_does_not_require_a_title_record():
    b = builder()
    events = b.events.iloc[:0].copy()
    row = {column: "" for column in events.columns}
    row.update(
        {
            "_event_dt": pd.Timestamp("2025-04-02"), "_state_class": "OOS", "state": "PA",
            "_canonical_source": "license", "credential_status": "active",
            "event_type": "credential_update", "phase": "T0",
        }
    )
    events = pd.DataFrame([row], columns=events.columns)
    features = b._post_move_features(events, pd.Timestamp("2025-01-01"), "PA")
    assert features["post_60d_oos_vehicle_signal_count"] == 0
    assert features["post_60d_active_oos_credential_signal_count"] == 1
    assert features["post_60d_combined_title_active_credential_conflict_present"] == 0
    assert features["post_60d_prior_state_persistence_any_present"] == 1


def test_t1_credential_update_without_status_is_not_assumed_active():
    b = builder()
    events = b.events.iloc[:0].copy()
    row = {column: "" for column in events.columns}
    row.update(
        {
            "_event_dt": pd.Timestamp("2025-04-02"), "_state_class": "OOS", "state": "NJ",
            "_canonical_source": "license", "credential_status": "",
            "event_type": "update_event", "phase": "T1",
        }
    )
    events = pd.DataFrame([row], columns=events.columns)
    features = b._post_move_features(events, pd.Timestamp("2025-01-01"), "NJ")
    assert features["post_60d_oos_credential_update_status_unknown_count"] == 1
    assert features["post_60d_active_oos_credential_signal_count"] == 0


def test_legal_rules_are_proxies_not_violation_labels():
    rules = json.loads((ROOT / "oos_review" / "configs" / "temporal_feature_rules_v1.json").read_text())
    assert rules["new_resident_vehicle_registration_window_days"] == 60
    assert "not" in rules["residency_proxy_policy"]["legal_status"].lower()
    assert "not" in rules["vehicle_signal_policy"]["legal_status"].lower()
