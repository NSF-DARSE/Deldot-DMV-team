"""Feature construction and rule-baseline policy tests."""

import pandas as pd

from oos_review.baseline import apply_baseline
from oos_review.features import build_case_features, build_t0_t1_features


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-DE",
                "first_name": "SYNGIV-A",
                "last_name": "SYNFAM-A",
                "date_of_birth": "SYNDOB-1980-01-01",
                "observed_street_address": "SYNLOC-1",
                "observed_state": "PA",
                "candidate_observed_date": "2026-06-01",
                "review_status": "unreviewed",
            },
            {
                "candidate_record_id": "CAN-OOS",
                "first_name": "SYNGIV-B",
                "last_name": "SYNFAM-B",
                "date_of_birth": "SYNDOB-1981-01-01",
                "observed_street_address": "SYNLOC-2",
                "observed_state": "DE",
                "candidate_observed_date": "2026-06-01",
                "review_status": "unreviewed",
            },
            {
                "candidate_record_id": "CAN-THIN",
                "first_name": "SYNGIV-C",
                "last_name": "SYNFAM-C",
                "date_of_birth": "SYNDOB-1982-01-01",
                "observed_street_address": "SYNLOC-3",
                "observed_state": "MD",
                "candidate_observed_date": "2026-06-01",
                "review_status": "unreviewed",
            },
        ]
    )


def _sources() -> dict[str, pd.DataFrame]:
    addr = pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-DE",
                "state": "PA",
                "effective_start_date": "2022-01-01",
                "effective_end_date": "2024-01-01",
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-DE",
                "state": "DE",
                "effective_start_date": "2025-06-01",
                "effective_end_date": pd.NaT,
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-OOS",
                "state": "DE",
                "effective_start_date": "2021-01-01",
                "effective_end_date": "2023-01-01",
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-OOS",
                "state": "MD",
                "effective_start_date": "2025-01-01",
                "effective_end_date": pd.NaT,
                "match_score": 0.86,
            },
        ]
    )
    lic = pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-DE",
                "credential_state": "DE",
                "event_date": "2025-07-01",
                "credential_status": "active",
                "match_score": 1.0,
            },
            {
                "candidate_record_id": "CAN-OOS",
                "credential_state": "MD",
                "event_date": "2025-07-01",
                "credential_status": "active",
                "match_score": 1.0,
            },
        ]
    )
    ttl = pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-DE",
                "event_state": "PA",
                "event_date": "2022-06-01",
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-DE",
                "event_state": "DE",
                "event_date": "2026-01-01",
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-OOS",
                "event_state": "DE",
                "event_date": "2022-06-01",
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-OOS",
                "event_state": "MD",
                "event_date": "2026-01-01",
                "match_score": 0.86,
            },
        ]
    )
    wrk = pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-DE",
                "work_state": "DE",
                "observed_date": "2026-03-01",
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-OOS",
                "work_state": "MD",
                "observed_date": "2026-03-01",
                "match_score": 0.86,
            },
        ]
    )
    ext = pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-DE",
                "signal_state": "DE",
                "effective_date": "2025-12-01",
                "match_score": 0.86,
            },
            {
                "candidate_record_id": "CAN-OOS",
                "signal_state": "MD",
                "effective_date": "2025-12-01",
                "match_score": 0.86,
            },
        ]
    )
    t1 = pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-DE",
                "source_domain": "address",
                "record_action": "status_update",
                "state": "PA",
                "effective_date": "2026-07-01",
                "observed_date": "2026-08-09",
            }
        ]
    )
    return {
        "address_history": addr,
        "license_id_events": lic,
        "vehicle_title_events": ttl,
        "work_location_signals": wrk,
        "external_context_signals": ext,
        "evidence_update_stream": t1,
    }


def test_recency_votes_follow_newer_state():
    feats = build_case_features(_candidates(), _sources(), phase="T0").set_index(
        "candidate_record_id"
    )
    de = feats.loc["CAN-DE"]
    oos = feats.loc["CAN-OOS"]
    assert de["address_recency_vote"] == 1
    assert de["title_recency_vote"] == 1
    assert de["open_address_is_de"]
    assert de["de_oos_score"] > 2
    assert oos["address_recency_vote"] == -1
    assert oos["title_recency_vote"] == -1
    assert not oos["open_address_is_de"]
    assert oos["de_oos_score"] < -2


def test_thin_file_has_zero_votes():
    feats = build_case_features(_candidates(), _sources(), phase="T0").set_index(
        "candidate_record_id"
    )
    thin = feats.loc["CAN-THIN"]
    assert thin["n_sources_present"] == 0
    assert thin["n_recency_votes"] == 0
    assert thin["de_oos_score"] == 0


def test_baseline_classes_match_score_policy():
    feats = build_case_features(_candidates(), _sources(), phase="T0")
    preds = apply_baseline(feats).set_index("candidate_record_id")
    assert preds.loc["CAN-DE", "predicted_class"] == "review_warranted"
    assert preds.loc["CAN-OOS", "predicted_class"] == "review_not_warranted"
    assert preds.loc["CAN-THIN", "predicted_class"] == "insufficient_evidence"
    for cid in preds.index:
        total = (
            preds.loc[cid, "p_review_warranted"]
            + preds.loc[cid, "p_review_not_warranted"]
            + preds.loc[cid, "p_insufficient_evidence"]
        )
        assert abs(total - 1) < 1e-9
    assert "current address DE" in preds.loc["CAN-DE", "rule_reason"]
    assert preds.loc["CAN-DE", "review_priority"] > preds.loc["CAN-OOS", "review_priority"]


def test_t1_address_update_can_flip_current_address_and_vote():
    sources = _sources()
    t0 = build_case_features(_candidates(), sources, phase="T0").set_index(
        "candidate_record_id"
    )
    t1 = build_case_features(
        _candidates(), sources, phase="T1", t1_stream=sources["evidence_update_stream"]
    ).set_index("candidate_record_id")
    assert t0.loc["CAN-DE", "open_address_state"] == "DE"
    assert t1.loc["CAN-DE", "open_address_state"] == "PA"
    assert t1.loc["CAN-DE", "address_recency_vote"] == -1
    assert t1.loc["CAN-DE", "de_oos_score"] < t0.loc["CAN-DE", "de_oos_score"]


def test_stacked_features_have_two_phases_per_case():
    stacked = build_t0_t1_features(_candidates(), _sources())
    assert set(stacked["phase"]) == {"T0", "T1"}
    assert stacked.groupby("candidate_record_id").size().eq(2).all()
