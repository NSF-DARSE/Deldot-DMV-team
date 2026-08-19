from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from linkage_v1.normalization import normalize_address, normalize_dob, normalize_name
from linkage_v1.resolver import CandidateIndex, LinkDecision, LinkagePipeline
from linkage_v1.similarity import jaro_winkler


ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / "configs" / "linkage_rules_v1.json").read_text())


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-A",
                "first_name": "Alice",
                "last_name": "Morrison",
                "date_of_birth": "1980-01-02",
                "observed_street_address": "10 North Main Street Apt 2",
            },
            {
                "candidate_record_id": "CAN-B",
                "first_name": "Alicia",
                "last_name": "Morrison",
                "date_of_birth": "1990-03-04",
                "observed_street_address": "20 Oak Road",
            },
            {
                "candidate_record_id": "CAN-C",
                "first_name": "John",
                "last_name": "Smith",
                "date_of_birth": "1975-05-06",
                "observed_street_address": "30 Pine Avenue",
            },
            {
                "candidate_record_id": "CAN-D",
                "first_name": "John",
                "last_name": "Smith",
                "date_of_birth": "1988-07-08",
                "observed_street_address": "40 Pine Avenue",
            },
        ]
    )


def test_normalization_is_generic_and_preserves_units():
    assert normalize_name("  Alí-ce ") == "ALICE"
    assert normalize_dob("SYNDOB-1980-01-02") == "1980-01-02"
    assert normalize_address("10 N. Main St., Apartment 2") == "10 N MAIN ST APT 2"


def test_jaro_winkler_known_examples():
    assert round(jaro_winkler("MARTHA", "MARHTA"), 3) == 0.961
    assert round(jaro_winkler("DIXON", "DICKSONX"), 3) == 0.813


def test_dob_and_name_anchor_beats_similar_name():
    index = CandidateIndex(candidates(), RULES)
    decision = index.resolve_dob_anchor("Alyce", "Morrison", "1980-01-02")
    assert decision.candidate_record_id == "CAN-A"
    assert decision.link_tier == "A"


def test_ambiguous_exact_name_is_not_forced():
    index = CandidateIndex(candidates(), RULES)
    decision = index.resolve_name("John", "Smith")
    assert decision.candidate_record_id is None
    assert decision.link_method == "ambiguous_exact_full_name"


def test_state_is_not_an_input_to_linkage_rules():
    serialized = json.dumps(RULES)
    assert RULES["policies"]["use_de_or_oos_state_for_identity"] is False
    assert "state" not in RULES["thresholds"]
    assert "DE" not in serialized and "OOS" not in serialized


def test_candidate_row_order_does_not_change_identity_decisions():
    original = CandidateIndex(candidates(), RULES)
    shuffled = CandidateIndex(candidates().sample(frac=1, random_state=17), RULES)
    queries = [
        ("Alyce", "Morrison", "1980-01-02"),
        ("John", "Smith", "1975-05-06"),
        ("John", "Smith", ""),
    ]
    for first, last, dob in queries:
        if dob:
            left = original.resolve_dob_anchor(first, last, dob)
            right = shuffled.resolve_dob_anchor(first, last, dob)
        else:
            left = original.resolve_name(first, last)
            right = shuffled.resolve_name(first, last)
        assert left.candidate_record_id == right.candidate_record_id
        assert left.link_method == right.link_method


def link_decision(candidate_id: str | None, confidence: float = 0.0) -> LinkDecision:
    return LinkDecision(
        candidate_id,
        confidence,
        "B" if candidate_id else "UNRESOLVED",
        "test_name_link" if candidate_id else "test_unresolved",
        1.0 if candidate_id else 0.0,
        1.0 if candidate_id else 0.0,
        1.0 if candidate_id else 0.0,
        0,
        0,
        1 if candidate_id else 0,
        1.0 if candidate_id else 0.0,
        "",
    )


def bridge_pipeline(rows: list[dict], decisions: list[LinkDecision]) -> LinkagePipeline:
    pipeline = object.__new__(LinkagePipeline)
    pipeline.rules = RULES
    pipeline.index = CandidateIndex(candidates(), RULES)
    pipeline.frames = {"vehicle_title_events": pd.DataFrame(rows)}
    pipeline.decisions = {"vehicle_title_events": decisions}
    return pipeline


def test_t0_vehicle_bridge_recovers_only_strong_name_supported_row():
    pipeline = bridge_pipeline(
        [
            {"vehicle_ref": "VH-1", "owner_first_name": "Alice", "owner_last_name": "Morrison"},
            {"vehicle_ref": "VH-1", "owner_first_name": "Alice", "owner_last_name": "Morrison"},
        ],
        [link_decision("CAN-A", 0.955), link_decision(None)],
    )
    pipeline._bridge_t0_vehicle_titles()
    recovered = pipeline.decisions["vehicle_title_events"][1]
    assert recovered.candidate_record_id == "CAN-A"
    assert recovered.link_method == "unanimous_t0_vehicle_ref_with_strong_name"
    assert pipeline._t0_vehicle_bridge_diagnostics["rows_recovered"] == 1


def test_t0_vehicle_bridge_abstains_when_strong_anchors_conflict():
    pipeline = bridge_pipeline(
        [
            {"vehicle_ref": "VH-1", "owner_first_name": "Alice", "owner_last_name": "Morrison"},
            {"vehicle_ref": "VH-1", "owner_first_name": "Alicia", "owner_last_name": "Morrison"},
            {"vehicle_ref": "VH-1", "owner_first_name": "Alice", "owner_last_name": "Morrison"},
        ],
        [link_decision("CAN-A", 0.955), link_decision("CAN-B", 0.955), link_decision(None)],
    )
    pipeline._bridge_t0_vehicle_titles()
    assert pipeline.decisions["vehicle_title_events"][2].candidate_record_id is None
    assert pipeline._t0_vehicle_bridge_diagnostics["vehicle_refs_with_conflicting_strong_anchors"] == 1


def test_t0_vehicle_bridge_abstains_on_any_independent_owner_conflict():
    pipeline = bridge_pipeline(
        [
            {"vehicle_ref": "VH-1", "owner_first_name": "Alice", "owner_last_name": "Morrison"},
            {"vehicle_ref": "VH-1", "owner_first_name": "Alicia", "owner_last_name": "Morrison"},
            {"vehicle_ref": "VH-1", "owner_first_name": "Alice", "owner_last_name": "Morrison"},
        ],
        [link_decision("CAN-A", 0.955), link_decision("CAN-B", 0.885), link_decision(None)],
    )
    pipeline._bridge_t0_vehicle_titles()
    assert pipeline.decisions["vehicle_title_events"][2].candidate_record_id is None
    assert pipeline._t0_vehicle_bridge_diagnostics["vehicle_refs_with_any_linked_owner_conflict"] == 1
