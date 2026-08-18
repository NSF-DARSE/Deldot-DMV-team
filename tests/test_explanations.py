from src.explanations.rules import deterministic_explanation


def test_explanation_is_deterministic_and_non_causal():
    features = {
        "active_credential_latest_non_de": 1,
        "active_credential_latest_state": "PA",
        "title_non_de_records_180d": 1,
        "title_latest_state": "PA",
        "repeated_non_de_address_evidence": 1,
        "external_standard_non_de_count": 1,
        "max_sources_same_non_de_state": 3,
        "dominant_evidence_state": "PA",
        "de_non_de_conflict": 0,
        "cross_source_agreement_score": 0.8,
        "conflicting_evidence_score": 0.1,
        "missing_evidence_categories": 0,
        "t1_reinforces_t0": 1,
        "t1_contradicts_t0": 0,
        "t1_resolves_uncertainty": 0,
    }
    prediction = {"predicted_class": "review_warranted"}
    first = deterministic_explanation(features, prediction, "T1")
    second = deterministic_explanation(features, prediction, "T1")
    assert first == second
    assert "not a legal or enforcement finding" in first["summary"]
    assert any("active credential" in bullet for bullet in first["evidence_bullets"])


def test_insufficient_evidence_explanation_names_conflict():
    result = deterministic_explanation(
        {
            "missing_evidence_categories": 0,
            "conflicting_evidence_score": 0.8,
            "cross_source_agreement_score": 0.3,
        },
        {"predicted_class": "insufficient_evidence"},
        "T0",
    )
    assert "conflict" in result["summary"]

