"""Linker policy tests. These encode the matching rules, not the full dataset."""

import pandas as pd

from oos_review.linker import PersonIndex, apply_vehicle_ref_pass, link_frame
from oos_review.names import parse_person


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-A",
                "first_name": "SYNGIV-Nwzgpc",
                "last_name": "SYNFAM-Nspy",
                "date_of_birth": "SYNDOB-1982-10-14",
            },
            {
                "candidate_record_id": "CAN-B",
                "first_name": "SYNGIV-Uzlyyp",
                "last_name": "SYNFAM-Nspy",
                "date_of_birth": "SYNDOB-1990-01-02",
            },
            {
                "candidate_record_id": "CAN-C",
                "first_name": "SYNGIV-Alcvpc",
                "last_name": "SYNFAM-Alcvd",
                "date_of_birth": "SYNDOB-1975-03-03",
            },
        ]
    )


def _index() -> PersonIndex:
    return PersonIndex.from_candidates(_candidates())


def test_identity_match_with_dob():
    match = _index().match(
        parse_person("SYNGIV-Nwzgpc", "SYNFAM-Nspy", "SYNDOB-1982-10-14")
    )
    assert match is not None
    assert match.candidate_record_id == "CAN-A"
    assert match.match_rule == "identity"


def test_mixed_case_family_still_matches():
    match = _index().match(
        parse_person("SYNGIV-Nwzgpc", "SYNFAM-NSPY", "SYNDOB-1982-10-14")
    )
    assert match is not None
    assert match.candidate_record_id == "CAN-A"


def test_truncated_given_with_dob_matches_unique_person():
    match = _index().match(
        parse_person("SYNGIV-N", "SYNFAM-Nspy", "SYNDOB-1982-10-14")
    )
    assert match is not None
    assert match.candidate_record_id == "CAN-A"
    assert match.match_rule == "dob_initial"


def test_dob_mismatch_is_rejected_even_if_names_match():
    match = _index().match(
        parse_person("SYNGIV-Nwzgpc", "SYNFAM-Nspy", "SYNDOB-1990-01-02")
    )
    assert match is None


def test_family_prefix_is_not_a_match():
    """ALCV must not attach to candidate ALCVD."""
    match = _index().match(
        parse_person("SYNGIV-Alcvpc", "SYNFAM-Alcv", "SYNDOB-1975-03-03")
    )
    assert match is None


def test_one_char_given_without_dob_is_rejected_when_family_has_two_people():
    match = _index().match(parse_person("SYNGIV-N", "SYNFAM-Nspy"))
    assert match is None


def test_three_char_prefix_without_dob_is_accepted_when_unique():
    match = _index().match(parse_person("SYNGIV-Nwz", "SYNFAM-Nspy"))
    assert match is not None
    assert match.candidate_record_id == "CAN-A"
    assert match.match_rule == "name_prefix"


def test_exact_name_without_dob():
    match = _index().match(parse_person("SYNGIV-Uzlyyp", "SYNFAM-Nspy"))
    assert match is not None
    assert match.candidate_record_id == "CAN-B"
    assert match.match_rule == "name_exact"


def test_link_frame_adds_columns_and_keeps_unlinked_rows():
    source = pd.DataFrame(
        [
            {
                "source_record_id": "1",
                "first_name": "SYNGIV-Nwzgpc",
                "last_name": "SYNFAM-Nspy",
                "date_of_birth": "SYNDOB-1982-10-14",
            },
            {
                "source_record_id": "2",
                "first_name": "SYNGIV-Nobody",
                "last_name": "SYNFAM-Missing",
                "date_of_birth": "SYNDOB-2000-01-01",
            },
        ]
    )
    linked = link_frame(
        source,
        _index(),
        first_col="first_name",
        last_col="last_name",
        dob_col="date_of_birth",
    )
    assert linked.loc[0, "candidate_record_id"] == "CAN-A"
    assert pd.isna(linked.loc[1, "candidate_record_id"])
    assert linked.loc[1, "match_rule"] == "unlinked"
    assert len(linked) == 2


def test_vehicle_ref_fills_truncated_given_but_not_a_different_person():
    linked = pd.DataFrame(
        [
            {
                "owner_first_name": "SYNGIV-Nwzgpc",
                "owner_last_name": "SYNFAM-Nspy",
                "vehicle_ref": "VH-1",
                "candidate_record_id": "CAN-A",
                "match_rule": "name_exact",
                "match_score": 0.86,
                "given_relation": "exact",
                "overlap_len": 6,
                "n_matches": 1,
                "is_ambiguous": False,
            },
            {
                "owner_first_name": "SYNGIV-N",
                "owner_last_name": "SYNFAM-Nspy",
                "vehicle_ref": "VH-1",
                "candidate_record_id": pd.NA,
                "match_rule": "unlinked",
                "match_score": 0.0,
                "given_relation": "none",
                "overlap_len": 0,
                "n_matches": 0,
                "is_ambiguous": False,
            },
            {
                "owner_first_name": "SYNGIV-Uzlyyp",
                "owner_last_name": "SYNFAM-Nspy",
                "vehicle_ref": "VH-1",
                "candidate_record_id": pd.NA,
                "match_rule": "unlinked",
                "match_score": 0.0,
                "given_relation": "none",
                "overlap_len": 0,
                "n_matches": 0,
                "is_ambiguous": False,
            },
        ]
    )
    filled = apply_vehicle_ref_pass(
        linked,
        _index(),
        first_col="owner_first_name",
        last_col="owner_last_name",
    )
    assert filled.loc[1, "candidate_record_id"] == "CAN-A"
    assert filled.loc[1, "match_rule"] == "vehicle_ref"
    assert pd.isna(filled.loc[2, "candidate_record_id"])
    assert filled.loc[2, "match_rule"] == "unlinked"


def test_ambiguous_given_prefix_is_left_unassigned():
    candidates = pd.DataFrame(
        [
            {
                "candidate_record_id": "CAN-1",
                "first_name": "SYNGIV-Nwzgpc",
                "last_name": "SYNFAM-Nspy",
                "date_of_birth": "SYNDOB-1982-10-14",
            },
            {
                "candidate_record_id": "CAN-2",
                "first_name": "SYNGIV-Nwzxyz",
                "last_name": "SYNFAM-Nspy",
                "date_of_birth": "SYNDOB-1991-01-01",
            },
        ]
    )
    source = pd.DataFrame(
        [{"first_name": "SYNGIV-Nwz", "last_name": "SYNFAM-Nspy"}]
    )
    linked = link_frame(
        source,
        PersonIndex.from_candidates(candidates),
        first_col="first_name",
        last_col="last_name",
    )
    assert linked.loc[0, "match_rule"] == "ambiguous_unassigned"
    assert pd.isna(linked.loc[0, "candidate_record_id"])

