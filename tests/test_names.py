"""Unit tests for synthetic name parsing and given-name relations."""

from oos_review.names import given_overlap_len, given_relation, parse_dob, parse_family, parse_given


def test_parse_is_case_insensitive():
    assert parse_given("SYNGIV-Nwzgpc") == "NWZGPC"
    assert parse_given("syngiv-nwzgpc") == "NWZGPC"
    assert parse_family("SYNFAM-Nspy") == "NSPY"
    assert parse_family("SYNFAM-NSPY") == "NSPY"


def test_parse_dob_strips_prefix():
    assert parse_dob("SYNDOB-1982-10-14") == "1982-10-14"
    assert parse_dob(None) is None


def test_exact_given_relation():
    assert given_relation("NWZGPC", "NWZGPC") == "exact"
    assert given_overlap_len("NWZGPC", "NWZGPC") == 6


def test_truncated_given_is_prefix():
    assert given_relation("N", "NWZGPC") == "prefix"
    assert given_relation("NWZGPC", "N") == "prefix"
    assert given_overlap_len("N", "NWZGPC") == 1


def test_unrelated_givens_do_not_match():
    assert given_relation("NWZGPC", "XZCRLY") == "none"
    assert given_overlap_len("NWZGPC", "XZCRLY") == 0


def test_empty_tokens_do_not_match():
    assert given_relation("", "NWZGPC") == "none"
    assert parse_given(None) == ""
    assert parse_family(float("nan")) == ""
