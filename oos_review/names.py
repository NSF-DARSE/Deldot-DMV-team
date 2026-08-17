"""Normalize synthetic identity tokens used across challenge files.

Why this module exists
----------------------
Source files do not carry ``candidate_record_id``. The only join keys are
synthetic names (and date of birth on license rows). Those values are noisy:

* mixed case (``SYNFAM-Nspy`` vs ``SYNFAM-NSPY``)
* truncated given names (``SYNGIV-N`` standing in for ``SYNGIV-Nwzgpc``)
* family names that look like prefixes of other family names (``ALCV`` vs
  ``ALCVD``) and must **not** be prefix-matched

This module only parses and compares tokens. Matching policy (when a parse is
good enough to attach a row to a candidate) lives in ``linker.py``.

Token layout
------------
Challenge names use a reserved prefix plus a payload:

* given:  ``SYNGIV-<payload>``
* family: ``SYNFAM-<payload>``
* other:  ``SYNNAME-<payload>`` (accepted, rare)
* DOB:    ``SYNDOB-YYYY-MM-DD``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

_GIVEN_PREFIXES = ("SYNGIV", "SYNNAME")
_FAMILY_PREFIXES = ("SYNFAM", "SYNNAME")
_DOB_PREFIX = "SYNDOB"

_PREFIX_RE = re.compile(r"^([A-Za-z]+)-(.+)$")
_DOB_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

GivenRelation = Literal["exact", "prefix", "none"]


@dataclass(frozen=True)
class ParsedName:
    """Uppercased identity tokens used for matching.

    Empty strings mean "token missing" and never match a candidate.
    """

    given: str
    family: str
    dob: Optional[str] = None


def _strip_prefix(value: object, allowed_prefixes: tuple[str, ...]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    match = _PREFIX_RE.match(text)
    if match:
        prefix, payload = match.group(1).upper(), match.group(2)
        if prefix in allowed_prefixes or prefix.startswith("SYN"):
            return payload.upper()
    return text.upper()


def parse_given(value: object) -> str:
    """Return the uppercased given-name payload, or ``''`` if missing."""
    return _strip_prefix(value, _GIVEN_PREFIXES)


def parse_family(value: object) -> str:
    """Return the uppercased family-name payload, or ``''`` if missing."""
    return _strip_prefix(value, _FAMILY_PREFIXES)


def parse_dob(value: object) -> Optional[str]:
    """Return ``YYYY-MM-DD`` from a ``SYNDOB-`` value, or ``None`` if missing."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if text.upper().startswith(_DOB_PREFIX + "-"):
        text = text[len(_DOB_PREFIX) + 1 :]
    match = _DOB_RE.match(text)
    return match.group(1) if match else text


def parse_person(
    first_name: object,
    last_name: object,
    date_of_birth: object = None,
) -> ParsedName:
    return ParsedName(
        given=parse_given(first_name),
        family=parse_family(last_name),
        dob=parse_dob(date_of_birth) if date_of_birth is not None else None,
    )


def given_relation(source_given: str, candidate_given: str) -> GivenRelation:
    """How two given-name payloads relate.

    * ``exact`` — identical after normalization
    * ``prefix`` — one is a leading substring of the other (truncation)
    * ``none`` — not compatible

    Family names are never compared here. Callers must require an exact
    family match before using this result.
    """
    if not source_given or not candidate_given:
        return "none"
    if source_given == candidate_given:
        return "exact"
    if candidate_given.startswith(source_given) or source_given.startswith(
        candidate_given
    ):
        return "prefix"
    return "none"


def given_overlap_len(source_given: str, candidate_given: str) -> int:
    """Shared leading length; 0 if the names are not prefix-compatible."""
    relation = given_relation(source_given, candidate_given)
    if relation == "none":
        return 0
    return min(len(source_given), len(candidate_given))
