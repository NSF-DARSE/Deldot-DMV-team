from __future__ import annotations

import re
import unicodedata
from datetime import datetime


_SYNTHETIC_PREFIXES = ("SYNGIV-", "SYNFAM-", "SYNNAME-", "SYNDOB-", "SYNLOC-")
_ADDRESS_TOKEN_MAP = {
    "STREET": "ST",
    "ROAD": "RD",
    "AVENUE": "AVE",
    "BOULEVARD": "BLVD",
    "DRIVE": "DR",
    "LANE": "LN",
    "COURT": "CT",
    "PLACE": "PL",
    "PARKWAY": "PKWY",
    "HIGHWAY": "HWY",
    "ROUTE": "RTE",
    "APARTMENT": "APT",
    "SUITE": "STE",
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
}


def _ascii_upper(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).upper()


def strip_known_synthetic_prefix(value: str) -> str:
    """Remove only documented reserved namespaces, never arbitrary prefixes."""
    for prefix in _SYNTHETIC_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def normalize_name(value: object) -> str:
    text = strip_known_synthetic_prefix(_ascii_upper(value))
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_dob(value: object) -> str:
    text = strip_known_synthetic_prefix(_ascii_upper(value))
    match = re.search(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", text)
    if not match:
        return ""
    candidate = "-".join(match.groups())
    try:
        return datetime.strptime(candidate, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return ""


def normalize_address(value: object) -> str:
    text = strip_known_synthetic_prefix(_ascii_upper(value))
    tokens = re.findall(r"[A-Z0-9]+", text)
    normalized = [_ADDRESS_TOKEN_MAP.get(token, token) for token in tokens]
    return " ".join(normalized)


def clean_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "nan" else text
