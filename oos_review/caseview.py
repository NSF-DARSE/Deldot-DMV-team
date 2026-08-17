"""Pull the linked evidence rows that belong to one candidate."""

from __future__ import annotations

import pandas as pd


def rows_for_candidate(
    linked: pd.DataFrame,
    candidate_record_id: str,
) -> pd.DataFrame:
    if "candidate_record_id" not in linked.columns:
        raise KeyError("DataFrame was not produced by the linker")
    return linked.loc[linked["candidate_record_id"] == candidate_record_id].copy()


def dossier(
    candidate_record_id: str,
    candidates: pd.DataFrame,
    linked_sources: dict[str, pd.DataFrame],
    labels: pd.DataFrame | None = None,
) -> dict:
    """Return a JSON-serializable case file for notebooks and audit logs."""
    person = candidates.loc[
        candidates["candidate_record_id"] == candidate_record_id
    ]
    if person.empty:
        raise KeyError(f"Unknown candidate_record_id: {candidate_record_id}")

    payload = {
        "candidate": person.iloc[0].to_dict(),
        "label": None,
        "evidence": {},
        "link_counts": {},
    }
    if labels is not None:
        hit = labels.loc[labels["candidate_record_id"] == candidate_record_id]
        if not hit.empty:
            payload["label"] = hit.iloc[0].to_dict()

    for name, frame in linked_sources.items():
        subset = rows_for_candidate(frame, candidate_record_id)
        payload["evidence"][name] = subset
        payload["link_counts"][name] = int(len(subset))
    return payload


def linkage_summary(linked: pd.DataFrame) -> pd.DataFrame:
    """One-row coverage table for a linked source file."""
    total = len(linked)
    attached = linked["candidate_record_id"].notna()
    rule_counts = linked["match_rule"].value_counts()
    return pd.DataFrame(
        {
            "rows": [total],
            "linked": [int(attached.sum())],
            "linked_share": [float(attached.mean()) if total else 0.0],
            "unlinked": [int((linked["match_rule"] == "unlinked").sum())],
            "ambiguous": [
                int((linked["match_rule"] == "ambiguous_unassigned").sum())
            ],
            "identity": [int(rule_counts.get("identity", 0))],
            "dob_prefix": [int(rule_counts.get("dob_prefix", 0))],
            "dob_initial": [int(rule_counts.get("dob_initial", 0))],
            "name_exact": [int(rule_counts.get("name_exact", 0))],
            "name_prefix": [int(rule_counts.get("name_prefix", 0))],
            "vehicle_ref": [int(rule_counts.get("vehicle_ref", 0))],
        }
    )
