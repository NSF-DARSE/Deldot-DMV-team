from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_compact_manifest_is_label_free_and_has_no_identity_fields():
    manifest = json.loads((ROOT / "oos_review" / "configs" / "compact_model_features_v1.json").read_text())
    assert manifest["selection_policy"]["labels_used_to_choose_manifest"] is False
    names = [
        entry["name"]
        for key in (
            "t0_features",
            "t1_update_features",
            "t0_audit_only_review_signals",
            "t1_audit_only_review_signals",
        )
        for entry in manifest[key]
    ]
    banned = ("name", "dob", "birth", "street", "email", "phone", "label", "outcome")
    assert not any(any(term in name.lower() for term in banned) for name in names)


def test_compact_outputs_match_manifest_when_present():
    output = ROOT / "data" / "outputs" / "model_features_v1"
    if not output.exists():
        return
    manifest = json.loads((output / "compact_model_features_v1.json").read_text())
    t0 = pd.read_csv(output / "t0_compact_features.csv")
    t1 = pd.read_csv(output / "t1_compact_update_features.csv")
    t0_names = [entry["name"] for entry in manifest["t0_features"]]
    t1_names = [entry["name"] for entry in manifest["t1_update_features"]]
    t0_audit_names = [entry["name"] for entry in manifest["t0_audit_only_review_signals"]]
    t1_audit_names = [entry["name"] for entry in manifest["t1_audit_only_review_signals"]]
    audit_t0 = pd.read_csv(output / "t0_audit_only_review_signals.csv")
    audit_t1 = pd.read_csv(output / "t1_audit_only_review_signals.csv")
    assert t0.columns.tolist() == ["candidate_record_id", *t0_names]
    assert t1.columns.tolist() == ["candidate_record_id", *t1_names]
    assert audit_t0.columns.tolist() == ["candidate_record_id", *t0_audit_names]
    assert audit_t1.columns.tolist() == ["candidate_record_id", *t1_audit_names]
    assert len(t0) == len(t1) == 12_000
    assert t0.candidate_record_id.is_unique
    assert t1.candidate_record_id.is_unique
    assert set(t0.candidate_record_id) == set(t1.candidate_record_id)
    assert not [column for column in t0_names if t0[column].nunique(dropna=False) <= 1]
    assert not [column for column in t1_names if t1[column].nunique(dropna=False) <= 1]
