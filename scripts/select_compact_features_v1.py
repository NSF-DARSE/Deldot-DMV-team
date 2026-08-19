from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_names(manifest: dict, key: str) -> list[str]:
    return [entry["name"] for entry in manifest[key]]


def require_columns(frame: pd.DataFrame, columns: list[str], artifact: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{artifact} is missing manifest columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select compact label-free model feature matrices.")
    parser.add_argument(
        "--feature-dir", type=Path, default=ROOT / "outputs" / "feature_prep_v1"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs" / "compact_model_features_v1.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "model_features_v1"
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    t0 = pd.read_csv(args.feature_dir / "features_t0.csv")
    t1_delta = pd.read_csv(args.feature_dir / "features_t1_delta.csv")
    dictionary = pd.read_csv(args.feature_dir / "temporal_feature_dictionary.csv")
    t0_names = feature_names(manifest, "t0_features")
    t1_names = feature_names(manifest, "t1_update_features")
    t0_audit_names = feature_names(manifest, "t0_audit_only_review_signals")
    t1_audit_names = feature_names(manifest, "t1_audit_only_review_signals")
    require_columns(t0, ["candidate_record_id", *t0_names, *t0_audit_names], "features_t0.csv")
    require_columns(
        t1_delta,
        ["candidate_record_id", *t1_names, *t1_audit_names],
        "features_t1_delta.csv",
    )

    compact_t0 = t0[["candidate_record_id", *t0_names]].copy()
    compact_t1 = t1_delta[["candidate_record_id", *t1_names]].copy()
    audit_t0 = t0[["candidate_record_id", *t0_audit_names]].copy()
    audit_t1 = t1_delta[["candidate_record_id", *t1_audit_names]].copy()
    if compact_t0["candidate_record_id"].duplicated().any():
        raise ValueError("T0 compact matrix has duplicate candidate_record_id values")
    if compact_t1["candidate_record_id"].duplicated().any():
        raise ValueError("T1 compact matrix has duplicate candidate_record_id values")
    if set(compact_t0["candidate_record_id"]) != set(compact_t1["candidate_record_id"]):
        raise ValueError("T0 and T1 compact candidate sets differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0_path = args.output_dir / "t0_compact_features.csv"
    t1_path = args.output_dir / "t1_compact_update_features.csv"
    audit_t0_path = args.output_dir / "t0_audit_only_review_signals.csv"
    audit_t1_path = args.output_dir / "t1_audit_only_review_signals.csv"
    frozen_manifest = args.output_dir / "compact_model_features_v1.json"
    dictionary_path = args.output_dir / "compact_feature_dictionary.csv"
    compact_t0.to_csv(t0_path, index=False)
    compact_t1.to_csv(t1_path, index=False)
    audit_t0.to_csv(audit_t0_path, index=False)
    audit_t1.to_csv(audit_t1_path, index=False)
    frozen_manifest.write_bytes(args.manifest.read_bytes())

    selected = {
        "candidate_record_id",
        *t0_names,
        *t1_names,
        *t0_audit_names,
        *t1_audit_names,
    }
    compact_dictionary = dictionary[dictionary["column"].isin(selected)].copy()
    rationale = {
        entry["name"]: (entry["group"], entry["reason"])
        for key in (
            "t0_features",
            "t1_update_features",
            "t0_audit_only_review_signals",
            "t1_audit_only_review_signals",
        )
        for entry in manifest[key]
    }
    compact_dictionary["selection_group"] = compact_dictionary["column"].map(
        lambda column: rationale.get(column, ("metadata", "Stable join key."))[0]
    )
    compact_dictionary["selection_reason"] = compact_dictionary["column"].map(
        lambda column: rationale.get(column, ("metadata", "Stable join key."))[1]
    )
    model_names = set(t0_names) | set(t1_names)
    audit_names = set(t0_audit_names) | set(t1_audit_names)
    compact_dictionary["model_use"] = compact_dictionary["column"].map(
        lambda column: (
            "compact_model_predictor"
            if column in model_names
            else "audit_only_review_signal"
            if column in audit_names
            else "metadata"
        )
    )
    compact_dictionary.to_csv(dictionary_path, index=False)

    diagnostics = {
        "version": manifest["version"],
        "candidate_count": len(compact_t0),
        "t0_predictor_count": len(t0_names),
        "t1_update_predictor_count": len(t1_names),
        "t0_audit_only_review_signal_count": len(t0_audit_names),
        "t1_audit_only_review_signal_count": len(t1_audit_names),
        "labels_used_to_choose_manifest": manifest["selection_policy"][
            "labels_used_to_choose_manifest"
        ],
        "t0_all_missing_columns": [c for c in t0_names if compact_t0[c].isna().all()],
        "t1_all_missing_columns": [c for c in t1_names if compact_t1[c].isna().all()],
        "t0_constant_columns": [c for c in t0_names if compact_t0[c].nunique(dropna=False) <= 1],
        "t1_constant_columns": [c for c in t1_names if compact_t1[c].nunique(dropna=False) <= 1],
        "output_sha256": {},
    }
    for path in (
        t0_path,
        t1_path,
        audit_t0_path,
        audit_t1_path,
        frozen_manifest,
        dictionary_path,
    ):
        diagnostics["output_sha256"][path.name] = sha256_file(path)
    diagnostics_path = args.output_dir / "compact_feature_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    report = [
        "# Compact Model Feature Selection v1",
        "",
        f"- Candidates: {len(compact_t0):,}",
        f"- T0 predictors: {len(t0_names)} (reduced from {len(t0.columns) - 5} model-candidate columns)",
        f"- T1 update predictors: {len(t1_names)} (reduced from {len(t1_delta.columns) - 1})",
        f"- T0 audit-only review signals: {len(t0_audit_names)}",
        f"- T1 audit-only review signals: {len(t1_audit_names)}",
        "- Labels used to choose this manifest: no",
        "- PII or raw identity fields included: no",
        "",
        "## Leakage policy",
        "",
        "The compact manifest is fixed from domain and data-definition reasoning before reading labels. Any later supervised feature selection, constant removal, imputation, hyperparameter tuning, early stopping, and probability calibration must be fit using training-fold data only.",
        "Development candidate IDs, but not label values, were used for an unsupervised sparsity check. Extremely sparse legal-window interactions remain available as audit-only review signals and are excluded from CatBoost inputs.",
        "",
        "## T1 prior policy",
        "",
        "The T1 model matrix will append calibrated out-of-fold T0 class probabilities and T0 review priority during cross-validation. In-sample T0 predictions must never be used as T1 priors.",
        "",
        "## Runtime note",
        "",
        "This selection reads the frozen feature matrices directly and does not rerun linkage or per-candidate event aggregation.",
    ]
    report_path = args.output_dir / "compact_feature_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
