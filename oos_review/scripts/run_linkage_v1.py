from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from oos_review.paths import (
    BASELINE,
    CHALLENGE_DATA,
    CONFIGS,
    DASHBOARD_DATA,
    OUTPUTS,
    PACKAGE_ROOT,
    REPO_ROOT as ROOT,
    ensure_import_path,
)

ensure_import_path()


from linkage_v1 import LinkagePipeline


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted((PACKAGE_ROOT / "linkage_v1").glob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        digest.update(str(path.relative_to(PACKAGE_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_report(output_dir: Path, diagnostics: dict, summary: pd.DataFrame) -> None:
    headers = list(summary.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in summary.itertuples(index=False, name=None):
        table_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines = [
        "# Frozen Linkage v1 Report",
        "",
        f"- Rule version: `{diagnostics['link_rule_version']}`",
        f"- Frozen rule SHA-256: `{diagnostics['rules_sha256']}`",
        f"- Frozen implementation SHA-256: `{diagnostics['implementation_sha256']}`",
        f"- Candidate records: {diagnostics['candidate_count']:,}",
        f"- Evidence/update records evaluated: {diagnostics['source_record_count']:,}",
        f"- Linked: {diagnostics['linked_record_count']:,} ({diagnostics['overall_link_rate']:.1%})",
        f"- Unresolved: {diagnostics['unresolved_record_count']:,}",
        "",
        "## Transferability safeguards",
        "",
        "- No row-order or repeated-block assumptions.",
        "- No labels, predictions, DE/OOS state, or model metrics used in linkage.",
        "- Dates are preserved for later features but are not identity evidence.",
        "- Aliases are learned only from strong DOB/name or address/name anchors.",
        "- Ambiguous and contradictory records remain unresolved instead of being forced.",
        "- Vehicle references propagate identity only when T0 ownership is unambiguous and the name is noncontradictory.",
        "- Link confidence is a deterministic rule-strength score, not an empirical probability.",
        "",
        "## T0 vehicle-reference bridge",
        "",
        f"- High-confidence name-linked anchor rows: {diagnostics['t0_vehicle_ref_bridge']['strong_anchor_rows']:,}",
        f"- Vehicle refs with at least one strong anchor: {diagnostics['t0_vehicle_ref_bridge']['vehicle_refs_with_strong_anchors']:,}",
        f"- Conflicting strong-anchor vehicle refs (abstained): {diagnostics['t0_vehicle_ref_bridge']['vehicle_refs_with_conflicting_strong_anchors']:,}",
        f"- Vehicle refs with any independently linked owner conflict (abstained): {diagnostics['t0_vehicle_ref_bridge']['vehicle_refs_with_any_linked_owner_conflict']:,}",
        f"- Previously unresolved T0 title rows recovered: {diagnostics['t0_vehicle_ref_bridge']['rows_recovered']:,}",
        f"- Leave-one-alias-out accepted reference rows: {diagnostics['vehicle_bridge_holdout_audit']['accepted_reference_rows']:,}",
        f"- Precision against held-out strong-name reference: {diagnostics['vehicle_bridge_holdout_audit']['precision_against_reference']:.1%}",
        "- The bridge uses only vehicle_ref and owner names; row order, labels, state, and dates are prohibited identity inputs.",
        "- This is an internal consistency audit, not authoritative ground truth.",
        "",
        "## Source results",
        "",
        *table_lines,
        "",
        "## Strong-anchor holdout audit",
        "",
        f"- Holdout reference rows: {diagnostics['anchor_holdout_audit']['holdout_rows']:,}",
        f"- Accepted holdout rows: {diagnostics['anchor_holdout_audit']['accepted_holdout_rows']:,}",
        f"- Precision against DOB-anchor reference: {diagnostics['anchor_holdout_audit']['precision_against_reference']:.1%}",
        f"- Coverage on DOB-anchor holdout: {diagnostics['anchor_holdout_audit']['coverage_on_reference_holdout']:.1%}",
        "- This is an internal consistency audit, not authoritative link accuracy; no ground-truth links were supplied.",
        "",
        "## Real DMV deployment requirements",
        "",
        "- Revalidate normalization and thresholds on a manually adjudicated, representative linkage sample.",
        "- Monitor false-link and missed-link rates separately by source, name pattern, and missingness pattern.",
        "- Keep raw PII in governed source systems; downstream feature tables should use candidate/source identifiers and linkage diagnostics.",
        "- Treat low coverage as uncertainty, not evidence that a candidate has no out-of-state activity.",
        "- Version and re-freeze the rules whenever source schemas or data-quality patterns change.",
        "",
        "## Freeze boundary",
        "",
        "Any threshold or rule change requires a new rule version, a new hash, and regeneration of every downstream artifact.",
    ]
    (output_dir / "linkage_freeze_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portable frozen linkage v1.")
    parser.add_argument(
        "--data-root", type=Path,
        default=CHALLENGE_DATA,
    )
    parser.add_argument(
        "--rules", type=Path,
        default=CONFIGS / "linkage_rules_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=OUTPUTS / "linkage_v1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LinkagePipeline(args.data_root, args.rules)
    pipeline.run()
    linked, unresolved, summary, methods, coverage = pipeline.outputs()
    diagnostics = pipeline.diagnostics()
    diagnostics["anchor_holdout_audit"] = pipeline.anchor_holdout_audit()
    diagnostics["vehicle_bridge_holdout_audit"] = pipeline.vehicle_bridge_holdout_audit()
    diagnostics["implementation_sha256"] = implementation_sha256()

    linked_path = args.output_dir / "linked_events.csv"
    unresolved_path = args.output_dir / "unresolved_or_ambiguous_events.csv"
    summary_path = args.output_dir / "linkage_summary.csv"
    methods_path = args.output_dir / "linkage_method_summary.csv"
    coverage_path = args.output_dir / "candidate_linkage_coverage.csv"
    linked.to_csv(linked_path, index=False)
    unresolved.to_csv(unresolved_path, index=False)
    summary.to_csv(summary_path, index=False)
    methods.to_csv(methods_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    frozen_rules_path = args.output_dir / "frozen_linkage_rules_v1.json"
    frozen_rules_path.write_bytes(args.rules.read_bytes())

    diagnostics["output_sha256"] = {
        path.name: file_sha256(path)
        for path in (linked_path, unresolved_path, summary_path, methods_path, coverage_path, frozen_rules_path)
    }
    (args.output_dir / "linkage_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir, diagnostics, summary)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
