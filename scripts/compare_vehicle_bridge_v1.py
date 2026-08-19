from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling_v1.t0 import metric_bundle
from scripts.generate_final_metrics_v1 import PROBABILITY_COLUMNS, priority_metrics


CLASS_ORDER = ["review_warranted", "review_not_warranted", "insufficient_evidence"]
LOWER_IS_BETTER = {"log_loss", "brier_multiclass", "ece_10bin"}


def phase_metrics(frame: pd.DataFrame) -> dict[str, float]:
    label_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    y = frame["actual_class"].map(label_to_index).to_numpy(dtype=int)
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    metrics = metric_bundle(y, probabilities)
    metrics.update(priority_metrics(frame["actual_class"], frame["review_priority"].to_numpy(), [0.1, 0.2]))
    return metrics


def aligned_predictions(baseline_path: Path, updated_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(baseline_path).sort_values("candidate_record_id").reset_index(drop=True)
    updated = pd.read_csv(updated_path).sort_values("candidate_record_id").reset_index(drop=True)
    identity_columns = ["candidate_record_id", "outer_fold", "actual_class"]
    if not baseline[identity_columns].equals(updated[identity_columns]):
        raise ValueError("Baseline and updated OOF predictions do not use identical candidates, folds, and labels.")
    return baseline, updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre/post T0 vehicle-ref bridge on identical OOF folds.")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "baseline_snapshot",
        help="Root containing the compact pre-bridge OOF/linkage snapshot bundled with the final package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "vehicle_ref_bridge_comparison_v1",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    point_rows = []
    bootstrap_rows = []
    folds_identical = True
    rng = np.random.default_rng(args.seed)
    metric_names = [
        "log_loss",
        "macro_f1",
        "brier_multiclass",
        "ece_10bin",
        "average_precision_review_warranted",
        "ndcg_review_warranted",
        "precision_at_top_10pct",
        "recall_at_top_20pct",
    ]

    for phase in ("T0", "T1"):
        baseline, updated = aligned_predictions(
            args.baseline_root / "outputs" / f"{phase.lower()}_model_v1" / f"{phase.lower()}_oof_predictions.csv",
            ROOT / "outputs" / f"{phase.lower()}_model_v1" / f"{phase.lower()}_oof_predictions.csv",
        )
        folds_identical &= baseline[["candidate_record_id", "outer_fold"]].equals(
            updated[["candidate_record_id", "outer_fold"]]
        )
        baseline_metrics = phase_metrics(baseline)
        updated_metrics = phase_metrics(updated)
        for metric in metric_names:
            delta = updated_metrics[metric] - baseline_metrics[metric]
            point_rows.append({
                "phase": phase,
                "metric": metric,
                "baseline": baseline_metrics[metric],
                "vehicle_bridge": updated_metrics[metric],
                "delta_vehicle_bridge_minus_baseline": delta,
                "point_direction": (
                    "improved"
                    if (delta < 0 if metric in LOWER_IS_BETTER else delta > 0)
                    else "unchanged" if delta == 0 else "worsened"
                ),
            })

        deltas = {metric: [] for metric in metric_names}
        for _ in range(args.bootstrap_resamples):
            indices = rng.integers(0, len(baseline), size=len(baseline))
            baseline_sample = baseline.iloc[indices]
            updated_sample = updated.iloc[indices]
            before = phase_metrics(baseline_sample)
            after = phase_metrics(updated_sample)
            for metric in metric_names:
                deltas[metric].append(after[metric] - before[metric])
        for metric, samples in deltas.items():
            values = np.asarray(samples, dtype=float)
            beneficial = values < 0 if metric in LOWER_IS_BETTER else values > 0
            bootstrap_rows.append({
                "phase": phase,
                "metric": metric,
                "mean_delta": float(values.mean()),
                "delta_ci_2_5": float(np.quantile(values, 0.025)),
                "delta_ci_97_5": float(np.quantile(values, 0.975)),
                "bootstrap_probability_beneficial": float(beneficial.mean()),
            })

    point = pd.DataFrame(point_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    point.to_csv(args.output_dir / "vehicle_bridge_metric_comparison.csv", index=False)
    bootstrap.to_csv(args.output_dir / "vehicle_bridge_paired_bootstrap.csv", index=False)

    baseline_linkage = pd.read_csv(
        args.baseline_root / "outputs" / "linkage_v1" / "linkage_summary.csv"
    )
    updated_linkage = pd.read_csv(ROOT / "outputs" / "linkage_v1" / "linkage_summary.csv")
    before_title = baseline_linkage.query("phase == 'T0' and source == 'vehicle_title_events'").iloc[0]
    after_title = updated_linkage.query("phase == 'T0' and source == 'vehicle_title_events'").iloc[0]
    linkage_diagnostics = json.loads(
        (ROOT / "outputs" / "linkage_v1" / "linkage_diagnostics.json").read_text()
    )
    bridge = linkage_diagnostics["t0_vehicle_ref_bridge"]
    audit = linkage_diagnostics["vehicle_bridge_holdout_audit"]

    lines = [
        "# T0 Vehicle-Reference Bridge: Before/After Comparison",
        "",
        "## Decision",
        "",
        "The naive vehicle-reference propagation is unsafe. The implemented linkage-v1.1.0 bridge is deliberately narrow: it requires unanimous high-confidence name anchors, no conflicting independently linked owner, strong agreement on both owner-name parts, and a positive runner-up margin.",
        "",
        f"- Identical OOF candidate/fold assignments: `{str(folds_identical).lower()}`",
        f"- T0 title linked rows: {int(before_title.linked_records):,} -> {int(after_title.linked_records):,} ({int(after_title.linked_records - before_title.linked_records):+,})",
        f"- T0 title coverage: {before_title.link_rate:.4%} -> {after_title.link_rate:.4%} ({after_title.link_rate - before_title.link_rate:+.4%})",
        f"- Strong name-linked anchor rows: {bridge['strong_anchor_rows']:,}",
        f"- Vehicle refs with strong anchors: {bridge['vehicle_refs_with_strong_anchors']:,}",
        f"- Conflicting strong-anchor refs abstained: {bridge['vehicle_refs_with_conflicting_strong_anchors']:,}",
        f"- Refs with any independently linked owner conflict abstained: {bridge['vehicle_refs_with_any_linked_owner_conflict']:,}",
        f"- Rows rejected by name safeguards: {bridge['rows_abstained_by_name_safeguards']:,}",
        f"- Leave-one-alias-out precision: {audit['precision_against_reference']:.2%} ({audit['correct_against_reference']:,}/{audit['accepted_reference_rows']:,}; internal reference, not authoritative truth)",
        "- Identity assignment uses only vehicle_ref and owner names. Row order, labels, state, and dates are not used.",
        "",
        "## OOF point metrics",
        "",
        "| Phase | Metric | Baseline | Bridge | Delta | Direction |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in point.itertuples(index=False):
        lines.append(
            f"| {row.phase} | {row.metric} | {row.baseline:.6f} | {row.vehicle_bridge:.6f} | {row.delta_vehicle_bridge_minus_baseline:+.6f} | {row.point_direction} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Log loss, macro-F1, and multiclass Brier improve slightly at both phases. T1 ECE improves, while T0 ECE worsens. Priority behavior is mixed: T0 AP rises slightly, T1 AP/NDCG/precision@10% fall, and T1 recall@20% rises. The paired bootstrap intervals in `vehicle_bridge_paired_bootstrap.csv` should be used to judge uncertainty; with only 300 labeled candidates, these point changes are not evidence of a broad predictive improvement.",
        "",
        "The bridge is retained because its linkage rule is conservative and its internal precision audit is clean, not because synthetic-label model metrics clearly dominate the baseline.",
    ])
    (args.output_dir / "vehicle_bridge_comparison_report.md").write_text("\n".join(lines) + "\n")

    diagnostics = {
        "version": "vehicle-ref-bridge-comparison-v1.0.0",
        "oof_folds_identical": folds_identical,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "t0_title_linked_before": int(before_title.linked_records),
        "t0_title_linked_after": int(after_title.linked_records),
        "t0_title_rows_recovered": int(after_title.linked_records - before_title.linked_records),
        "t0_title_link_rate_before": float(before_title.link_rate),
        "t0_title_link_rate_after": float(after_title.link_rate),
        "vehicle_bridge_holdout_audit": audit,
    }
    (args.output_dir / "vehicle_bridge_comparison_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
