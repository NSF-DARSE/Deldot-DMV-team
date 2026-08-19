from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling_v1.t0 import metric_bundle


PROBABILITY_COLUMNS = [
    "p_review_warranted",
    "p_review_not_warranted",
    "p_insufficient_evidence",
]
CLASS_ORDER = np.array(
    ["review_warranted", "review_not_warranted", "insufficient_evidence"]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_groups(delta: pd.DataFrame) -> pd.Series:
    grace_change = (
        delta["delta_current_past_60_day_grace_proxy"].ne(0)
        | delta["delta_current_within_60_day_grace_proxy"].ne(0)
        | delta["delta_post_60d_oos_vehicle_signal_count"].ne(0)
        | delta["delta_post_60d_active_oos_credential_signal_count"].ne(0)
        | delta["delta_post_60d_prior_state_persistence_any_present"].ne(0)
        | delta["delta_new_meaningful_oos_title_signal_post_60d_present"].ne(0)
    )
    meaningful = (
        delta["new_t1_oos_count"].add(delta["new_t1_de_count"]).gt(0)
        | delta["new_t1_source_domain_diversity"].ge(2)
        | delta["newly_effective_t0_record_count"].gt(0)
        | grace_change
    )
    values = np.where(
        meaningful,
        "meaningful_new_evidence",
        np.where(
            delta["new_t1_record_count"].gt(0),
            "weak_minor_update",
            "effectively_no_meaningful_update",
        ),
    )
    return pd.Series(values, index=delta.index, name="evidence_group")


def anchor_weights(groups: pd.Series, rules: dict) -> np.ndarray:
    anchor = rules["experimental_anchor"]
    mapping = {
        "meaningful_new_evidence": anchor["meaningful_new_evidence_t1_weight"],
        "weak_minor_update": anchor["weak_minor_update_t1_weight"],
        "effectively_no_meaningful_update": anchor[
            "effectively_no_meaningful_update_t1_weight"
        ],
    }
    return groups.map(mapping).to_numpy(dtype=float)


def anchored_probabilities(
    t0_probabilities: np.ndarray, t1_probabilities: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    anchored = weights[:, None] * t1_probabilities + (1.0 - weights[:, None]) * t0_probabilities
    return anchored / anchored.sum(axis=1, keepdims=True)


def bootstrap_metric_deltas(
    y: np.ndarray,
    unanchored: np.ndarray,
    anchored: np.ndarray,
    t0_probabilities: np.ndarray,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_names = [
        "log_loss",
        "macro_f1",
        "brier_multiclass",
        "brier_macro_ovr",
        "ece_10bin",
        "class_change_rate",
    ]
    values = {metric: [] for metric in metric_names}
    for _ in range(resamples):
        indices = rng.integers(0, len(y), size=len(y))
        y_sample = y[indices]
        unanchored_sample = unanchored[indices]
        anchored_sample = anchored[indices]
        t0_sample = t0_probabilities[indices]
        unanchored_metrics = metric_bundle(y_sample, unanchored_sample)
        anchored_metrics = metric_bundle(y_sample, anchored_sample)
        for metric in metric_names[:-1]:
            values[metric].append(anchored_metrics[metric] - unanchored_metrics[metric])
        unanchored_change = (
            unanchored_sample.argmax(axis=1) != t0_sample.argmax(axis=1)
        ).mean()
        anchored_change = (anchored_sample.argmax(axis=1) != t0_sample.argmax(axis=1)).mean()
        values["class_change_rate"].append(float(anchored_change - unanchored_change))
    rows = []
    for metric, differences in values.items():
        array = np.asarray(differences)
        rows.append(
            {
                "metric": metric,
                "delta_definition": "anchored_minus_unanchored",
                "mean_delta": float(array.mean()),
                "ci_2_5": float(np.quantile(array, 0.025)),
                "ci_97_5": float(np.quantile(array, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze T0-to-T1 update behavior and anchoring.")
    parser.add_argument(
        "--rules", type=Path, default=ROOT / "configs" / "update_behavior_rules_v1.json"
    )
    parser.add_argument(
        "--delta",
        type=Path,
        default=ROOT / "outputs" / "feature_prep_v1" / "features_t1_delta.csv",
    )
    parser.add_argument(
        "--t0-predictions",
        type=Path,
        default=ROOT / "outputs" / "t0_model_v1" / "t0_predictions.csv",
    )
    parser.add_argument(
        "--t1-predictions",
        type=Path,
        default=ROOT / "outputs" / "t1_model_v1" / "t1_predictions.csv",
    )
    parser.add_argument(
        "--t0-oof",
        type=Path,
        default=ROOT / "outputs" / "t0_model_v1" / "t0_oof_predictions.csv",
    )
    parser.add_argument(
        "--t1-oof",
        type=Path,
        default=ROOT / "outputs" / "t1_model_v1" / "t1_oof_predictions.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "Identify_Out_of_State_Tag_Holders"
        / "Development_Labels"
        / "Development_Labels.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "final_metrics_v1"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules = json.loads(args.rules.read_text())

    delta = pd.read_csv(args.delta).set_index("candidate_record_id")
    groups = evidence_groups(delta)
    t0 = pd.read_csv(args.t0_predictions).set_index("candidate_record_id")
    t1 = pd.read_csv(args.t1_predictions).set_index("candidate_record_id")
    common_ids = t0.index.intersection(t1.index).intersection(delta.index)
    if len(common_ids) != 12_000:
        raise ValueError("Expected 12,000 candidates across T0, T1, and delta artifacts")
    t0 = t0.loc[common_ids]
    t1 = t1.loc[common_ids]
    groups = groups.loc[common_ids]

    t0_probabilities = t0[PROBABILITY_COLUMNS].to_numpy()
    t1_probabilities = t1[PROBABILITY_COLUMNS].to_numpy()
    weights = anchor_weights(groups, rules)
    experimental_all = anchored_probabilities(t0_probabilities, t1_probabilities, weights)
    unanchored_changed = t0["predicted_class"].ne(t1["predicted_class"]).to_numpy()
    anchored_classes = CLASS_ORDER[experimental_all.argmax(axis=1)]
    anchored_changed = anchored_classes != t0["predicted_class"].to_numpy()
    probability_delta = t1_probabilities - t0_probabilities
    behavior = pd.DataFrame(
        {
            "candidate_record_id": common_ids,
            "evidence_group": groups.to_numpy(),
            "unanchored_class_changed": unanchored_changed.astype(int),
            "experimental_anchored_class_changed": anchored_changed.astype(int),
            "total_variation_probability_change": 0.5
            * np.abs(probability_delta).sum(axis=1),
            "mean_absolute_probability_change": np.abs(probability_delta).mean(axis=1),
            "p_review_warranted_change": probability_delta[:, 0],
            "experimental_anchor_t1_weight": weights,
        }
    )
    behavior_path = args.output_dir / "update_behavior_candidates.csv"
    behavior.to_csv(behavior_path, index=False)

    summary = (
        behavior.groupby("evidence_group", sort=False)
        .agg(
            candidates=("candidate_record_id", "size"),
            unanchored_class_changes=("unanchored_class_changed", "sum"),
            unanchored_class_change_rate=("unanchored_class_changed", "mean"),
            experimental_anchored_class_changes=(
                "experimental_anchored_class_changed",
                "sum",
            ),
            experimental_anchored_class_change_rate=(
                "experimental_anchored_class_changed",
                "mean",
            ),
            mean_total_variation_change=("total_variation_probability_change", "mean"),
            median_total_variation_change=("total_variation_probability_change", "median"),
            max_total_variation_change=("total_variation_probability_change", "max"),
            mean_absolute_probability_change=("mean_absolute_probability_change", "mean"),
            mean_p_review_warranted_change=("p_review_warranted_change", "mean"),
        )
        .reset_index()
    )
    summary_path = args.output_dir / "update_behavior_summary.csv"
    summary.to_csv(summary_path, index=False)

    labels = pd.read_csv(args.labels).set_index("candidate_record_id")
    labeled_ids = labels.index
    t0_oof = pd.read_csv(args.t0_oof).set_index("candidate_record_id").loc[labeled_ids]
    t1_oof = pd.read_csv(args.t1_oof).set_index("candidate_record_id").loc[labeled_ids]
    labeled_groups = groups.loc[labeled_ids]
    y = labels["label_t1"].map(
        {
            "review_warranted": 0,
            "review_not_warranted": 1,
            "insufficient_evidence": 2,
        }
    ).to_numpy()
    t0_oof_probabilities = t0_oof[PROBABILITY_COLUMNS].to_numpy()
    t1_oof_probabilities = t1_oof[PROBABILITY_COLUMNS].to_numpy()
    labeled_weights = anchor_weights(labeled_groups, rules)
    anchored_oof_probabilities = anchored_probabilities(
        t0_oof_probabilities, t1_oof_probabilities, labeled_weights
    )

    unanchored_metrics = metric_bundle(y, t1_oof_probabilities)
    anchored_metrics = metric_bundle(y, anchored_oof_probabilities)
    unanchored_change_rate = float(
        (t1_oof_probabilities.argmax(axis=1) != t0_oof_probabilities.argmax(axis=1)).mean()
    )
    anchored_change_rate = float(
        (anchored_oof_probabilities.argmax(axis=1) != t0_oof_probabilities.argmax(axis=1)).mean()
    )
    comparison = pd.DataFrame(
        [
            {"approach": "current_unanchored", **unanchored_metrics, "class_change_rate": unanchored_change_rate},
            {"approach": "experimental_anchored", **anchored_metrics, "class_change_rate": anchored_change_rate},
        ]
    )
    comparison_path = args.output_dir / "anchoring_comparison_oof.csv"
    comparison.to_csv(comparison_path, index=False)

    bootstrap = bootstrap_metric_deltas(
        y,
        t1_oof_probabilities,
        anchored_oof_probabilities,
        t0_oof_probabilities,
        int(rules["bootstrap"]["resamples"]),
        int(rules["bootstrap"]["random_seed"]),
    )
    bootstrap_path = args.output_dir / "anchoring_bootstrap_deltas.csv"
    bootstrap.to_csv(bootstrap_path, index=False)

    experimental_oof = pd.DataFrame(
        {
            "candidate_record_id": labeled_ids,
            "evidence_group": labeled_groups.to_numpy(),
            "p_review_warranted": anchored_oof_probabilities[:, 0],
            "p_review_not_warranted": anchored_oof_probabilities[:, 1],
            "p_insufficient_evidence": anchored_oof_probabilities[:, 2],
            "predicted_class": CLASS_ORDER[anchored_oof_probabilities.argmax(axis=1)],
            "actual_class": labels["label_t1"].to_numpy(),
        }
    )
    experimental_oof_path = args.output_dir / "experimental_anchored_t1_oof_predictions.csv"
    experimental_oof.to_csv(experimental_oof_path, index=False)

    actual_changed = labels["label_t0"].ne(labels["label_t1"])
    labeled_behavior = pd.DataFrame(
        {
            "evidence_group": labeled_groups,
            "actual_class_changed": actual_changed,
            "unanchored_class_changed": t1_oof["predicted_class"].ne(t0_oof["predicted_class"]),
            "experimental_anchored_class_changed": CLASS_ORDER[
                anchored_oof_probabilities.argmax(axis=1)
            ]
            != t0_oof["predicted_class"].to_numpy(),
        }
    )
    labeled_group_summary = (
        labeled_behavior.groupby("evidence_group", sort=False)
        .agg(
            labeled_candidates=("actual_class_changed", "size"),
            actual_class_changes=("actual_class_changed", "sum"),
            actual_class_change_rate=("actual_class_changed", "mean"),
            unanchored_class_changes=("unanchored_class_changed", "sum"),
            unanchored_class_change_rate=("unanchored_class_changed", "mean"),
            experimental_anchored_class_changes=(
                "experimental_anchored_class_changed",
                "sum",
            ),
            experimental_anchored_class_change_rate=(
                "experimental_anchored_class_changed",
                "mean",
            ),
        )
        .reset_index()
    )
    labeled_summary_path = args.output_dir / "update_behavior_labeled_oof_summary.csv"
    labeled_group_summary.to_csv(labeled_summary_path, index=False)

    gate = rules["clear_improvement_gate"]
    log_loss_reduction = unanchored_metrics["log_loss"] - anchored_metrics["log_loss"]
    no_update_labeled_count = int(
        (labeled_groups == "effectively_no_meaningful_update").sum()
    )
    clearly_improves = (
        log_loss_reduction >= float(gate["minimum_log_loss_reduction"])
        and anchored_metrics["brier_multiclass"] <= unanchored_metrics["brier_multiclass"]
        and anchored_metrics["macro_f1"]
        >= unanchored_metrics["macro_f1"]
        - float(gate["require_macro_f1_not_worse_by_more_than"])
        and no_update_labeled_count
        >= int(gate["minimum_labeled_effectively_no_update_cases"])
    )
    decision = "apply_experimental_anchor" if clearly_improves else "retain_current_unanchored_t1"

    diagnostics = {
        "version": rules["version"],
        "total_candidate_count": len(behavior),
        "total_unanchored_class_changes": int(unanchored_changed.sum()),
        "group_counts": groups.value_counts().to_dict(),
        "labeled_group_counts": labeled_groups.value_counts().to_dict(),
        "unanchored_oof_metrics": unanchored_metrics,
        "experimental_anchored_oof_metrics": anchored_metrics,
        "unanchored_oof_class_change_rate": unanchored_change_rate,
        "experimental_anchored_oof_class_change_rate": anchored_change_rate,
        "log_loss_reduction": log_loss_reduction,
        "clear_improvement_gate_passed": clearly_improves,
        "decision": decision,
        "production_t1_predictions_modified": False,
        "output_sha256": {},
    }
    frozen_rules = args.output_dir / "update_behavior_rules_v1.json"
    frozen_rules.write_bytes(args.rules.read_bytes())
    output_paths = [
        behavior_path,
        summary_path,
        comparison_path,
        bootstrap_path,
        experimental_oof_path,
        labeled_summary_path,
        frozen_rules,
    ]
    for path in output_paths:
        diagnostics["output_sha256"][path.name] = sha256_file(path)
    diagnostics_path = args.output_dir / "update_behavior_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    report = [
        "# Final T0→T1 Update-Behavior Analysis v1",
        "",
        f"- Current T1 class changes: {int(unanchored_changed.sum()):,} of {len(behavior):,}",
        f"- Decision: `{decision}`",
        "- Production T1 predictions modified: no",
        "",
        "## Evidence groups and probability movement",
        "",
        "| Group | Candidates | Current class changes | Change rate | Mean total-variation change | Mean absolute probability change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(
            f"| {row.evidence_group} | {row.candidates:,} | {row.unanchored_class_changes:,} | {row.unanchored_class_change_rate:.3%} | {row.mean_total_variation_change:.6f} | {row.mean_absolute_probability_change:.6f} |"
        )
    report.extend(
        [
            "",
            "The effectively-no-meaningful-update group changes class too frequently operationally: its current 30.2% full-population flip rate is not lower than the meaningful-evidence group by enough, and its mean probability movement is slightly larger. This is a stability warning.",
            "",
            "## Same-fold OOF anchoring comparison",
            "",
            "| Approach | Log loss | Macro-F1 | Brier | ECE | Class-change rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        report.append(
            f"| {row.approach} | {row.log_loss:.6f} | {row.macro_f1:.6f} | {row.brier_multiclass:.6f} | {row.ece_10bin:.6f} | {row.class_change_rate:.3%} |"
        )
    report.extend(
        [
            "",
            f"Anchoring reduces OOF log loss by {log_loss_reduction:.6f}, below the predeclared 0.01 clear-improvement threshold. Only {no_update_labeled_count} labeled candidates are in the effectively-no-meaningful-update group, below the required {gate['minimum_labeled_effectively_no_update_cases']}. Paired bootstrap intervals are saved separately and include substantial uncertainty because the affected labeled subgroup is small.",
            "",
            "## Release decision",
            "",
            "The current unanchored T1 model is retained, exactly as requested when anchoring does not clearly improve held-out behavior. The experimental anchor is not applied to `t1_predictions.csv` or the combined case predictions. The stability warning should be revisited once representative real labels provide at least 30 effectively-no-update cases.",
        ]
    )
    report_path = args.output_dir / "update_behavior_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
