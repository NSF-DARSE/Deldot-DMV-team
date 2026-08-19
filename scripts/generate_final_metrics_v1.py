from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    ndcg_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling_v1.t0 import metric_bundle


PROBABILITY_COLUMNS = [
    "p_review_warranted",
    "p_review_not_warranted",
    "p_insufficient_evidence",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def priority_metrics(
    actual_labels: pd.Series,
    priorities: np.ndarray,
    top_fractions: list[float],
) -> dict[str, float]:
    relevant = actual_labels.eq("review_warranted").to_numpy(dtype=int)
    metrics = {
        "average_precision_review_warranted": float(
            average_precision_score(relevant, priorities)
        ),
        "ndcg_review_warranted": float(ndcg_score(relevant[None, :], priorities[None, :])),
    }
    order = np.argsort(-priorities)
    total_relevant = max(1, int(relevant.sum()))
    for fraction in top_fractions:
        count = max(1, int(math.ceil(len(relevant) * fraction)))
        selected_relevant = int(relevant[order[:count]].sum())
        suffix = str(int(round(fraction * 100)))
        metrics[f"precision_at_top_{suffix}pct"] = selected_relevant / count
        metrics[f"recall_at_top_{suffix}pct"] = selected_relevant / total_relevant
    return metrics


def bootstrap_phase_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    priorities: np.ndarray,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [
        "log_loss",
        "macro_f1",
        "brier_multiclass",
        "ece_10bin",
        "average_precision_review_warranted",
    ]
    values = {name: [] for name in names}
    for _ in range(resamples):
        indices = rng.integers(0, len(y), size=len(y))
        metrics = metric_bundle(y[indices], probabilities[indices])
        for name in names[:-1]:
            values[name].append(metrics[name])
        relevant = (y[indices] == 0).astype(int)
        values["average_precision_review_warranted"].append(
            float(average_precision_score(relevant, priorities[indices]))
        )
    rows = []
    for name, samples in values.items():
        array = np.asarray(samples)
        rows.append(
            {
                "metric": name,
                "bootstrap_mean": float(array.mean()),
                "ci_2_5": float(np.quantile(array, 0.025)),
                "ci_97_5": float(np.quantile(array, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(
    phase: str, y: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (confidence >= lower) & (confidence <= upper)
            if index == bins - 1
            else (confidence >= lower) & (confidence < upper)
        )
        rows.append(
            {
                "phase": phase,
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
                "calibration_gap": (
                    float(correct[mask].mean() - confidence[mask].mean())
                    if mask.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def distribution_audit(
    phase: str,
    frame: pd.DataFrame,
    predictor_names: list[str],
    labeled_ids: set[str],
    thresholds: dict,
) -> pd.DataFrame:
    is_labeled = frame["candidate_record_id"].isin(labeled_ids)
    rows = []
    for feature in predictor_names:
        labeled = pd.to_numeric(frame.loc[is_labeled, feature], errors="coerce")
        unlabeled = pd.to_numeric(frame.loc[~is_labeled, feature], errors="coerce")
        labeled_nonmissing = labeled.dropna()
        unlabeled_nonmissing = unlabeled.dropna()
        ks = (
            float(ks_2samp(labeled_nonmissing, unlabeled_nonmissing).statistic)
            if len(labeled_nonmissing) and len(unlabeled_nonmissing)
            else np.nan
        )
        full_std = float(pd.concat([labeled_nonmissing, unlabeled_nonmissing]).std(ddof=0))
        mean_difference = float(labeled_nonmissing.mean() - unlabeled_nonmissing.mean())
        standardized = mean_difference / full_std if full_std > 0 else 0.0
        missing_difference = float(labeled.isna().mean() - unlabeled.isna().mean())
        warning = (
            (pd.notna(ks) and ks > float(thresholds["ks_statistic"]))
            or abs(standardized)
            > float(thresholds["absolute_standardized_mean_difference"])
            or abs(missing_difference)
            > float(thresholds["absolute_missing_rate_difference"])
        )
        rows.append(
            {
                "phase": phase,
                "feature": feature,
                "labeled_count": int(labeled_nonmissing.size),
                "unlabeled_count": int(unlabeled_nonmissing.size),
                "labeled_mean": float(labeled_nonmissing.mean()),
                "unlabeled_mean": float(unlabeled_nonmissing.mean()),
                "standardized_mean_difference": standardized,
                "ks_statistic": ks,
                "labeled_missing_rate": float(labeled.isna().mean()),
                "unlabeled_missing_rate": float(unlabeled.isna().mean()),
                "missing_rate_difference": missing_difference,
                "warning": bool(warning),
            }
        )
    return pd.DataFrame(rows)


def feature_contract(
    phase: str, frame: pd.DataFrame, predictor_names: list[str], labeled_ids: set[str]
) -> list[dict]:
    is_labeled = frame["candidate_record_id"].isin(labeled_ids)
    rows = []
    for feature in predictor_names:
        full = pd.to_numeric(frame[feature], errors="coerce")
        labeled = pd.to_numeric(frame.loc[is_labeled, feature], errors="coerce")
        rows.append(
            {
                "phase": phase,
                "feature": feature,
                "dtype": str(frame[feature].dtype),
                "full_missing_rate": float(full.isna().mean()),
                "full_min": float(full.min()) if full.notna().any() else None,
                "full_q01": float(full.quantile(0.01)) if full.notna().any() else None,
                "full_median": float(full.median()) if full.notna().any() else None,
                "full_q99": float(full.quantile(0.99)) if full.notna().any() else None,
                "full_max": float(full.max()) if full.notna().any() else None,
                "labeled_min": float(labeled.min()) if labeled.notna().any() else None,
                "labeled_max": float(labeled.max()) if labeled.notna().any() else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final leakage-safe model metrics.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "final_metrics_v1.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "final_metrics_v1"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    class_order = config["class_order"]
    class_to_index = {label: index for index, label in enumerate(class_order)}

    labels = pd.read_csv(
        ROOT
        / "Identify_Out_of_State_Tag_Holders"
        / "Development_Labels"
        / "Development_Labels.csv"
    ).set_index("candidate_record_id")
    labeled_ids = set(labels.index)
    phase_inputs = {
        "T0": (
            pd.read_csv(ROOT / "outputs" / "t0_model_v1" / "t0_oof_predictions.csv").set_index(
                "candidate_record_id"
            ),
            "label_t0",
        ),
        "T1": (
            pd.read_csv(ROOT / "outputs" / "t1_model_v1" / "t1_oof_predictions.csv").set_index(
                "candidate_record_id"
            ),
            "label_t1",
        ),
    }
    overall_rows = []
    per_class_rows = []
    confusion_rows = []
    calibration_pieces = []
    bootstrap_pieces = []
    phase_cache = {}
    for phase, (predictions, label_column) in phase_inputs.items():
        predictions = predictions.loc[labels.index]
        actual = labels[label_column]
        y = actual.map(class_to_index).to_numpy()
        probabilities = predictions[PROBABILITY_COLUMNS].to_numpy()
        predicted_indices = probabilities.argmax(axis=1)
        metrics = metric_bundle(y, probabilities)
        ranking = priority_metrics(
            actual,
            predictions["review_priority"].to_numpy(),
            [float(value) for value in config["priority_top_fractions"]],
        )
        overall_rows.append({"phase": phase, **metrics, **ranking})
        precision, recall, f1, support = precision_recall_fscore_support(
            y, predicted_indices, labels=list(range(len(class_order))), zero_division=0
        )
        for index, class_name in enumerate(class_order):
            per_class_rows.append(
                {
                    "phase": phase,
                    "class": class_name,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
            )
        matrix = confusion_matrix(y, predicted_indices, labels=list(range(len(class_order))))
        for actual_index, actual_name in enumerate(class_order):
            for predicted_index, predicted_name in enumerate(class_order):
                confusion_rows.append(
                    {
                        "phase": phase,
                        "actual_class": actual_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[actual_index, predicted_index]),
                    }
                )
        calibration_pieces.append(calibration_table(phase, y, probabilities))
        bootstrap = bootstrap_phase_metrics(
            y,
            probabilities,
            predictions["review_priority"].to_numpy(),
            int(config["bootstrap"]["resamples"]),
            int(config["bootstrap"]["random_seed"]) + (0 if phase == "T0" else 1),
        )
        bootstrap.insert(0, "phase", phase)
        bootstrap_pieces.append(bootstrap)
        phase_cache[phase] = {
            "predictions": predictions,
            "actual": actual,
            "y": y,
            "probabilities": probabilities,
            "predicted_indices": predicted_indices,
        }

    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(per_class_rows)
    confusion = pd.DataFrame(confusion_rows)
    calibration = pd.concat(calibration_pieces, ignore_index=True)
    bootstrap = pd.concat(bootstrap_pieces, ignore_index=True)

    t0_cache, t1_cache = phase_cache["T0"], phase_cache["T1"]
    actual_changed = t0_cache["actual"].ne(t1_cache["actual"]).to_numpy()
    predicted_changed = (
        t0_cache["predicted_indices"] != t1_cache["predicted_indices"]
    )
    transition_metrics = {
        "actual_class_change_count": int(actual_changed.sum()),
        "actual_class_change_rate": float(actual_changed.mean()),
        "predicted_class_change_count": int(predicted_changed.sum()),
        "predicted_class_change_rate": float(predicted_changed.mean()),
        "change_detection_precision": float(
            precision_score(actual_changed, predicted_changed, zero_division=0)
        ),
        "change_detection_recall": float(
            recall_score(actual_changed, predicted_changed, zero_division=0)
        ),
        "change_detection_f1": float(
            f1_score(actual_changed, predicted_changed, zero_division=0)
        ),
        "mean_total_variation_probability_change": float(
            0.5
            * np.abs(t1_cache["probabilities"] - t0_cache["probabilities"])
            .sum(axis=1)
            .mean()
        ),
    }
    transition_rows = []
    for transition_type, before, after in (
        ("actual", t0_cache["actual"], t1_cache["actual"]),
        (
            "predicted",
            pd.Series(np.asarray(class_order)[t0_cache["predicted_indices"]], index=labels.index),
            pd.Series(np.asarray(class_order)[t1_cache["predicted_indices"]], index=labels.index),
        ),
    ):
        table = pd.crosstab(before, after).reindex(index=class_order, columns=class_order, fill_value=0)
        for from_class in class_order:
            for to_class in class_order:
                transition_rows.append(
                    {
                        "transition_type": transition_type,
                        "from_class": from_class,
                        "to_class": to_class,
                        "count": int(table.loc[from_class, to_class]),
                    }
                )
    transitions = pd.DataFrame(transition_rows)

    t0_features = pd.read_csv(
        ROOT / "outputs" / "model_features_v1" / "t0_compact_features.csv"
    )
    t1_matrix = pd.read_csv(
        ROOT / "outputs" / "t1_model_v1" / "t1_inference_update_matrix.csv"
    )
    t0_predictors = [column for column in t0_features.columns if column != "candidate_record_id"]
    t1_diagnostics = json.loads(
        (ROOT / "outputs" / "t1_model_v1" / "t1_model_diagnostics.json").read_text()
    )
    t1_predictors = t1_diagnostics["predictor_names"]
    drift = pd.concat(
        [
            distribution_audit(
                "T0",
                t0_features,
                t0_predictors,
                labeled_ids,
                config["drift_warning_thresholds"],
            ),
            distribution_audit(
                "T1",
                t1_matrix,
                t1_predictors,
                labeled_ids,
                config["drift_warning_thresholds"],
            ),
        ],
        ignore_index=True,
    )
    contract = {
        "version": config["version"],
        "scope": config["release_scope"],
        "candidate_id_column": "candidate_record_id",
        "class_order": class_order,
        "probability_columns": PROBABILITY_COLUMNS,
        "t0_features": feature_contract("T0", t0_features, t0_predictors, labeled_ids),
        "t1_features": feature_contract("T1", t1_matrix, t1_predictors, labeled_ids),
        "real_data_checks_required": [
            "all required columns present with compatible numeric types",
            "candidate identifiers unique at each phase",
            "missingness and distribution drift reviewed before scoring",
            "linkage precision sampled and audited",
            "probability calibration refit using representative real labeled cases",
            "performance and update stability approved before operational use",
        ],
    }

    paths = {
        "overall": args.output_dir / "final_oof_metrics.csv",
        "per_class": args.output_dir / "final_per_class_metrics.csv",
        "confusion": args.output_dir / "final_confusion_matrices.csv",
        "calibration": args.output_dir / "final_calibration_bins.csv",
        "bootstrap": args.output_dir / "final_metric_bootstrap_intervals.csv",
        "transitions": args.output_dir / "final_transition_matrices.csv",
        "drift": args.output_dir / "labeled_unlabeled_feature_drift.csv",
        "contract": args.output_dir / "real_data_feature_contract.json",
    }
    overall.to_csv(paths["overall"], index=False)
    per_class.to_csv(paths["per_class"], index=False)
    confusion.to_csv(paths["confusion"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    bootstrap.to_csv(paths["bootstrap"], index=False)
    transitions.to_csv(paths["transitions"], index=False)
    drift.to_csv(paths["drift"], index=False)
    paths["contract"].write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    frozen_config = args.output_dir / "final_metrics_v1.json"
    frozen_config.write_bytes(args.config.read_bytes())

    combined = pd.read_csv(
        ROOT / "outputs" / "t1_model_v1" / "case_predictions_t0_t1.csv"
    )
    expected_submission_columns = [
        "candidate_record_id",
        "phase",
        "predicted_class",
        "p_review_warranted",
        "p_review_not_warranted",
        "p_insufficient_evidence",
        "review_priority",
    ]
    template = pd.read_csv(
        ROOT
        / "Identify_Out_of_State_Tag_Holders"
        / "Submission_Template.csv"
    )[["candidate_record_id", "phase"]]
    final_submission = template.merge(
        combined, on=["candidate_record_id", "phase"], how="left", validate="one_to_one"
    )[expected_submission_columns]
    submission_path = args.output_dir / "case_predictions.csv"
    final_submission.to_csv(submission_path, index=False)
    (ROOT / "case_predictions.csv").write_bytes(submission_path.read_bytes())
    dashboard_predictions = ROOT / "backend" / "data" / "case_predictions.csv"
    if dashboard_predictions.parent.exists():
        dashboard_predictions.write_bytes(submission_path.read_bytes())
    probability_sum_error = float(
        np.abs(combined[PROBABILITY_COLUMNS].sum(axis=1) - 1.0).max()
    )
    update_diagnostics = json.loads(
        (args.output_dir / "update_behavior_diagnostics.json").read_text()
    )
    diagnostics = {
        "version": config["version"],
        "oof_metrics": overall.set_index("phase").to_dict(orient="index"),
        "transition_metrics": transition_metrics,
        "drift_warning_count": int(drift["warning"].sum()),
        "drift_feature_count": len(drift),
        "combined_prediction_rows": len(combined),
        "unique_candidate_phase_rows": int(
            combined[["candidate_record_id", "phase"]].drop_duplicates().shape[0]
        ),
        "probability_sum_max_error": probability_sum_error,
        "update_behavior_decision": update_diagnostics["decision"],
        "anchoring_applied": False,
        "release_status": "synthetic_prototype_ready_real_data_validation_required",
        "output_sha256": {},
    }
    output_paths = list(paths.values()) + [frozen_config, submission_path]
    for path in output_paths:
        diagnostics["output_sha256"][path.name] = sha256_file(path)
    diagnostics_path = args.output_dir / "final_metrics_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    report = [
        "# Final Metrics and Release-Readiness Report v1",
        "",
        f"- Status: `{diagnostics['release_status']}`",
        "- T0/T1 production predictions retained without experimental anchoring.",
        f"- Feature drift warnings: {diagnostics['drift_warning_count']} of {diagnostics['drift_feature_count']} checked predictors.",
        "",
        "## Leakage-safe OOF metrics",
        "",
        "| Phase | Log loss | Macro-F1 | Accuracy | Multiclass Brier | ECE | AP warranted | Precision@10% | Recall@20% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        report.append(
            f"| {row.phase} | {row.log_loss:.6f} | {row.macro_f1:.6f} | {row.accuracy:.6f} | {row.brier_multiclass:.6f} | {row.ece_10bin:.6f} | {row.average_precision_review_warranted:.6f} | {row.precision_at_top_10pct:.6f} | {row.recall_at_top_20pct:.6f} |"
        )
    report.extend(
        [
            "",
            "Bootstrap confidence intervals are provided separately. They should be emphasized over point differences because only 300 labeled candidates are available.",
            "",
            "## T0→T1 transition behavior on labeled OOF candidates",
            "",
            f"- Actual label changes: {transition_metrics['actual_class_change_count']} ({transition_metrics['actual_class_change_rate']:.2%})",
            f"- Predicted class changes: {transition_metrics['predicted_class_change_count']} ({transition_metrics['predicted_class_change_rate']:.2%})",
            f"- Change-detection precision: {transition_metrics['change_detection_precision']:.4f}",
            f"- Change-detection recall: {transition_metrics['change_detection_recall']:.4f}",
            f"- Change-detection F1: {transition_metrics['change_detection_f1']:.4f}",
            "",
            "The model changes class more often than labels do. The experimental anchor improved point metrics but failed the clear-improvement and minimum-support gates, so the current T1 model remains unchanged and update stability is a release warning.",
            "",
            "## Real-data readiness",
            "",
            "The pipeline is suitable for shadow validation on real DMV data, not direct production enforcement. The feature contract records required columns, types, missingness, and reference ranges. Real labeled cases are still required for retraining, calibration, linkage precision review, update-stability approval, and operational threshold selection.",
            "",
            "No demographic attributes are supplied, so demographic fairness cannot be evaluated from this package.",
            "",
            "The canonical `case_predictions.csv` follows the exact submission-template row and column order.",
        ]
    )
    report_path = args.output_dir / "final_metrics_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
