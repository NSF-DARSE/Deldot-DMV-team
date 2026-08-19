from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling_v1.t0 import (
    apply_temperature,
    catboost_model,
    fit_outer_fold,
    fit_temperature,
    logistic_pipeline,
    metric_bundle,
    most_common_config,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rename_t0_priors(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "p_review_warranted": "p_review_warranted_t0",
            "p_review_not_warranted": "p_review_not_warranted_t0",
            "p_insufficient_evidence": "p_insufficient_evidence_t0",
            "review_priority": "priority_t0",
            "prediction_origin": "prediction_origin_t0",
        }
    )


def build_update_matrix(priors: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    priors = rename_t0_priors(priors)
    prior_columns = [
        "candidate_record_id",
        "p_review_warranted_t0",
        "p_review_not_warranted_t0",
        "p_insufficient_evidence_t0",
        "priority_t0",
    ]
    if "prediction_origin_t0" in priors.columns:
        prior_columns.append("prediction_origin_t0")
    return priors[prior_columns].merge(
        deltas, on="candidate_record_id", how="inner", validate="one_to_one"
    )


def probability_frame(
    candidate_ids: pd.Series, probabilities: np.ndarray, class_order: list[str]
) -> pd.DataFrame:
    predicted = np.asarray(class_order)[probabilities.argmax(axis=1)]
    return pd.DataFrame(
        {
            "candidate_record_id": candidate_ids.to_numpy(),
            "phase": "T1",
            "p_review_warranted": probabilities[:, class_order.index("review_warranted")],
            "p_review_not_warranted": probabilities[:, class_order.index("review_not_warranted")],
            "p_insufficient_evidence": probabilities[:, class_order.index("insufficient_evidence")],
            "predicted_class": predicted,
            "review_priority": probabilities[:, class_order.index("review_warranted")],
        }
    )


def choose_model(comparison: dict[str, dict[str, float]]) -> tuple[str, str]:
    catboost = comparison["catboost"]
    logistic = comparison["logistic"]
    improvement = logistic["log_loss"] - catboost["log_loss"]
    if improvement >= 0.01:
        return "catboost", f"CatBoost improved calibrated log loss by {improvement:.6f}."
    if improvement > -0.01 and catboost["macro_f1"] - logistic["macro_f1"] >= 0.02:
        return "catboost", "Log loss was effectively tied and CatBoost improved macro-F1 by at least 0.02."
    return "logistic", "The simpler logistic baseline was preferred under the predeclared near-tie rule."


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the leakage-aware T1 update model.")
    parser.add_argument(
        "--delta-features",
        type=Path,
        default=ROOT
        / "outputs"
        / "model_features_v1"
        / "t1_compact_update_features.csv",
    )
    parser.add_argument(
        "--training-priors",
        type=Path,
        default=ROOT / "outputs" / "t0_model_v1" / "t0_training_priors.csv",
    )
    parser.add_argument(
        "--inference-priors",
        type=Path,
        default=ROOT / "outputs" / "t0_model_v1" / "t0_predictions.csv",
    )
    parser.add_argument(
        "--t0-folds",
        type=Path,
        default=ROOT / "outputs" / "t0_model_v1" / "t0_outer_fold_assignments.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "Identify_Out_of_State_Tag_Holders"
        / "Development_Labels"
        / "Development_Labels.csv",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "t1_model_v1.json")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "t1_model_v1"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text())
    class_order = config["class_order"]
    class_to_index = {label: index for index, label in enumerate(class_order)}
    deltas = pd.read_csv(args.delta_features)
    training_matrix = build_update_matrix(pd.read_csv(args.training_priors), deltas)
    inference_matrix = build_update_matrix(pd.read_csv(args.inference_priors), deltas)
    if len(training_matrix) != len(inference_matrix) or len(training_matrix) != 12_000:
        raise ValueError("T1 update matrices must contain exactly 12,000 candidates")

    training_matrix_path = args.output_dir / "t1_training_update_matrix.csv"
    inference_matrix_path = args.output_dir / "t1_inference_update_matrix.csv"
    training_matrix.to_csv(training_matrix_path, index=False)
    inference_matrix.to_csv(inference_matrix_path, index=False)

    labels = pd.read_csv(args.labels)
    folds = pd.read_csv(args.t0_folds)
    labeled = (
        labels[["candidate_record_id", "label_t1"]]
        .merge(training_matrix, on="candidate_record_id", how="inner", validate="one_to_one")
        .merge(folds, on="candidate_record_id", how="inner", validate="one_to_one")
    )
    if len(labeled) != len(labels):
        raise ValueError("Not every T1 label matched both the update matrix and saved T0 fold")
    if labeled["candidate_record_id"].duplicated().any():
        raise ValueError("Development candidates are not unique")

    excluded = {
        "candidate_record_id",
        "label_t1",
        "outer_fold",
        "priority_t0",
        "prediction_origin_t0",
    }
    predictor_names = [column for column in labeled.columns if column not in excluded]
    expected_prior_names = config["t0_prior_predictors"]
    if not set(expected_prior_names).issubset(predictor_names):
        raise ValueError("T1 prior predictors are missing")
    if "priority_t0" in predictor_names:
        raise ValueError("Duplicate T0 priority must not be a T1 predictor")

    x = labeled[predictor_names]
    y = labeled["label_t1"].map(class_to_index).to_numpy()
    seed = int(config["random_seed"])
    bounds = tuple(float(value) for value in config["calibration"]["temperature_bounds"])
    outer_splits = []
    for fold in sorted(labeled["outer_fold"].unique()):
        validation_index = np.flatnonzero(labeled["outer_fold"].to_numpy() == fold)
        train_index = np.flatnonzero(labeled["outer_fold"].to_numpy() != fold)
        outer_splits.append((train_index, validation_index))

    model_grids = {
        "catboost": config["catboost_grid"],
        "logistic": config["logistic_grid"],
    }
    all_results = {}
    comparison_metrics = {}
    fold_metric_rows = []
    for model_type, grid in model_grids.items():
        raw_oof = np.zeros((len(labeled), 3), dtype=float)
        calibrated_oof = np.zeros((len(labeled), 3), dtype=float)
        results = []
        for fold, (train_index, validation_index) in enumerate(outer_splits):
            inner_cv = StratifiedKFold(
                n_splits=int(config["inner_cv_folds"]),
                shuffle=True,
                random_state=seed + fold + 101,
            )
            inner_splits = list(inner_cv.split(x.iloc[train_index], y[train_index]))
            result = fit_outer_fold(
                model_type,
                grid,
                x,
                y,
                train_index,
                validation_index,
                inner_splits,
                fold,
                seed + 10_000,
                bounds,
            )
            results.append(result)
            raw_oof[validation_index] = result.raw_probabilities
            calibrated_oof[validation_index] = result.calibrated_probabilities
            row = {
                "model": model_type,
                "outer_fold": fold,
                "selected_config": result.config_id,
                "inner_calibrated_log_loss": result.inner_log_loss,
                "temperature": result.temperature,
                "selected_iterations": result.selected_iterations,
            }
            row.update(metric_bundle(y[validation_index], result.calibrated_probabilities))
            fold_metric_rows.append(row)
        all_results[model_type] = {
            "results": results,
            "raw_oof": raw_oof,
            "calibrated_oof": calibrated_oof,
        }
        comparison_metrics[model_type] = metric_bundle(y, calibrated_oof)

    selected_model, selection_reason = choose_model(comparison_metrics)
    selected = all_results[selected_model]
    selected_results = selected["results"]
    selected_config_id = most_common_config(selected_results)
    selected_grid = {item["id"]: item for item in model_grids[selected_model]}
    selected_config = selected_grid[selected_config_id]
    selected_iterations = None
    if selected_model == "catboost":
        iteration_values = [
            result.selected_iterations
            for result in selected_results
            if result.config_id == selected_config_id and result.selected_iterations is not None
        ]
        selected_iterations = max(1, int(np.median(iteration_values)))

    final_temperature = fit_temperature(selected["raw_oof"], y, bounds)
    if selected_model == "catboost":
        final_model = catboost_model(selected_config, seed + 10_000, iterations=selected_iterations)
        final_model.fit(labeled[predictor_names], y, verbose=False)
        model_path = args.output_dir / "t1_catboost_model.cbm"
        final_model.save_model(model_path)
        importance_values = final_model.get_feature_importance()
    else:
        final_model = logistic_pipeline(predictor_names, selected_config, seed + 10_000)
        final_model.fit(labeled[predictor_names], y)
        model_path = args.output_dir / "t1_logistic_model.joblib"
        joblib.dump(final_model, model_path)
        coefficients = final_model.named_steps["classifier"].coef_
        importance_values = np.abs(coefficients[:, : len(predictor_names)]).mean(axis=0)

    importance = pd.DataFrame(
        {"feature": predictor_names, "importance": importance_values}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance_path = args.output_dir / "t1_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    inference_predictors = inference_matrix[predictor_names]
    final_raw = final_model.predict_proba(inference_predictors)
    final_probabilities = apply_temperature(final_raw, final_temperature)
    t1_predictions = probability_frame(
        inference_matrix["candidate_record_id"], final_probabilities, class_order
    )
    t1_predictions_path = args.output_dir / "t1_predictions.csv"
    t1_predictions.to_csv(t1_predictions_path, index=False)

    oof_predictions = probability_frame(
        labeled["candidate_record_id"], selected["calibrated_oof"], class_order
    )
    oof_predictions["outer_fold"] = labeled["outer_fold"].to_numpy()
    oof_predictions["actual_class"] = labeled["label_t1"].to_numpy()
    oof_predictions["model"] = selected_model
    oof_path = args.output_dir / "t1_oof_predictions.csv"
    oof_predictions.to_csv(oof_path, index=False)

    comparison_path = args.output_dir / "t1_model_comparison.csv"
    pd.DataFrame(
        [{"model": model_type, **metrics} for model_type, metrics in comparison_metrics.items()]
    ).to_csv(comparison_path, index=False)
    fold_metrics_path = args.output_dir / "t1_fold_metrics.csv"
    pd.DataFrame(fold_metric_rows).to_csv(fold_metrics_path, index=False)

    t0_predictions = pd.read_csv(args.inference_priors)[
        [
            "candidate_record_id",
            "phase",
            "p_review_warranted",
            "p_review_not_warranted",
            "p_insufficient_evidence",
            "predicted_class",
            "review_priority",
        ]
    ]
    case_predictions = pd.concat([t0_predictions, t1_predictions], ignore_index=True)
    phase_order = pd.Categorical(case_predictions["phase"], categories=["T0", "T1"], ordered=True)
    case_predictions = (
        case_predictions.assign(_phase_order=phase_order)
        .sort_values(["candidate_record_id", "_phase_order"])
        .drop(columns="_phase_order")
    )
    case_predictions_path = args.output_dir / "case_predictions_t0_t1.csv"
    case_predictions.to_csv(case_predictions_path, index=False)

    frozen_config_path = args.output_dir / "t1_model_v1.json"
    frozen_config_path.write_bytes(args.config.read_bytes())
    probability_columns = [
        "p_review_warranted",
        "p_review_not_warranted",
        "p_insufficient_evidence",
    ]
    diagnostics = {
        "version": config["version"],
        "candidate_count": len(inference_matrix),
        "labeled_candidate_count": len(labeled),
        "predictor_count": len(predictor_names),
        "predictor_names": predictor_names,
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "selected_config": selected_config,
        "selected_iterations": selected_iterations,
        "final_temperature": final_temperature,
        "comparison_metrics": comparison_metrics,
        "outer_folds_reused_from_t0": True,
        "priority_t0_excluded_as_duplicate": "priority_t0" not in predictor_names,
        "probability_sum_max_error": float(
            np.abs(t1_predictions[probability_columns].sum(axis=1) - 1.0).max()
        ),
        "output_sha256": {},
    }
    output_paths = [
        model_path,
        training_matrix_path,
        inference_matrix_path,
        importance_path,
        t1_predictions_path,
        oof_path,
        comparison_path,
        fold_metrics_path,
        case_predictions_path,
        frozen_config_path,
    ]
    for path in output_paths:
        diagnostics["output_sha256"][path.name] = sha256_file(path)
    diagnostics_path = args.output_dir / "t1_model_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    report = [
        "# T1 Model Breakpoint v1",
        "",
        f"- Selected model: `{selected_model}`",
        f"- Selection reason: {selection_reason}",
        f"- Labeled candidates: {len(labeled)}",
        f"- Predictors: {len(predictor_names)} (3 T0 probabilities + {len(predictor_names) - 3} T1 update/delta features)",
        "- Outer folds: reused from T0 so outer-validation priors are OOF for the same candidates",
        f"- Inner tuning/calibration folds: {config['inner_cv_folds']}",
        f"- Final deployment temperature: {final_temperature:.6f}",
        "- `priority_t0` is retained in the update matrix but excluded from modeling because it duplicates `p_review_warranted_t0`.",
        "",
        "## Nested OOF selection metrics",
        "",
        "| Model | Log loss | Macro-F1 | Accuracy | Multiclass Brier | Macro OVR Brier | ECE (10 bins) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_type in ("catboost", "logistic"):
        metrics = comparison_metrics[model_type]
        report.append(
            f"| {model_type} | {metrics['log_loss']:.6f} | {metrics['macro_f1']:.6f} | {metrics['accuracy']:.6f} | {metrics['brier_multiclass']:.6f} | {metrics['brier_macro_ovr']:.6f} | {metrics['ece_10bin']:.6f} |"
        )
    report.extend(["", "## Final-model feature importance", ""])
    for row in importance.head(10).itertuples(index=False):
        report.append(f"- {row.rank}. `{row.feature}`: {row.importance:.6f}")
    report.extend(
        [
            "",
            "## Generalization and stacking boundary",
            "",
            "The validation candidate's T0 prior is always generated without that candidate because T1 reuses the T0 outer folds. All labeled training priors are candidate-wise OOF. This is a practical cross-fitted stack for 300 labels; synthetic validation still cannot establish production DMV performance.",
            "",
            "## Breakpoint",
            "",
            "T1 predictions are complete. The standalone final metrics and update-behavior analysis stage has not started.",
        ]
    )
    report_path = args.output_dir / "t1_model_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
