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


def probability_frame(
    candidate_ids: pd.Series,
    probabilities: np.ndarray,
    class_order: list[str],
) -> pd.DataFrame:
    predicted = np.asarray(class_order)[probabilities.argmax(axis=1)]
    return pd.DataFrame(
        {
            "candidate_record_id": candidate_ids.to_numpy(),
            "phase": "T0",
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
    parser = argparse.ArgumentParser(description="Train and evaluate leakage-safe T0 models.")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "outputs" / "model_features_v1" / "t0_compact_features.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "Identify_Out_of_State_Tag_Holders"
        / "Development_Labels"
        / "Development_Labels.csv",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "t0_model_v1.json")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "t0_model_v1"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text())
    class_order = config["class_order"]
    class_to_index = {label: index for index, label in enumerate(class_order)}
    features = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    labeled = labels[["candidate_record_id", "label_t0"]].merge(
        features, on="candidate_record_id", how="inner", validate="one_to_one"
    )
    if len(labeled) != len(labels):
        raise ValueError("Not every development label matched the compact T0 feature matrix")
    if labeled["candidate_record_id"].duplicated().any():
        raise ValueError("Development candidates are not unique")

    feature_names = [column for column in features.columns if column != "candidate_record_id"]
    x = labeled[feature_names]
    y = labeled["label_t0"].map(class_to_index).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise ValueError("Unknown T0 class label")

    seed = int(config["random_seed"])
    bounds = tuple(float(value) for value in config["calibration"]["temperature_bounds"])
    outer_cv = StratifiedKFold(
        n_splits=int(config["outer_cv_folds"]), shuffle=True, random_state=seed
    )
    outer_splits = list(outer_cv.split(x, y))
    model_grids = {
        "catboost": config["catboost_grid"],
        "logistic": config["logistic_grid"],
    }
    all_results = {}
    comparison_metrics = {}
    fold_metric_rows = []
    fold_assignment_rows = []

    for fold, (_, validation_index) in enumerate(outer_splits):
        for index in validation_index:
            fold_assignment_rows.append(
                {"candidate_record_id": labeled.iloc[index]["candidate_record_id"], "outer_fold": fold}
            )

    for model_type, grid in model_grids.items():
        raw_oof = np.zeros((len(labeled), 3), dtype=float)
        calibrated_oof = np.zeros((len(labeled), 3), dtype=float)
        results = []
        for fold, (train_index, validation_index) in enumerate(outer_splits):
            inner_cv = StratifiedKFold(
                n_splits=int(config["inner_cv_folds"]),
                shuffle=True,
                random_state=seed + fold + 1,
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
                seed,
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
        final_model = catboost_model(
            selected_config, seed, iterations=selected_iterations
        )
        final_model.fit(labeled[feature_names], y, verbose=False)
        model_path = args.output_dir / "t0_catboost_model.cbm"
        final_model.save_model(model_path)
        importance_values = final_model.get_feature_importance()
    else:
        final_model = logistic_pipeline(feature_names, selected_config, seed)
        final_model.fit(labeled[feature_names], y)
        model_path = args.output_dir / "t0_logistic_model.joblib"
        joblib.dump(final_model, model_path)
        coefficients = final_model.named_steps["classifier"].coef_
        importance_values = np.abs(coefficients[:, : len(feature_names)]).mean(axis=0)

    importance = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance_path = args.output_dir / "t0_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    final_raw = final_model.predict_proba(features[feature_names])
    final_probabilities = apply_temperature(final_raw, final_temperature)
    t0_predictions = probability_frame(features["candidate_record_id"], final_probabilities, class_order)
    t0_predictions_path = args.output_dir / "t0_predictions.csv"
    t0_predictions.to_csv(t0_predictions_path, index=False)

    oof_predictions = probability_frame(
        labeled["candidate_record_id"], selected["calibrated_oof"], class_order
    )
    fold_map = pd.DataFrame(fold_assignment_rows)
    oof_predictions = oof_predictions.merge(fold_map, on="candidate_record_id", how="left")
    oof_predictions["actual_class"] = labeled["label_t0"].to_numpy()
    oof_predictions["model"] = selected_model
    oof_path = args.output_dir / "t0_oof_predictions.csv"
    oof_predictions.to_csv(oof_path, index=False)

    training_priors = t0_predictions.copy()
    replacement = oof_predictions.set_index("candidate_record_id")
    probability_columns = [
        "p_review_warranted",
        "p_review_not_warranted",
        "p_insufficient_evidence",
    ]
    training_priors["prediction_origin"] = "full_fit"
    training_priors = training_priors.set_index("candidate_record_id")
    for candidate_id in replacement.index:
        training_priors.loc[candidate_id, probability_columns] = replacement.loc[
            candidate_id, probability_columns
        ].to_numpy()
        training_priors.loc[candidate_id, "predicted_class"] = replacement.loc[
            candidate_id, "predicted_class"
        ]
        training_priors.loc[candidate_id, "review_priority"] = replacement.loc[
            candidate_id, "review_priority"
        ]
        training_priors.loc[candidate_id, "prediction_origin"] = "outer_oof"
    training_priors = training_priors.reset_index()
    priors_path = args.output_dir / "t0_training_priors.csv"
    training_priors.to_csv(priors_path, index=False)

    comparison_rows = []
    for model_type, metrics in comparison_metrics.items():
        comparison_rows.append({"model": model_type, **metrics})
    comparison_path = args.output_dir / "t0_model_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    fold_metrics_path = args.output_dir / "t0_fold_metrics.csv"
    pd.DataFrame(fold_metric_rows).to_csv(fold_metrics_path, index=False)
    folds_path = args.output_dir / "t0_outer_fold_assignments.csv"
    fold_map.sort_values("candidate_record_id").to_csv(folds_path, index=False)
    frozen_config_path = args.output_dir / "t0_model_v1.json"
    frozen_config_path.write_bytes(args.config.read_bytes())

    diagnostics = {
        "version": config["version"],
        "candidate_count": len(features),
        "labeled_candidate_count": len(labeled),
        "feature_count": len(feature_names),
        "selected_model": selected_model,
        "selection_reason": selection_reason,
        "selected_config": selected_config,
        "selected_iterations": selected_iterations,
        "final_temperature": final_temperature,
        "comparison_metrics": comparison_metrics,
        "priority_formula": config["review_priority"]["formula"],
        "probability_sum_max_error": float(
            np.abs(t0_predictions[probability_columns].sum(axis=1) - 1.0).max()
        ),
        "oof_training_prior_count": int(
            (training_priors["prediction_origin"] == "outer_oof").sum()
        ),
        "output_sha256": {},
    }
    output_paths = [
        model_path,
        t0_predictions_path,
        oof_path,
        priors_path,
        comparison_path,
        fold_metrics_path,
        folds_path,
        frozen_config_path,
        importance_path,
    ]
    for path in output_paths:
        diagnostics["output_sha256"][path.name] = sha256_file(path)
    diagnostics_path = args.output_dir / "t0_model_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    report = [
        "# T0 Model Breakpoint v1",
        "",
        f"- Selected model: `{selected_model}`",
        f"- Selection reason: {selection_reason}",
        f"- Labeled candidates: {len(labeled)}",
        f"- Compact predictors: {len(feature_names)}",
        f"- Validation: {config['outer_cv_folds']}-fold outer CV with {config['inner_cv_folds']}-fold tuning/calibration inside each outer training fold",
        f"- Final deployment temperature: {final_temperature:.6f}, fit on nested OOF raw probabilities",
        f"- Review priority: `{config['review_priority']['formula']}`",
        "",
        "## Nested OOF metrics",
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
            "Accuracy is reported but did not control model selection. Log loss is primary because calibrated probabilities and safe uncertainty handling matter operationally.",
            "",
            "## Leakage safeguards",
            "",
            "- Candidate IDs are unique, so no candidate crosses an outer train/validation boundary.",
            "- Hyperparameter choice and early stopping occur only inside each outer training partition.",
            "- Each outer-fold temperature is fit on inner OOF probabilities, never on in-fold predictions.",
            "- The 300 labeled T0 priors supplied to the later T1 training stage are outer-OOF probabilities, not predictions from a model trained on those candidates.",
            "",
            "## Generalization boundary",
            "",
            "Synthetic validation cannot prove real-DMV performance. The portable safeguards are frozen linkage, stable feature definitions, small model capacity, nested validation, calibrated uncertainty, and audit-only treatment of sparse legal-window signals. Real deployment still requires schema checks plus retraining and calibration on representative real labeled cases.",
            "",
            "## Breakpoint",
            "",
            "T0 is complete. T1 training has not started.",
        ]
    )
    report_path = args.output_dir / "t0_model_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
