from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED, VALID_CLASSES


ID_COLUMNS = {"candidate_record_id", "phase", "phase_t1"}


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    usable = frame.drop(columns=[column for column in ID_COLUMNS if column in frame])
    categorical = usable.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in usable.columns if column not in categorical]
    return numeric, categorical


def _preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def model_candidates(numeric: list[str], categorical: list[str]) -> tuple[dict[str, Any], list[str]]:
    logistic = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=True)),
            (
                "classifier",
                LogisticRegression(
                    C=0.35,
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=260,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    hist_gradient_boosting = Pipeline(
        [
            ("preprocess", _preprocessor(numeric, categorical, scale=False)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_iter=180,
                    max_leaf_nodes=12,
                    min_samples_leaf=12,
                    l2_regularization=1.0,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    models: dict[str, Any] = {
        "logistic_regression": logistic,
        "logistic_regression_sigmoid": CalibratedClassifierCV(
            estimator=logistic, method="sigmoid", cv=3
        ),
        "random_forest": random_forest,
        "random_forest_sigmoid": CalibratedClassifierCV(
            estimator=random_forest, method="sigmoid", cv=3
        ),
        "hist_gradient_boosting": hist_gradient_boosting,
        "hist_gradient_boosting_sigmoid": CalibratedClassifierCV(
            estimator=hist_gradient_boosting, method="sigmoid", cv=3
        ),
    }
    skipped: list[str] = []

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        xgb = Pipeline(
            [
                ("preprocess", _preprocessor(numeric, categorical, scale=False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=220,
                        max_depth=3,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        models["xgboost"] = xgb
        models["xgboost_sigmoid"] = CalibratedClassifierCV(xgb, method="sigmoid", cv=3)
    else:
        skipped.append("XGBoost (package not installed)")

    if importlib.util.find_spec("catboost"):
        from catboost import CatBoostClassifier

        cat = Pipeline(
            [
                ("preprocess", _preprocessor(numeric, categorical, scale=False)),
                (
                    "classifier",
                    CatBoostClassifier(
                        iterations=220,
                        depth=4,
                        learning_rate=0.035,
                        loss_function="MultiClass",
                        random_seed=RANDOM_SEED,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
        models["catboost"] = cat
        models["catboost_sigmoid"] = CalibratedClassifierCV(cat, method="sigmoid", cv=3)
    else:
        skipped.append("CatBoost (package not installed)")
    return models, skipped


def align_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(VALID_CLASSES)), dtype=float)
    for source_index, label in enumerate(classes):
        aligned[:, VALID_CLASSES.index(str(label))] = probabilities[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sums, out=np.full_like(aligned, 1 / len(VALID_CLASSES)), where=row_sums > 0)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    predicted_index = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    predicted = np.array(VALID_CLASSES)[predicted_index]
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def evaluate_probabilities(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    predicted = np.array(VALID_CLASSES)[np.argmax(proba, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=list(VALID_CLASSES), zero_division=0
    )
    one_hot = np.column_stack([(y_true == label).astype(float) for label in VALID_CLASSES])
    return {
        # log_loss sorts `labels` internally, so passing VALID_CLASSES (schema
        # order, not alphabetical) made it read column 0 as insufficient_evidence
        # and column 2 as review_warranted. Score on indices instead.
        "log_loss": float(
            log_loss(
                np.array([VALID_CLASSES.index(str(label)) for label in y_true]),
                proba,
                labels=list(range(len(VALID_CLASSES))),
            )
        ),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "multiclass_brier": float(np.mean(np.sum((proba - one_hot) ** 2, axis=1))),
        "ece_10_bin": expected_calibration_error(y_true, proba),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(VALID_CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=list(VALID_CLASSES)
        ).tolist(),
    }


def cross_validated_predictions(
    estimator: Any, features: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros((len(features), len(VALID_CLASSES)), dtype=float)
    for train_index, valid_index in splitter.split(features, target):
        fitted = clone(estimator)
        fitted.fit(features.iloc[train_index], target.iloc[train_index])
        fold_probability = fitted.predict_proba(features.iloc[valid_index])
        oof[valid_index] = align_probabilities(fold_probability, fitted.classes_)
    return oof


def train_and_evaluate(
    t0: pd.DataFrame,
    t1: pd.DataFrame,
    labels: pd.DataFrame,
    model_dir: Path,
    report_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    numeric, categorical = feature_columns(t0)
    all_models, skipped = model_candidates(numeric, categorical)
    phase_frames = {"T0": t0, "T1": t1}
    phase_label_columns = {"T0": "label_t0", "T1": "label_t1"}
    evaluation: dict[str, Any] = {"skipped": skipped, "models": {}}
    oof_rows: list[pd.DataFrame] = []

    for name, estimator in all_models.items():
        evaluation["models"][name] = {}
        for phase, full_frame in phase_frames.items():
            labeled = labels[["candidate_record_id", phase_label_columns[phase]]].merge(
                full_frame, on="candidate_record_id", how="left", validate="one_to_one"
            )
            target = labeled.pop(phase_label_columns[phase]).astype(str)
            candidate_ids = labeled.pop("candidate_record_id")
            features = labeled.drop(columns=["phase", "phase_t1"], errors="ignore")
            probabilities = cross_validated_predictions(estimator, features, target)
            metrics = evaluate_probabilities(target.to_numpy(), probabilities)
            evaluation["models"][name][phase] = metrics
            oof = pd.DataFrame(probabilities, columns=[f"p_{label}" for label in VALID_CLASSES])
            oof.insert(0, "candidate_record_id", candidate_ids.to_numpy())
            oof.insert(1, "phase", phase)
            oof["actual_class"] = target.to_numpy()
            oof["model_name"] = name
            oof_rows.append(oof)

    aggregate_rows = []
    for name, phases in evaluation["models"].items():
        aggregate_rows.append(
            {
                "model_name": name,
                "mean_log_loss": float(np.mean([phases[p]["log_loss"] for p in ("T0", "T1")])),
                "mean_brier": float(np.mean([phases[p]["multiclass_brier"] for p in ("T0", "T1")])),
                "mean_macro_f1": float(np.mean([phases[p]["macro_f1"] for p in ("T0", "T1")])),
                "mean_ece": float(np.mean([phases[p]["ece_10_bin"] for p in ("T0", "T1")])),
            }
        )
    leaderboard = pd.DataFrame(aggregate_rows)
    leaderboard["selection_score"] = (
        leaderboard["mean_log_loss"]
        + 0.20 * leaderboard["mean_brier"]
        + 0.10 * leaderboard["mean_ece"]
        - 0.05 * leaderboard["mean_macro_f1"]
    )
    leaderboard = leaderboard.sort_values(
        ["selection_score", "mean_log_loss", "mean_macro_f1"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    selected_name = str(leaderboard.iloc[0]["model_name"])
    evaluation["selection"] = {
        "model_name": selected_name,
        "criterion": "lowest composite emphasizing cross-validated log loss, Brier score, and calibration error with a small macro-F1 reward",
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    version = f"dmv-review-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{selected_name}"
    fitted_models: dict[str, Any] = {}
    for phase, full_frame in phase_frames.items():
        labeled = labels[["candidate_record_id", phase_label_columns[phase]]].merge(
            full_frame, on="candidate_record_id", how="left", validate="one_to_one"
        )
        target = labeled.pop(phase_label_columns[phase]).astype(str)
        features = labeled.drop(columns=["candidate_record_id", "phase", "phase_t1"], errors="ignore")
        final_model = clone(all_models[selected_name])
        final_model.fit(features, target)
        fitted_models[phase] = final_model

    bundle = {
        "model_version": version,
        "selected_model": selected_name,
        "class_order": list(VALID_CLASSES),
        "feature_columns": [column for column in t0.columns if column not in ID_COLUMNS],
        "numeric_features": numeric,
        "categorical_features": categorical,
        "models": fitted_models,
        "evaluation": deepcopy(evaluation),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    all_oof = pd.concat(oof_rows, ignore_index=True)
    all_oof.to_csv(model_dir / "oof_predictions.csv", index=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "model_evaluation.md").write_text(
        render_model_report(evaluation, leaderboard), encoding="utf-8"
    )
    return bundle, leaderboard


def render_model_report(evaluation: dict[str, Any], leaderboard: pd.DataFrame) -> str:
    def table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(row) + " |" for row in rows),
            ]
        )

    compact = leaderboard.copy()
    for column in compact.columns[1:]:
        compact[column] = compact[column].map(lambda value: f"{value:.4f}")
    selected = evaluation["selection"]["model_name"]
    lines = [
        "# Model evaluation",
        "",
        "All metrics are five-fold stratified out-of-fold estimates, evaluated separately at T0 and T1 so the same candidate never appears twice in a fold. Preprocessing is fitted inside each fold. Model selection emphasizes probability quality and generalization.",
        "",
        "## Aggregate leaderboard",
        "",
        table(compact),
        "",
        f"Selected model: **{selected}**.",
        "",
        "The selection score combines mean multiclass log loss, multiclass Brier score, 10-bin expected calibration error, and a small macro-F1 reward. Lower is better.",
        "",
    ]
    if evaluation["skipped"]:
        lines.extend(
            [
                "Unavailable optional baselines: " + ", ".join(evaluation["skipped"]) + ". The reproducible core uses scikit-learn models present in the runtime.",
                "",
            ]
        )
    for name, phases in evaluation["models"].items():
        lines.extend([f"## {name}", ""])
        phase_rows = []
        for phase in ("T0", "T1"):
            metrics = phases[phase]
            phase_rows.append(
                {
                    "phase": phase,
                    "log_loss": f"{metrics['log_loss']:.4f}",
                    "accuracy": f"{metrics['accuracy']:.4f}",
                    "macro_f1": f"{metrics['macro_f1']:.4f}",
                    "brier": f"{metrics['multiclass_brier']:.4f}",
                    "ece": f"{metrics['ece_10_bin']:.4f}",
                }
            )
        lines.extend([table(pd.DataFrame(phase_rows)), ""])
        for phase in ("T0", "T1"):
            metrics = phases[phase]
            per_class = pd.DataFrame(metrics["per_class"]).T.reset_index().rename(columns={"index": "class"})
            for col in ("precision", "recall", "f1"):
                per_class[col] = per_class[col].map(lambda value: f"{value:.4f}")
            lines.extend(
                [
                    f"### {phase} per-class metrics",
                    "",
                    table(per_class),
                    "",
                    "Confusion matrix (rows=true, columns=predicted; class order: " + ", ".join(VALID_CLASSES) + "):",
                    "",
                    "```text",
                    *[str(row) for row in metrics["confusion_matrix"]],
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## Calibration conclusion",
            "",
            "Sigmoid (Platt-style) calibration was compared with each available baseline. Isotonic calibration was not selected because 300 labels per phase are too few for stable three-class isotonic fits. Every production probability vector is reordered to the official class schema and renormalized before output.",
        ]
    )
    return "\n".join(lines) + "\n"

