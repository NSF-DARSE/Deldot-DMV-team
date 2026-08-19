from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-12


@dataclass
class OuterFoldResult:
    fold: int
    config_id: str
    inner_log_loss: float
    temperature: float
    selected_iterations: int | None
    raw_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    validation_indices: np.ndarray


def _normalized_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = _normalized_probabilities(probabilities)
    logits = np.log(probabilities) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    probabilities: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float] = (0.25, 5.0),
) -> float:
    probabilities = _normalized_probabilities(probabilities)

    def objective(temperature: float) -> float:
        return float(log_loss(y, apply_temperature(probabilities, temperature), labels=[0, 1, 2]))

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x if result.success else 1.0)


def expected_calibration_error(
    y: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    probabilities = _normalized_probabilities(probabilities)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return float(error)


def metric_bundle(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = _normalized_probabilities(probabilities)
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[y]
    squared = (probabilities - one_hot) ** 2
    return {
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "accuracy": float(accuracy_score(y, predicted)),
        "brier_multiclass": float(squared.sum(axis=1).mean()),
        "brier_macro_ovr": float(squared.mean()),
        "ece_10bin": expected_calibration_error(y, probabilities, bins=10),
    }


def logistic_pipeline(feature_names: list[str], config: dict[str, Any], seed: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    C=float(config["C"]),
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def catboost_model(
    config: dict[str, Any], seed: int, iterations: int | None = None
) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=int(iterations if iterations is not None else config["iterations"]),
        learning_rate=float(config["learning_rate"]),
        depth=int(config["depth"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_strength=float(config["random_strength"]),
        boosting_type="Ordered",
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        random_seed=seed,
        allow_writing_files=False,
        thread_count=1,
        verbose=False,
    )


def inner_oof_predictions(
    model_type: str,
    config: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    probabilities = np.zeros((len(x), 3), dtype=float)
    best_iterations: list[int] = []
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        fold_seed = seed + inner_fold + 1
        if model_type == "catboost":
            model = catboost_model(config, fold_seed)
            model.fit(
                x.iloc[train_index],
                y[train_index],
                eval_set=(x.iloc[validation_index], y[validation_index]),
                use_best_model=True,
                early_stopping_rounds=int(config["early_stopping_rounds"]),
                verbose=False,
            )
            best_iteration = model.get_best_iteration()
            best_iterations.append(max(1, int(best_iteration) + 1))
        else:
            model = logistic_pipeline(x.columns.tolist(), config, fold_seed)
            model.fit(x.iloc[train_index], y[train_index])
        probabilities[validation_index] = model.predict_proba(x.iloc[validation_index])
    return probabilities, best_iterations


def choose_config(
    model_type: str,
    grid: list[dict[str, Any]],
    x: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    temperature_bounds: tuple[float, float],
) -> tuple[dict[str, Any], np.ndarray, float, float, int | None]:
    candidates = []
    for config_index, config in enumerate(grid):
        probabilities, best_iterations = inner_oof_predictions(
            model_type, config, x, y, splits, seed + config_index * 100
        )
        temperature = fit_temperature(probabilities, y, temperature_bounds)
        calibrated = apply_temperature(probabilities, temperature)
        score = float(log_loss(y, calibrated, labels=[0, 1, 2]))
        selected_iterations = (
            max(1, int(np.median(best_iterations))) if best_iterations else None
        )
        candidates.append(
            (score, config["id"], config, probabilities, temperature, selected_iterations)
        )
    score, _, config, probabilities, temperature, selected_iterations = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return config, probabilities, score, temperature, selected_iterations


def fit_outer_fold(
    model_type: str,
    grid: list[dict[str, Any]],
    x: pd.DataFrame,
    y: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    fold: int,
    seed: int,
    temperature_bounds: tuple[float, float],
) -> OuterFoldResult:
    x_train = x.iloc[train_index].reset_index(drop=True)
    y_train = y[train_index]
    config, _, inner_score, temperature, iterations = choose_config(
        model_type,
        grid,
        x_train,
        y_train,
        inner_splits,
        seed + fold * 1000,
        temperature_bounds,
    )
    if model_type == "catboost":
        model = catboost_model(config, seed + fold, iterations=iterations)
        model.fit(x_train, y_train, verbose=False)
    else:
        model = logistic_pipeline(x.columns.tolist(), config, seed + fold)
        model.fit(x_train, y_train)
    raw = model.predict_proba(x.iloc[validation_index])
    calibrated = apply_temperature(raw, temperature)
    return OuterFoldResult(
        fold=fold,
        config_id=config["id"],
        inner_log_loss=inner_score,
        temperature=temperature,
        selected_iterations=iterations,
        raw_probabilities=raw,
        calibrated_probabilities=calibrated,
        validation_indices=validation_index,
    )


def most_common_config(results: list[OuterFoldResult]) -> str:
    counts = Counter(result.config_id for result in results)
    return sorted(counts, key=lambda config_id: (-counts[config_id], config_id))[0]
