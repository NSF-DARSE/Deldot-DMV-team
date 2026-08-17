"""3-class model on the stage-2 feature table.

Why a model on top of the rule
------------------------------
The recency-vote rule is the auditable decision. A shallow histogram-gradient
booster is trained on the same features (including ``de_oos_score``) so it can
use interactions the linear score misses — for example DE-newer title together
with an out-of-state observation.

Validation
----------
Only 300 people are labeled. Rows are T0 and T1 for those people (600 rows).
Cross-validation is **grouped by** ``candidate_record_id`` so a person's T0
row never trains a model that is then tested on that same person's T1 row.

Outer 5-fold GroupKFold estimates accuracy / macro-F1 / log-loss.
Inner 3-fold GroupKFold selects HGB hyperparameters. The reported number is
the outer-fold mean, not the fit-on-all-300 score.

The fitted model used for the 12,000-case file is trained on all labeled
rows after that estimate is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oos_review.baseline import (
    CLASSES,
    PRIORITY_INSUFFICIENT_WEIGHT,
    PRIORITY_WARRANTED_WEIGHT,
    SUBMISSION_COLUMNS,
)

RANDOM_STATE = 17
OUTER_FOLDS = 5
INNER_FOLDS = 3

NUMERIC_FEATURES = [
    "de_oos_score",
    "address_recency_vote",
    "license_recency_vote",
    "title_recency_vote",
    "work_recency_vote",
    "external_recency_vote",
    "n_address",
    "n_license",
    "n_title",
    "n_work",
    "n_external",
    "n_t1",
    "n_current_de_ties",
    "n_current_oos_ties",
    "n_distinct_current_states",
    "n_de_newer_sources",
    "n_oos_newer_sources",
    "n_recency_votes",
    "n_sources_present",
    "mean_match_score",
]
BOOLEAN_FEATURES = [
    "observed_is_de",
    "has_open_address",
    "open_address_is_de",
    "latest_license_is_de",
    "has_active_de_license",
    "latest_title_is_de",
    "latest_work_is_de",
    "latest_external_is_de",
    "has_state_conflict",
    "oos_observed_open_de",
]
CATEGORICAL_FEATURES = [
    "phase",
    "observed_state",
    "open_address_state",
    "latest_address_state",
    "latest_license_state",
    "latest_title_state",
    "latest_work_state",
    "latest_external_state",
    "latest_license_status",
]

HGB_PARAM_GRID = {
    "clf__max_depth": [2, 3],
    "clf__learning_rate": [0.05, 0.08],
    "clf__min_samples_leaf": [15, 25],
    "clf__l2_regularization": [0.5, 1.0],
}

DEFAULT_HGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 100,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}


def stack_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Turn T0/T1 label columns into one row per candidate-phase."""
    t0 = (
        labels.rename(columns={"label_t0": "y"})[["candidate_record_id", "y"]]
        .assign(phase="T0")
    )
    t1 = (
        labels.rename(columns={"label_t1": "y"})[["candidate_record_id", "y"]]
        .assign(phase="T1")
    )
    return pd.concat([t0, t1], ignore_index=True)


def labeled_frame(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    stacked = stack_labels(labels)
    merged = features.merge(stacked, on=["candidate_record_id", "phase"], how="inner")
    if merged.empty:
        raise ValueError("No overlap between feature rows and development labels")
    return merged


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns the estimator sees. IDs and raw timestamps are excluded."""
    missing = [
        c
        for c in NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
        if c not in frame.columns
    ]
    if missing:
        raise KeyError(f"Feature table missing columns: {missing}")
    out = frame[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in BOOLEAN_FEATURES:
        out[col] = out[col].fillna(False).astype(int)
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        # sklearn imputers cannot consume pandas pd.NA.
        values = frame[col].astype(object)
        out[col] = values.where(pd.notna(values), np.nan)
    return out


def _hgb_pipeline(**clf_params: Any) -> Pipeline:
    params = {**DEFAULT_HGB_PARAMS, "random_state": RANDOM_STATE, **clf_params}
    pre = ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES + BOOLEAN_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", HistGradientBoostingClassifier(**params)),
        ]
    )


def logistic_pipeline(C: float = 0.25) -> Pipeline:
    """Linear companion used for coefficient tables, not the submission model."""
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES + BOOLEAN_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    C=C,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _align_proba(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    raw = estimator.predict_proba(X)
    order = list(estimator.named_steps["clf"].classes_)
    return np.column_stack([raw[:, order.index(c)] for c in CLASSES])


def _priority(p_w: np.ndarray, p_i: np.ndarray) -> np.ndarray:
    return np.clip(
        PRIORITY_WARRANTED_WEIGHT * p_w + PRIORITY_INSUFFICIENT_WEIGHT * p_i,
        0,
        1,
    )


@dataclass
class CVResult:
    oof_proba: np.ndarray
    oof_pred: np.ndarray
    fold_metrics: pd.DataFrame
    best_params_per_fold: list[dict] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        return float(self.fold_metrics["accuracy"].mean())

    @property
    def mean_macro_f1(self) -> float:
        return float(self.fold_metrics["macro_f1"].mean())

    @property
    def mean_log_loss(self) -> float:
        return float(self.fold_metrics["log_loss"].mean())


def nested_cv(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    search: bool = True,
) -> CVResult:
    """Grouped nested CV. ``search=False`` uses DEFAULT_HGB_PARAMS (tests)."""
    labeled = labeled_frame(features, labels)
    X = model_matrix(labeled)
    y = labeled["y"].to_numpy()
    groups = labeled["candidate_record_id"].to_numpy()
    oof = np.zeros((len(labeled), len(CLASSES)))
    fold_rows = []
    best_params: list[dict] = []
    outer = GroupKFold(n_splits=OUTER_FOLDS)

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
        if search:
            inner = GroupKFold(n_splits=INNER_FOLDS)
            gs = GridSearchCV(
                _hgb_pipeline(),
                HGB_PARAM_GRID,
                cv=inner,
                scoring="f1_macro",
                n_jobs=1,
                refit=True,
            )
            gs.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])
            estimator = gs.best_estimator_
            best_params.append(gs.best_params_)
            extra = str(gs.best_params_)
        else:
            estimator = _hgb_pipeline()
            estimator.fit(X.iloc[train_idx], y[train_idx])
            extra = "default"
        proba = _align_proba(estimator, X.iloc[test_idx])
        oof[test_idx] = proba
        pred = np.array(CLASSES)[proba.argmax(axis=1)]
        y_te = y[test_idx]
        fold_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_te, pred),
                "macro_f1": f1_score(y_te, pred, average="macro"),
                "log_loss": log_loss(y_te, proba, labels=list(CLASSES)),
                "n_people": pd.Series(groups[test_idx]).nunique(),
                "params": extra,
            }
        )

    pred_all = np.array(CLASSES)[oof.argmax(axis=1)]
    return CVResult(
        oof_proba=oof,
        oof_pred=pred_all,
        fold_metrics=pd.DataFrame(fold_rows),
        best_params_per_fold=best_params,
    )


def fit_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    params: Optional[dict] = None,
) -> Pipeline:
    """Train on every labeled candidate-phase row."""
    labeled = labeled_frame(features, labels)
    X = model_matrix(labeled)
    y = labeled["y"].to_numpy()
    clf_params = {k.replace("clf__", ""): v for k, v in (params or {}).items()}
    estimator = _hgb_pipeline(**clf_params)
    estimator.fit(X, y)
    return estimator


def majority_params(cv_result: CVResult) -> dict:
    """Use the most common inner-CV winner; fall back to defaults."""
    if not cv_result.best_params_per_fold:
        return {}
    keys = [tuple(sorted(p.items())) for p in cv_result.best_params_per_fold]
    winner = max(set(keys), key=keys.count)
    return dict(winner)


def apply_model(
    features: pd.DataFrame,
    estimator: Pipeline,
    *,
    baseline: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score every feature row. Optionally attach the rule's class and reason."""
    X = model_matrix(features)
    proba = _align_proba(estimator, X)
    # Numerical guard so submitted rows always sum to 1.
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    result = features.copy()
    result["predicted_class"] = pred
    result["p_review_warranted"] = proba[:, 0]
    result["p_review_not_warranted"] = proba[:, 1]
    result["p_insufficient_evidence"] = proba[:, 2]
    result["review_priority"] = _priority(proba[:, 0], proba[:, 2])
    result["model_margin"] = proba.max(axis=1) - np.sort(proba, axis=1)[:, -2]
    if baseline is not None:
        rule_cols = baseline[
            ["candidate_record_id", "phase", "predicted_class", "rule_reason"]
        ].rename(columns={"predicted_class": "rule_predicted_class"})
        result = result.merge(
            rule_cols,
            on=["candidate_record_id", "phase"],
            how="left",
        )
        result["model_agrees_with_rule"] = (
            result["predicted_class"] == result["rule_predicted_class"]
        )
        result["model_reason"] = [
            (
                f"model {pcls} (p={pw:.2f}/{pn:.2f}/{pi:.2f}); "
                f"rule {rcls}"
            )
            for pcls, pw, pn, pi, rcls in zip(
                result["predicted_class"],
                result["p_review_warranted"],
                result["p_review_not_warranted"],
                result["p_insufficient_evidence"],
                result["rule_predicted_class"].fillna("n/a"),
            )
        ]
    return result


def logistic_coefficients(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    C: float = 0.25,
) -> pd.DataFrame:
    """Fit the linear companion and return coefficients by class.

    This is an explanation table, not the submission model.
    """
    labeled = labeled_frame(features, labels)
    X = model_matrix(labeled)
    y = labeled["y"].to_numpy()
    pipe = logistic_pipeline(C=C)
    pipe.fit(X, y)
    clf = pipe.named_steps["clf"]
    pre = pipe.named_steps["pre"]
    names = pre.get_feature_names_out()
    coefs = pd.DataFrame(clf.coef_, columns=names, index=clf.classes_).T
    coefs.index.name = "feature"
    return coefs.reset_index()


def to_submission(preds: pd.DataFrame, template: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    missing = [c for c in SUBMISSION_COLUMNS if c not in preds.columns]
    if missing:
        raise KeyError(f"Model output missing columns: {missing}")
    out = preds[SUBMISSION_COLUMNS].copy()
    if template is not None:
        keys = template[["candidate_record_id", "phase"]]
        out = keys.merge(out, on=["candidate_record_id", "phase"], how="left")
        if out[SUBMISSION_COLUMNS[2:]].isna().any().any():
            raise ValueError("Submission is missing predictions for template rows")
    return out.reset_index(drop=True)
