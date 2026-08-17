"""Label-side metrics for the development set. Not used at submission time."""

from __future__ import annotations

import pandas as pd

from oos_review.baseline import CLASSES


def confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    return pd.crosstab(y_true, y_pred, rownames=["true"], colnames=["pred"]).reindex(
        index=list(CLASSES), columns=list(CLASSES), fill_value=0
    )


def per_class_scores(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    rows = []
    for label in CLASSES:
        actual = y_true.eq(label)
        predicted = y_pred.eq(label)
        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(actual.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize(y_true: pd.Series, y_pred: pd.Series) -> dict:
    scores = per_class_scores(y_true, y_pred)
    return {
        "accuracy": float(y_true.eq(y_pred).mean()),
        "macro_f1": float(scores["f1"].mean()),
        "per_class": scores,
        "confusion": confusion(y_true, y_pred),
    }
