#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_COLUMNS = [
    "candidate_record_id",
    "phase",
    "predicted_class",
    "p_review_warranted",
    "p_review_not_warranted",
    "p_insufficient_evidence",
    "review_priority",
]
PROBABILITY_COLUMNS = [
    "p_review_warranted",
    "p_review_not_warranted",
    "p_insufficient_evidence",
]
VALID_CLASSES = {
    "review_warranted",
    "review_not_warranted",
    "insufficient_evidence",
}
DEFAULT_SUBMISSION = ROOT / "case_predictions.csv"
DEFAULT_TEMPLATE = ROOT / "submission_template.csv"
FALLBACK_TEMPLATE = ROOT / "Identify_Out_of_State_Tag_Holders" / "Submission_Template.csv"


def validate_submission(path: Path, template_path: Path) -> None:
    errors: list[str] = []
    submission = pd.read_csv(path)
    template = pd.read_csv(template_path)
    if set(submission.columns) != set(EXPECTED_COLUMNS):
        errors.append(
            f"columns must be {EXPECTED_COLUMNS}; got {list(submission.columns)}"
        )
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in submission.columns]
    if missing_cols:
        raise AssertionError("Submission validation failed:\n- missing columns: " + ", ".join(missing_cols))
    if len(submission) != len(template):
        errors.append(f"expected {len(template):,} rows; got {len(submission):,}")
    if submission.isna().any().any():
        errors.append("submission contains missing values")
    if submission.duplicated(["candidate_record_id", "phase"]).any():
        errors.append("duplicate candidate_record_id/phase rows found")
    expected_pairs = set(map(tuple, template[["candidate_record_id", "phase"]].to_numpy()))
    actual_pairs = set(map(tuple, submission[["candidate_record_id", "phase"]].to_numpy()))
    missing_pairs = expected_pairs - actual_pairs
    extra_pairs = actual_pairs - expected_pairs
    if missing_pairs:
        errors.append(f"missing {len(missing_pairs)} candidate/phase rows; sample={list(missing_pairs)[:3]}")
    if extra_pairs:
        errors.append(f"found {len(extra_pairs)} unexpected candidate/phase rows; sample={list(extra_pairs)[:3]}")
    if not set(submission["phase"]).issubset({"T0", "T1"}):
        errors.append("phase contains values outside T0/T1")
    if not set(submission["predicted_class"]).issubset(set(VALID_CLASSES)):
        errors.append("predicted_class contains invalid class names")
    for column in PROBABILITY_COLUMNS + ["review_priority"]:
        numeric = pd.to_numeric(submission[column], errors="coerce")
        if numeric.isna().any():
            errors.append(f"{column} contains non-numeric values")
        if ((numeric < 0) | (numeric > 1)).any():
            errors.append(f"{column} contains values outside [0, 1]")
    sums = submission[PROBABILITY_COLUMNS].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-8):
        bad = int((~np.isclose(sums, 1.0, atol=1e-8)).sum())
        errors.append(f"{bad} rows have class probabilities that do not sum to 1")
    phase_counts = submission.groupby("candidate_record_id")["phase"].agg(lambda values: set(values))
    if not phase_counts.map(lambda values: values == {"T0", "T1"}).all():
        errors.append("one or more candidates do not have exactly T0 and T1")
    if submission["candidate_record_id"].nunique() != template["candidate_record_id"].nunique():
        errors.append("candidate count does not match template")
    if errors:
        raise AssertionError("Submission validation failed:\n- " + "\n- ".join(errors))
    print(
        f"VALID: {path} has {len(submission):,} rows, "
        f"{submission['candidate_record_id'].nunique():,} candidates, two phases each, "
        "valid classes, normalized probabilities, and bounded priorities."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE if DEFAULT_TEMPLATE.exists() else FALLBACK_TEMPLATE,
    )
    args = parser.parse_args()
    validate_submission(args.path, args.template)


if __name__ == "__main__":
    main()
