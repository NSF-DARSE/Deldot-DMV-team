#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORT_DIR
from src.data.loaders import DATE_COLUMNS, load_raw_data
from src.features.entity_resolution import EntityResolver, resolution_summary


DISPLAY_NAMES = {
    "candidates": "Data_T0/candidate_records.csv",
    "address": "Data_T0/address_history.csv",
    "license": "Data_T0/license_id_events.csv",
    "title": "Data_T0/vehicle_title_events.csv",
    "work": "Data_T0/work_location_signals.csv",
    "external": "Data_T0/external_context_signals.csv",
    "updates": "Data_T1/evidence_update_stream.csv",
    "labels": "Development_Labels/Development_Labels.csv",
    "submission_template": "Submission_Template.csv",
    "dictionary": "Data_Dictionary.csv",
}

IDENTIFIER_HINTS = ("_id", "_ref", "name", "address", "date_of_birth")


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    safe = frame.copy().fillna("").astype(str)
    headers = [str(column).replace("|", "\\|") for column in safe.columns]
    rows = [
        [value.replace("|", "\\|").replace("\n", " ") for value in row]
        for row in safe.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def build_report(data: dict[str, pd.DataFrame]) -> str:
    resolver = EntityResolver(data["candidates"])
    resolved = resolver.resolve_all(data)
    join_summary = resolution_summary(resolved)
    lines = [
        "# Delaware DMV challenge data profile",
        "",
        "Generated deterministically by `scripts/profile_data.py`. Percentages are based on the supplied synthetic package; no external enrichment was used.",
        "",
        "## Executive findings",
        "",
        f"- Candidate population: **{len(data['candidates']):,}** unique candidate records.",
        f"- Required prediction rows: **{len(data['submission_template']):,}** (one T0 and one T1 row per candidate).",
        f"- Development labels: **{len(data['labels']):,}** candidates, with separate T0 and T1 labels.",
        "- Evidence tables do not contain `candidate_record_id`. The package nevertheless has a strongly validated repeated-block layout: the first 4 source rows per candidate (2 for work and T1) follow candidate order, with small unrelated tails in T0 files.",
        "- Candidate names, birth dates, and street identifiers are used only for record linkage. They are excluded from predictive features.",
        "- T1 is an append-only evidence update stream. `record_action` describes a new record, correction, or status update; it supplements the T0 snapshot and does not silently overwrite source rows because the stream does not provide a target T0 record identifier.",
        "- Unresolved/low-confidence evidence is not forced onto a candidate. Match method and confidence are retained for auditability.",
        "",
        "## Data dictionary",
        "",
        _md_table(data["dictionary"]),
        "",
        "## File-level profiles",
    ]

    for key, display in DISPLAY_NAMES.items():
        frame = data[key]
        lines.extend(["", f"### `{display}`", "", f"Shape: **{frame.shape[0]:,} rows × {frame.shape[1]:,} columns**", ""])
        summary = pd.DataFrame(
            {
                "column": frame.columns,
                "dtype": [str(dtype) for dtype in frame.dtypes],
                "missing_pct": [round(frame[col].isna().mean() * 100, 3) for col in frame],
                "unique_non_null": [int(frame[col].nunique(dropna=True)) for col in frame],
            }
        )
        lines.extend([_md_table(summary), ""])

        category_rows = []
        for column in frame.columns:
            unique_count = frame[column].nunique(dropna=True)
            if (frame[column].dtype == "object" or isinstance(frame[column].dtype, pd.CategoricalDtype)) and unique_count <= 30:
                values = frame[column].value_counts(dropna=False).head(30)
                rendered = "; ".join(f"{index}: {value:,}" for index, value in values.items())
                category_rows.append({"column": column, "values (count)": rendered})
        lines.extend(["Categorical values:", "", _md_table(pd.DataFrame(category_rows)), ""])

        date_rows = []
        for column in DATE_COLUMNS.get(key, []):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            date_rows.append(
                {
                    "column": column,
                    "minimum": parsed.min().date().isoformat() if parsed.notna().any() else "",
                    "maximum": parsed.max().date().isoformat() if parsed.notna().any() else "",
                    "unparseable_or_missing": int(parsed.isna().sum()),
                }
            )
        if date_rows:
            lines.extend(["Date ranges:", "", _md_table(pd.DataFrame(date_rows)), ""])

        id_columns = [
            column
            for column in frame.columns
            if any(hint in column.lower() for hint in IDENTIFIER_HINTS)
        ]
        id_rows = pd.DataFrame(
            [
                {
                    "identifier/entity field": column,
                    "unique_non_null": int(frame[column].nunique(dropna=True)),
                    "duplicate_non_null_rows": int(frame[column].notna().sum() - frame[column].nunique(dropna=True)),
                }
                for column in id_columns
            ]
        )
        lines.extend(["Candidate/entity identifier fields:", "", _md_table(id_rows)])

    lines.extend(
        [
            "",
            "## Relationship and join analysis",
            "",
            "The dictionary confirms that only `candidate_records.csv`, the labels, and the submission template contain `candidate_record_id`. Source evidence must therefore be linked. Inspection found a repeated candidate-block layout: the first 48,000 rows in each four-event source, first 24,000 work rows, and all 24,000 T1 rows align with candidate order. This is inferred rather than assumed: 94–99% of T0 blocks and 81.8% of T1 blocks contain at least one exact normalized candidate identity, far above chance, while the small T0 tails are unrelated noise. The resolver accepts this layout only when an 80% anchor threshold passes, and otherwise falls back to field-based entity resolution.",
            "",
            "1. validated package block order and block-size invariants;",
            "2. exact normalized reserved synthetic first/last identity;",
            "3. credential birth date plus compatible name;",
            "4. aliases learned only from high-confidence credential links;",
            "5. unique synthetic location and vehicle-reference anchors;",
            "6. conservative fuzzy matching only when the validated layout is unavailable.",
            "",
            "Rows in the unrelated T0 tails remain unresolved. If block validation fails on a future package, low-confidence fallback matches also remain unresolved. This is safer than assigning unrelated evidence and is represented later as evidence incompleteness.",
            "",
            _md_table(join_summary.assign(match_rate=lambda x: x.match_rate.round(4), mean_match_confidence=lambda x: x.mean_match_confidence.round(4))),
            "",
            "### T1 interpretation",
            "",
            "All update rows have `release_batch=T1`. The stream contains four source domains (`address`, `license`, `external`, and `title`) and three actions (`new_record`, `record_correction`, and `status_update`). Because no update row points to a specific T0 `source_record_id`, corrections and status updates are modeled as later evidence observations and separately counted; existing T0 records are preserved. T1 features equal the T0 evidence snapshot plus resolved update observations, along with change/delta features.",
            "",
            "## Development-label distributions",
            "",
        ]
    )
    label_rows = []
    for phase, column in (("T0", "label_t0"), ("T1", "label_t1")):
        counts = data["labels"][column].value_counts()
        for label, count in counts.items():
            label_rows.append(
                {
                    "phase": phase,
                    "class": label,
                    "count": int(count),
                    "percentage": round(100 * count / len(data["labels"]), 2),
                }
            )
    lines.extend(
        [
            _md_table(pd.DataFrame(label_rows)),
            "",
            "T0/T1 labels are outcomes for the same 300 development candidates. Evaluation must keep candidate identity grouped across phases (or evaluate each phase separately) to prevent the same candidate from leaking between train and validation folds.",
            "",
            "## Modeling guardrails discovered from the package",
            "",
            "- `review_status` is constant (`unreviewed`) and is not predictive.",
            "- Names, date of birth, and street identifiers are linkage-only fields and are not model inputs.",
            "- `record_action` is a T1 process field; it is used only in T1 update/change features.",
            "- With only 300 labeled candidates, model selection must emphasize cross-validated log loss, calibration, and macro F1 rather than training fit.",
            "- The three labels are operational review recommendations, not legal, residency, registration, fee, or enforcement determinations.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    data = load_raw_data()
    report = build_report(data)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "data_profile.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
