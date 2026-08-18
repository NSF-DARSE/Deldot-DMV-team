from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR


T0_FILES = {
    "candidates": "candidate_records.csv",
    "address": "address_history.csv",
    "license": "license_id_events.csv",
    "title": "vehicle_title_events.csv",
    "work": "work_location_signals.csv",
    "external": "external_context_signals.csv",
}

DATE_COLUMNS = {
    "candidates": ["candidate_observed_date"],
    "address": ["effective_start_date", "effective_end_date"],
    "license": ["event_date"],
    "title": ["event_date"],
    "work": ["observed_date"],
    "external": ["effective_date"],
    "updates": ["effective_date", "observed_date"],
}


def _read_csv(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in date_columns or []:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_raw_data(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the complete challenge package without mutating source values."""
    data: dict[str, pd.DataFrame] = {}
    for domain, filename in T0_FILES.items():
        data[domain] = _read_csv(raw_dir / "Data_T0" / filename, DATE_COLUMNS[domain])
    data["updates"] = _read_csv(
        raw_dir / "Data_T1" / "evidence_update_stream.csv", DATE_COLUMNS["updates"]
    )
    data["labels"] = _read_csv(
        raw_dir / "Development_Labels" / "Development_Labels.csv"
    )
    data["submission_template"] = _read_csv(raw_dir / "Submission_Template.csv")
    data["dictionary"] = _read_csv(raw_dir / "Data_Dictionary.csv")
    return data

