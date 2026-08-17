"""Load challenge CSVs into pandas DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oos_review import paths


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_candidates() -> pd.DataFrame:
    return read_csv(paths.CANDIDATE_RECORDS)


def load_labels() -> pd.DataFrame:
    return read_csv(paths.LABELS_PATH)


def load_t0_sources() -> dict[str, pd.DataFrame]:
    return {
        "address_history": read_csv(paths.ADDRESS_HISTORY),
        "license_id_events": read_csv(paths.LICENSE_ID_EVENTS),
        "vehicle_title_events": read_csv(paths.VEHICLE_TITLE_EVENTS),
        "work_location_signals": read_csv(paths.WORK_LOCATION_SIGNALS),
        "external_context_signals": read_csv(paths.EXTERNAL_CONTEXT_SIGNALS),
    }


def load_t1_stream() -> pd.DataFrame:
    return read_csv(paths.EVIDENCE_UPDATE_STREAM)
