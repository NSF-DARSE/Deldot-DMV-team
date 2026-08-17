"""Filesystem locations for the challenge package.

All I/O should go through these paths so notebooks, tests, and later modeling
code resolve files the same way regardless of the process working directory.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_T0 = PROJECT_ROOT / "Data_T0"
DATA_T1 = PROJECT_ROOT / "Data_T1"
LABELS_PATH = PROJECT_ROOT / "Development_Labels" / "Development_Labels.csv"
SUBMISSION_TEMPLATE = PROJECT_ROOT / "Submission_Template.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LINKED_DIR = OUTPUT_DIR / "linked"
FEATURES_DIR = OUTPUT_DIR / "features"
BASELINE_DIR = OUTPUT_DIR / "baseline"

CANDIDATE_RECORDS = DATA_T0 / "candidate_records.csv"
ADDRESS_HISTORY = DATA_T0 / "address_history.csv"
LICENSE_ID_EVENTS = DATA_T0 / "license_id_events.csv"
VEHICLE_TITLE_EVENTS = DATA_T0 / "vehicle_title_events.csv"
WORK_LOCATION_SIGNALS = DATA_T0 / "work_location_signals.csv"
EXTERNAL_CONTEXT_SIGNALS = DATA_T0 / "external_context_signals.csv"
EVIDENCE_UPDATE_STREAM = DATA_T1 / "evidence_update_stream.csv"
