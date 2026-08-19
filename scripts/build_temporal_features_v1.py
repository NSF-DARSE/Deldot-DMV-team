from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_prep_v1 import TemporalFeatureBuilder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "feature_prep_v1").glob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def description_for(column: str) -> str:
    explicit = {
        "candidate_record_id": "Stable candidate identifier; metadata only and never a model predictor.",
        "phase": "Prediction phase metadata.",
        "as_of_date": "Date through which effective/available evidence is included; metadata only.",
        "de_residency_proxy_date": "Start of latest uninterrupted Delaware address-evidence run; audit metadata, not a legal residency date.",
        "prior_oos_address_state_proxy": "Most recent non-DE address state immediately before the latest uninterrupted DE address-evidence run; audit metadata only.",
        "de_residency_proxy_present": "Latest dated address-state evidence is Delaware.",
        "days_since_de_residency_proxy": "Nonnegative whole days from the DE address-evidence proxy date through the phase as-of date.",
        "de_residency_proxy_source_count": "Distinct address-evidence source types contributing to the latest uninterrupted DE run.",
        "prior_oos_address_state_proxy_present": "A non-DE address state exists before the latest uninterrupted DE address-evidence run.",
        "current_within_60_day_grace_proxy": "As-of date is no more than 60 days after the Delaware address-evidence proxy date.",
        "current_past_60_day_grace_proxy": "As-of date is more than 60 days after the Delaware address-evidence proxy date.",
        "days_past_60_day_grace_proxy": "Whole days beyond day 60 after the DE address-evidence proxy date; zero while within the proxy window or when no proxy exists.",
        "pre_de_proxy_oos_vehicle_signal_count": "OOS title-domain records dated before the DE address-evidence proxy date.",
        "within_60d_oos_vehicle_signal_count": "OOS title-domain records dated from day 0 through day 60 after the DE address-evidence proxy date, inclusive.",
        "post_60d_oos_vehicle_signal_count": "OOS title-domain signals occurring more than 60 days after the Delaware address-evidence proxy date.",
        "post_60d_oos_vehicle_signal_present": "At least one OOS title-domain signal occurs after the proxy 60-day window.",
        "days_past_grace_at_latest_oos_vehicle_signal": "Days beyond day 60 for the latest post-window OOS title-domain record; zero when absent.",
        "within_60d_active_oos_credential_signal_count": "OOS credential records explicitly marked active and dated from day 0 through day 60 after the DE address-evidence proxy date, inclusive.",
        "post_60d_active_oos_credential_signal_count": "Explicitly active OOS credential events occurring more than 60 days after the DE address-evidence proxy date; credential type is unknown.",
        "post_60d_active_oos_credential_signal_present": "At least one explicitly active OOS credential record occurs more than 60 days after the DE address-evidence proxy date.",
        "days_past_grace_at_latest_active_oos_credential_signal": "Days beyond day 60 for the latest explicitly active post-window OOS credential record; zero when absent.",
        "post_60d_oos_credential_update_status_unknown_count": "T1 OOS license-domain updates after the proxy window whose active status cannot be determined.",
        "post_60d_combined_title_active_credential_conflict_count": "When both components exist, total post-window supporting records across OOS title and explicitly active OOS credential sources; otherwise zero.",
        "post_60d_combined_title_active_credential_conflict_present": "Both a post-window OOS title-domain signal and an explicitly active post-window OOS credential signal are present.",
        "post_move_oos_title_record_count": "OOS title-domain records on or after the DE proxy date whose event_type is title_record.",
        "post_move_oos_ownership_change_count": "OOS title-domain records on or after the DE proxy date whose event_type is ownership_change.",
        "post_move_oos_record_update_count": "OOS title-domain records on or after the DE proxy date whose event_type is record_update.",
        "post_move_oos_t1_title_new_record_count": "OOS T1 title-domain updates on or after the DE proxy date whose record_action is new_record.",
        "post_move_oos_t1_title_status_update_count": "OOS T1 title-domain updates on or after the DE proxy date whose record_action is status_update.",
        "post_move_oos_t1_title_correction_count": "OOS T1 title-domain updates on or after the DE proxy date whose record_action is record_correction.",
        "post_60d_prior_state_persistence_cross_source_present": "After the proxy window, both title and active credential signals match the prior OOS address-state proxy.",
        "post_move_oos_vehicle_matches_prior_state_count": "OOS title-domain records on or after the DE proxy date whose state equals the prior OOS address-state proxy.",
        "post_60d_oos_vehicle_matches_prior_state_count": "OOS title-domain records after day 60 whose state equals the prior OOS address-state proxy.",
        "post_move_active_credential_matches_prior_state_count": "Explicitly active OOS credential records on or after the DE proxy date whose state equals the prior OOS address-state proxy.",
        "post_60d_active_credential_matches_prior_state_count": "Explicitly active OOS credential records after day 60 whose state equals the prior OOS address-state proxy.",
        "post_60d_prior_state_persistence_any_present": "At least one post-window OOS title or explicitly active credential record matches the prior OOS address-state proxy.",
        "new_meaningful_oos_title_signal_after_move_present": "After the DE proxy date, an OOS title_record, ownership_change, or T1 title new_record is present.",
        "new_meaningful_oos_title_signal_post_60d_present": "A meaningful OOS title event occurs after the proxy 60-day window.",
        "t1_update_observation_lag_mean_days": "Mean days between effective_date and observed_date for linked T1 updates.",
        "t1_update_observation_lag_max_days": "Maximum days between effective_date and observed_date among linked T1 updates; zero when absent.",
        "t1_late_arriving_update_count_over_30d": "Linked T1 updates observed more than 30 days after their effective date.",
        "recent_oos_independent_source_count_90d": "Distinct canonical sources with OOS records dated within 90 days of the phase as-of date.",
        "recent_de_independent_source_count_90d": "Distinct canonical sources with DE records dated within 90 days of the phase as-of date.",
        "recent_oos_cross_source_confirmation_present_90d": "At least two independent canonical sources contain OOS evidence in the latest 90 days.",
        "active_oos_credential_count": "Included credential records explicitly marked active whose credential_state is non-DE.",
        "active_de_credential_count": "Included credential records explicitly marked active whose credential_state is DE.",
        "expired_or_superseded_credential_count": "Included credential records explicitly marked expired or superseded.",
        "future_effective_record_count": "Phase-available linked records excluded because their effective/event date is after the phase as-of date.",
        "historical_oos_only": "OOS history exists, latest DE evidence is newer, and no OOS evidence occurs in the latest 365 days.",
        "de_after_oos": "Latest DE evidence is later than latest OOS evidence.",
        "oos_after_de": "Latest OOS evidence is later than latest DE evidence.",
    }
    if column in explicit:
        return explicit[column]
    if column.startswith("delta_"):
        return f"T1 value minus T0 value for {column.removeprefix('delta_')}."
    if column.startswith("new_t1_"):
        return "Count or diversity calculated only from linked T1 update-stream records."
    return column.replace("_", " ").capitalize() + "."


def source_urls_for(column: str, official_urls: list[str]) -> str:
    base = column.removeprefix("delta_")
    vehicle_urls = official_urls[:2]
    credential_urls = official_urls[2:]
    has_vehicle = any(term in base for term in ("vehicle", "title"))
    has_credential = "credential" in base
    shared_window = any(term in base for term in ("60_day_grace", "de_residency_proxy"))
    prior_state_combined = "prior_state_persistence" in base
    if (has_vehicle and has_credential) or prior_state_combined or shared_window:
        return "; ".join(official_urls)
    if has_vehicle and "60" in base:
        return "; ".join(vehicle_urls)
    if has_credential and "60" in base:
        return "; ".join(credential_urls)
    return ""


def write_dictionary(path: Path, t0: pd.DataFrame, delta: pd.DataFrame, official_urls: list[str]) -> None:
    rows = []
    for artifact, frame in (("features_t0/features_t1", t0), ("features_t1_delta", delta)):
        for column in frame.columns:
            role = "metadata" if column in {"candidate_record_id", "phase", "as_of_date", "de_residency_proxy_date", "prior_oos_address_state_proxy"} else "model_candidate_feature"
            source_url = source_urls_for(column, official_urls)
            rows.append(
                {
                    "artifact": artifact,
                    "column": column,
                    "role": role,
                    "dtype": str(frame[column].dtype),
                    "description": description_for(column),
                    "source_url": source_url,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build migration-aware T0/T1 features.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "Identify_Out_of_State_Tag_Holders")
    parser.add_argument("--linked-events", type=Path, default=ROOT / "outputs/linkage_v1/linked_events.csv")
    parser.add_argument("--rules", type=Path, default=ROOT / "configs/temporal_feature_rules_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/feature_prep_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builder = TemporalFeatureBuilder(args.data_root, args.linked_events, args.rules)
    t0, t1, delta = builder.build()
    t0_path = args.output_dir / "features_t0.csv"
    t1_path = args.output_dir / "features_t1.csv"
    delta_path = args.output_dir / "features_t1_delta.csv"
    dictionary_path = args.output_dir / "temporal_feature_dictionary.csv"
    t0.to_csv(t0_path, index=False)
    t1.to_csv(t1_path, index=False)
    delta.to_csv(delta_path, index=False)
    write_dictionary(dictionary_path, t0, delta, builder.rules["official_sources"])
    frozen_rules = args.output_dir / "frozen_temporal_feature_rules_v1.json"
    frozen_rules.write_bytes(args.rules.read_bytes())

    diagnostics = {
        "feature_rule_version": builder.rules["version"],
        "rules_sha256": builder.rules_sha256,
        "implementation_sha256": implementation_sha256(),
        "candidate_count": len(t0),
        "t0_feature_column_count": len(t0.columns),
        "t1_feature_column_count": len(t1.columns),
        "t1_delta_column_count": len(delta.columns),
        "t1_as_of_date": builder.t1_as_of.date().isoformat(),
        "t0_future_effective_records_excluded": int(t0["future_effective_record_count"].sum()),
        "t0_candidates_with_de_residency_proxy": int(t0["de_residency_proxy_present"].sum()),
        "t0_candidates_within_60_day_proxy_window": int(t0["current_within_60_day_grace_proxy"].sum()),
        "t0_candidates_past_60_day_proxy_window": int(t0["current_past_60_day_grace_proxy"].sum()),
        "t0_candidates_with_post_60d_oos_vehicle_signal": int(t0["post_60d_oos_vehicle_signal_present"].sum()),
        "t1_candidates_with_de_residency_proxy": int(t1["de_residency_proxy_present"].sum()),
        "t1_candidates_within_60_day_proxy_window": int(t1["current_within_60_day_grace_proxy"].sum()),
        "t1_candidates_past_60_day_proxy_window": int(t1["current_past_60_day_grace_proxy"].sum()),
        "t1_candidates_with_post_60d_oos_vehicle_signal": int(t1["post_60d_oos_vehicle_signal_present"].sum()),
        "t0_candidates_with_post_60d_active_oos_credential_signal": int(t0["post_60d_active_oos_credential_signal_present"].sum()),
        "t1_candidates_with_post_60d_active_oos_credential_signal": int(t1["post_60d_active_oos_credential_signal_present"].sum()),
        "t0_candidates_with_post_60d_combined_conflict": int(t0["post_60d_combined_title_active_credential_conflict_present"].sum()),
        "t1_candidates_with_post_60d_combined_conflict": int(t1["post_60d_combined_title_active_credential_conflict_present"].sum()),
        "t0_candidates_with_post_60d_prior_state_persistence": int(t0["post_60d_prior_state_persistence_any_present"].sum()),
        "t1_candidates_with_post_60d_prior_state_persistence": int(t1["post_60d_prior_state_persistence_any_present"].sum()),
        "t0_candidates_historical_oos_only": int(t0["historical_oos_only"].sum()),
        "t1_candidates_historical_oos_only": int(t1["historical_oos_only"].sum()),
        "t1_candidates_with_linked_updates": int(delta["t1_has_linked_update"].sum()),
        "known_legal_limitations": [
            "No authoritative legal-residency start date is supplied.",
            "No authoritative out-of-state registration-status field is supplied.",
            "Military and apportioned-vehicle exemption indicators are unavailable.",
            "Grace-window fields are decision-support proxies, never violation determinations.",
            "The license/ID source does not identify credential type; credential features are not authoritative driver-license features.",
        ],
    }
    outputs = [t0_path, t1_path, delta_path, dictionary_path, frozen_rules]
    diagnostics["output_sha256"] = {path.name: sha256_file(path) for path in outputs}
    diagnostics_path = args.output_dir / "feature_prep_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

    report = [
        "# Temporal Feature Preparation v1",
        "",
        f"- Feature rule version: `{diagnostics['feature_rule_version']}`",
        f"- Frozen rule SHA-256: `{diagnostics['rules_sha256']}`",
        f"- Frozen implementation SHA-256: `{diagnostics['implementation_sha256']}`",
        f"- Candidate rows per phase: {len(t0):,}",
        f"- T0/T1 cumulative columns: {len(t0.columns)}",
        f"- T1 delta columns: {len(delta.columns)}",
        "",
        "## Temporal safeguards",
        "",
        "- T0 uses candidate_observed_date as the candidate-specific as-of date.",
        f"- T1 uses the complete release batch cutoff: {builder.t1_as_of.date().isoformat()}.",
        f"- {diagnostics['t0_future_effective_records_excluded']:,} future-effective T0 records are excluded from T0 current-state features and retained as a data-quality count.",
        "- Old OOS evidence is retained as history but separated from recent windows and recency-decayed evidence.",
        "- DE-after-OOS and OOS-after-DE direction, per-source transitions, active addresses, and latest-source conflicts are explicit features.",
        "",
        "## Delaware 60-day window",
        "",
        "Delaware's 60-day new-resident vehicle registration requirement is represented only through proxy features. The clock uses the start of the latest uninterrupted Delaware address-evidence run. OOS title-domain signals are split into pre-move, within-window, and post-window counts.",
        "",
        "Official sources:",
        "",
        *[f"- {url}" for url in builder.rules["official_sources"]],
        "",
        "The data cannot establish legal residency, actual registration status, or exemptions. No feature is a violation flag.",
        "",
        "## Supported credential, persistence, and title-event additions",
        "",
        "- Active OOS credential events are separated into within-60-day and post-60-day counts. Because credential type is absent, these are not labeled as authoritative driver-license records.",
        "- T1 license-domain updates without credential_status are counted separately and never assumed active.",
        "- Combined conflict requires both a post-window OOS title signal and an explicitly active post-window OOS credential signal.",
        "- Prior-state persistence compares later title/active-credential states with the last OOS address state immediately preceding the latest DE address-evidence run.",
        "- Meaningful title evidence distinguishes title_record, ownership_change, and T1 new_record from older/pre-move or generic record_update evidence.",
        "- T1 observation lag and recent 90-day independent-source corroboration are included as data-supported context.",
        "",
        "## Breakpoint",
        "",
        "No model has been trained. These feature tables must be reviewed before CatBoost T0 training.",
    ]
    (args.output_dir / "temporal_feature_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
