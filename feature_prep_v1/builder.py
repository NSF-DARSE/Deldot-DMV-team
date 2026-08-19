from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_SOURCES = ("address", "license", "title", "external", "work")
SOURCE_MAP = {
    "address_history": "address",
    "license_id_events": "license",
    "vehicle_title_events": "title",
    "external_context_signals": "external",
    "work_location_signals": "work",
}


def _state_class(value: object, de_state: str = "DE") -> str:
    state = "" if value is None else str(value).strip().upper()
    if not state or state == "NAN":
        return "MISSING"
    return "DE" if state == de_state else "OOS"


def _safe_days(as_of: pd.Timestamp, value: pd.Timestamp | None) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float((as_of - value).days)


class TemporalFeatureBuilder:
    def __init__(self, data_root: Path, linked_events_path: Path, rules_path: Path):
        self.data_root = Path(data_root)
        self.linked_events_path = Path(linked_events_path)
        self.rules_path = Path(rules_path)
        self.rules_bytes = self.rules_path.read_bytes()
        self.rules = json.loads(self.rules_bytes)
        self.rules_sha256 = hashlib.sha256(self.rules_bytes).hexdigest()
        self.de_state = self.rules["de_state"]
        self.grace_days = int(self.rules["new_resident_vehicle_registration_window_days"])
        self.windows = [int(value) for value in self.rules["recency_windows_days"]]
        self.half_lives = [int(value) for value in self.rules["decay_half_lives_days"]]
        self.low_confidence = float(self.rules["low_link_confidence_threshold"])

        self.candidates = pd.read_csv(
            self.data_root / "Data_T0/candidate_records.csv", dtype=str, keep_default_na=False
        )
        self.candidates["_candidate_dt"] = pd.to_datetime(
            self.candidates["candidate_observed_date"], errors="raise"
        )
        self.events = pd.read_csv(self.linked_events_path, dtype=str, keep_default_na=False)
        self.events["_event_dt"] = pd.to_datetime(self.events["event_date"], errors="coerce")
        self.events["_observed_dt"] = pd.to_datetime(self.events["observed_date"], errors="coerce")
        self.events["_end_dt"] = pd.to_datetime(self.events["effective_end_date"], errors="coerce")
        self.events["_confidence"] = pd.to_numeric(self.events["link_confidence"], errors="coerce")
        self.events["_state_class"] = self.events["state"].map(
            lambda value: _state_class(value, self.de_state)
        )
        self.events["_canonical_source"] = self.events.apply(self._canonical_source, axis=1)
        self.by_candidate = {
            candidate_id: frame.copy()
            for candidate_id, frame in self.events.groupby("candidate_record_id", sort=False)
        }

        updates = pd.read_csv(
            self.data_root / "Data_T1/evidence_update_stream.csv", dtype=str, keep_default_na=False
        )
        self.t1_as_of = pd.to_datetime(updates["observed_date"], errors="raise").max()

    @staticmethod
    def _canonical_source(row: pd.Series) -> str:
        if row["phase"] == "T1":
            domain = str(row["source_domain"]).strip().lower()
            return domain if domain in CANONICAL_SOURCES else "external"
        return SOURCE_MAP.get(str(row["source"]), str(row["source"]))

    def _phase_events(self, candidate_id: str, phase: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, int]:
        frame = self.by_candidate.get(candidate_id)
        if frame is None:
            return self.events.iloc[0:0].copy(), 0
        eligible_phase = frame[frame["phase"] == "T0"] if phase == "T0" else frame
        future_effective = eligible_phase["_event_dt"].notna() & (eligible_phase["_event_dt"] > as_of)
        available = eligible_phase[~future_effective].copy()
        if phase == "T1":
            unavailable_update = (
                (available["phase"] == "T1")
                & available["_observed_dt"].notna()
                & (available["_observed_dt"] > as_of)
            )
            available = available[~unavailable_update].copy()
        return available, int(future_effective.sum())

    @staticmethod
    def _latest_timestamp(frame: pd.DataFrame, state_class: str) -> pd.Timestamp | None:
        values = frame.loc[frame["_state_class"] == state_class, "_event_dt"].dropna()
        return values.max() if not values.empty else None

    def _residency_proxy(
        self,
        candidate: pd.Series,
        events: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> tuple[dict, pd.Timestamp | None, str]:
        address = events[events["_canonical_source"] == "address"][
            ["_event_dt", "_state_class", "state", "phase", "source"]
        ].copy()
        address["_residence_source"] = np.where(
            address["phase"] == "T1", "t1_address_update", "address_history"
        )
        candidate_event = pd.DataFrame(
            {
                "_event_dt": [candidate["_candidate_dt"]],
                "_state_class": [_state_class(candidate["observed_state"], self.de_state)],
                "state": [candidate["observed_state"]],
                "phase": ["T0"],
                "source": ["candidate_records"],
                "_residence_source": ["candidate_observed_address"],
            }
        )
        timeline = pd.concat([address, candidate_event], ignore_index=True)
        timeline = timeline[
            timeline["_event_dt"].notna()
            & (timeline["_event_dt"] <= as_of)
            & timeline["_state_class"].isin(["DE", "OOS"])
        ].copy()
        if timeline.empty:
            return {
                "de_residency_proxy_present": 0,
                "days_since_de_residency_proxy": np.nan,
                "de_residency_proxy_source_count": 0,
                "current_within_60_day_grace_proxy": 0,
                "current_past_60_day_grace_proxy": 0,
                "days_past_60_day_grace_proxy": 0,
                "prior_oos_address_state_proxy_present": 0,
            }, None, ""

        priority = {"address_history": 0, "t1_address_update": 1, "candidate_observed_address": 2}
        timeline["_priority"] = timeline["_residence_source"].map(priority).fillna(0)
        timeline = timeline.sort_values(["_event_dt", "_priority"]).reset_index(drop=True)
        if timeline.iloc[-1]["_state_class"] != "DE":
            return {
                "de_residency_proxy_present": 0,
                "days_since_de_residency_proxy": np.nan,
                "de_residency_proxy_source_count": 0,
                "current_within_60_day_grace_proxy": 0,
                "current_past_60_day_grace_proxy": 0,
                "days_past_60_day_grace_proxy": 0,
                "prior_oos_address_state_proxy_present": 0,
            }, None, ""

        oos_indices = timeline.index[timeline["_state_class"] == "OOS"].tolist()
        last_oos_index = oos_indices[-1] if oos_indices else -1
        prior_oos_state = (
            str(timeline.iloc[last_oos_index]["state"]).strip().upper()
            if last_oos_index >= 0
            else ""
        )
        current_de_run = timeline.iloc[last_oos_index + 1 :]
        current_de_run = current_de_run[current_de_run["_state_class"] == "DE"]
        proxy_date = current_de_run["_event_dt"].min()
        days_since = max(0, int((as_of - proxy_date).days))
        source_count = int(current_de_run["_residence_source"].nunique())
        return {
            "de_residency_proxy_present": 1,
            "days_since_de_residency_proxy": days_since,
            "de_residency_proxy_source_count": source_count,
            "current_within_60_day_grace_proxy": int(days_since <= self.grace_days),
            "current_past_60_day_grace_proxy": int(days_since > self.grace_days),
            "days_past_60_day_grace_proxy": max(0, days_since - self.grace_days),
            "prior_oos_address_state_proxy_present": int(bool(prior_oos_state)),
        }, proxy_date, prior_oos_state

    def _post_move_features(
        self, events: pd.DataFrame, proxy_date: pd.Timestamp | None, prior_oos_state: str
    ) -> dict:
        output = {
            "pre_de_proxy_oos_vehicle_signal_count": 0,
            "within_60d_oos_vehicle_signal_count": 0,
            "post_60d_oos_vehicle_signal_count": 0,
            "post_60d_oos_vehicle_signal_present": 0,
            "days_past_grace_at_latest_oos_vehicle_signal": 0,
            "within_60d_active_oos_credential_signal_count": 0,
            "post_60d_active_oos_credential_signal_count": 0,
            "post_60d_active_oos_credential_signal_present": 0,
            "days_past_grace_at_latest_active_oos_credential_signal": 0,
            "post_60d_oos_credential_update_status_unknown_count": 0,
            "post_60d_combined_title_active_credential_conflict_count": 0,
            "post_60d_combined_title_active_credential_conflict_present": 0,
            "post_move_oos_title_record_count": 0,
            "post_move_oos_ownership_change_count": 0,
            "post_move_oos_record_update_count": 0,
            "post_move_oos_t1_title_new_record_count": 0,
            "post_move_oos_t1_title_status_update_count": 0,
            "post_move_oos_t1_title_correction_count": 0,
            "new_meaningful_oos_title_signal_after_move_present": 0,
            "new_meaningful_oos_title_signal_post_60d_present": 0,
            "post_move_oos_vehicle_matches_prior_state_count": 0,
            "post_60d_oos_vehicle_matches_prior_state_count": 0,
            "post_move_active_credential_matches_prior_state_count": 0,
            "post_60d_active_credential_matches_prior_state_count": 0,
            "post_60d_prior_state_persistence_any_present": 0,
            "post_60d_prior_state_persistence_cross_source_present": 0,
        }
        if proxy_date is None:
            return output
        vehicle = events[
            (events["_canonical_source"] == "title")
            & (events["_state_class"] == "OOS")
            & events["_event_dt"].notna()
        ].copy()
        vehicle["_days_from_proxy"] = (vehicle["_event_dt"] - proxy_date).dt.days
        output["pre_de_proxy_oos_vehicle_signal_count"] = int((vehicle["_days_from_proxy"] < 0).sum())
        output["within_60d_oos_vehicle_signal_count"] = int(
            vehicle["_days_from_proxy"].between(0, self.grace_days, inclusive="both").sum()
        )
        post = vehicle[vehicle["_days_from_proxy"] > self.grace_days]
        output["post_60d_oos_vehicle_signal_count"] = len(post)
        output["post_60d_oos_vehicle_signal_present"] = int(not post.empty)
        if not post.empty:
            output["days_past_grace_at_latest_oos_vehicle_signal"] = int(
                post["_days_from_proxy"].max() - self.grace_days
            )
        post_move = vehicle[vehicle["_days_from_proxy"] >= 0]
        output["post_move_oos_title_record_count"] = int((post_move["event_type"] == "title_record").sum())
        output["post_move_oos_ownership_change_count"] = int((post_move["event_type"] == "ownership_change").sum())
        output["post_move_oos_record_update_count"] = int((post_move["event_type"] == "record_update").sum())
        output["post_move_oos_t1_title_new_record_count"] = int((post_move["record_action"] == "new_record").sum())
        output["post_move_oos_t1_title_status_update_count"] = int((post_move["record_action"] == "status_update").sum())
        output["post_move_oos_t1_title_correction_count"] = int((post_move["record_action"] == "record_correction").sum())
        meaningful = post_move[
            post_move["event_type"].isin(["title_record", "ownership_change"])
            | (post_move["record_action"] == "new_record")
        ]
        output["new_meaningful_oos_title_signal_after_move_present"] = int(not meaningful.empty)
        output["new_meaningful_oos_title_signal_post_60d_present"] = int(
            (meaningful["_days_from_proxy"] > self.grace_days).any()
        ) if not meaningful.empty else 0

        credential = events[
            (events["_canonical_source"] == "license")
            & (events["_state_class"] == "OOS")
            & events["_event_dt"].notna()
        ].copy()
        credential["_days_from_proxy"] = (credential["_event_dt"] - proxy_date).dt.days
        active = credential[credential["credential_status"] == "active"]
        output["within_60d_active_oos_credential_signal_count"] = int(
            active["_days_from_proxy"].between(0, self.grace_days, inclusive="both").sum()
        )
        post_active = active[active["_days_from_proxy"] > self.grace_days]
        output["post_60d_active_oos_credential_signal_count"] = len(post_active)
        output["post_60d_active_oos_credential_signal_present"] = int(not post_active.empty)
        if not post_active.empty:
            output["days_past_grace_at_latest_active_oos_credential_signal"] = int(
                post_active["_days_from_proxy"].max() - self.grace_days
            )
        unknown_t1 = credential[
            (credential["phase"] == "T1")
            & (credential["credential_status"] == "")
            & (credential["_days_from_proxy"] > self.grace_days)
        ]
        output["post_60d_oos_credential_update_status_unknown_count"] = len(unknown_t1)
        combined_conflict = not post.empty and not post_active.empty
        output["post_60d_combined_title_active_credential_conflict_count"] = int(
            len(post) + len(post_active) if combined_conflict else 0
        )
        output["post_60d_combined_title_active_credential_conflict_present"] = int(combined_conflict)

        if prior_oos_state:
            vehicle_prior = post_move[post_move["state"].str.upper() == prior_oos_state]
            vehicle_prior_post = vehicle_prior[vehicle_prior["_days_from_proxy"] > self.grace_days]
            active_prior = active[
                (active["state"].str.upper() == prior_oos_state)
                & (active["_days_from_proxy"] >= 0)
            ]
            active_prior_post = active_prior[active_prior["_days_from_proxy"] > self.grace_days]
            output["post_move_oos_vehicle_matches_prior_state_count"] = len(vehicle_prior)
            output["post_60d_oos_vehicle_matches_prior_state_count"] = len(vehicle_prior_post)
            output["post_move_active_credential_matches_prior_state_count"] = len(active_prior)
            output["post_60d_active_credential_matches_prior_state_count"] = len(active_prior_post)
            output["post_60d_prior_state_persistence_any_present"] = int(
                not vehicle_prior_post.empty or not active_prior_post.empty
            )
            output["post_60d_prior_state_persistence_cross_source_present"] = int(
                not vehicle_prior_post.empty and not active_prior_post.empty
            )
        return output

    def _aggregate_candidate(
        self,
        candidate: pd.Series,
        phase: str,
        events: pd.DataFrame,
        as_of: pd.Timestamp,
        future_effective_count: int,
    ) -> dict:
        result: dict[str, object] = {
            "candidate_record_id": candidate["candidate_record_id"],
            "phase": phase,
            "as_of_date": as_of.date().isoformat(),
            "candidate_observed_is_de": int(candidate["observed_state"] == self.de_state),
            "candidate_observed_is_oos": int(
                bool(candidate["observed_state"]) and candidate["observed_state"] != self.de_state
            ),
            "future_effective_record_count": future_effective_count,
        }
        count = len(events)
        de_count = int((events["_state_class"] == "DE").sum())
        oos_count = int((events["_state_class"] == "OOS").sum())
        missing_count = int((events["_state_class"] == "MISSING").sum())
        result.update(
            {
                "linked_record_count": count,
                "de_count_total": de_count,
                "oos_count_total": oos_count,
                "state_missing_count": missing_count,
                "state_missing_ratio": missing_count / count if count else 0.0,
                "oos_share_known_state": oos_count / (oos_count + de_count) if (oos_count + de_count) else 0.0,
                "distinct_oos_states": int(
                    events.loc[events["_state_class"] == "OOS", "state"].replace("", np.nan).nunique()
                ),
                "source_diversity": int(events["_canonical_source"].nunique()) if count else 0,
            }
        )

        dated = events[events["_event_dt"].notna()].copy()
        dated["_age_days"] = (as_of - dated["_event_dt"]).dt.days.clip(lower=0)
        for window in self.windows:
            recent = dated[dated["_age_days"] <= window]
            result[f"oos_count_{window}d"] = int((recent["_state_class"] == "OOS").sum())
            result[f"de_count_{window}d"] = int((recent["_state_class"] == "DE").sum())
        result["historical_oos_count_over_365d"] = int(
            ((dated["_state_class"] == "OOS") & (dated["_age_days"] > 365)).sum()
        )

        latest_oos = self._latest_timestamp(events, "OOS")
        latest_de = self._latest_timestamp(events, "DE")
        result["days_since_latest_oos"] = _safe_days(as_of, latest_oos)
        result["days_since_latest_de"] = _safe_days(as_of, latest_de)
        result["de_after_oos"] = int(
            latest_de is not None and latest_oos is not None and latest_de > latest_oos
        )
        result["oos_after_de"] = int(
            latest_de is not None and latest_oos is not None and latest_oos > latest_de
        )
        result["latest_oos_recency_advantage_days"] = (
            _safe_days(as_of, latest_de) - _safe_days(as_of, latest_oos)
            if latest_de is not None and latest_oos is not None
            else np.nan
        )
        result["historical_oos_only"] = int(
            oos_count > 0
            and result["de_after_oos"] == 1
            and result.get("oos_count_365d", 0) == 0
        )
        result["oos_evidence_after_latest_de_count"] = int(
            ((dated["_state_class"] == "OOS") & (dated["_event_dt"] > latest_de)).sum()
        ) if latest_de is not None else 0
        result["de_evidence_after_latest_oos_count"] = int(
            ((dated["_state_class"] == "DE") & (dated["_event_dt"] > latest_oos)).sum()
        ) if latest_oos is not None else 0
        result["recent_oos_after_latest_de_present"] = int(
            result["oos_evidence_after_latest_de_count"] > 0
            and result.get("oos_count_365d", 0) > 0
        )
        recent_90 = dated[dated["_age_days"] <= 90]
        recent_oos_source_count = int(
            recent_90.loc[recent_90["_state_class"] == "OOS", "_canonical_source"].nunique()
        )
        recent_de_source_count = int(
            recent_90.loc[recent_90["_state_class"] == "DE", "_canonical_source"].nunique()
        )
        result["recent_oos_independent_source_count_90d"] = recent_oos_source_count
        result["recent_de_independent_source_count_90d"] = recent_de_source_count
        result["recent_oos_cross_source_confirmation_present_90d"] = int(recent_oos_source_count >= 2)

        for half_life in self.half_lives:
            weights = np.exp(-math.log(2) * dated["_age_days"] / half_life) if not dated.empty else pd.Series(dtype=float)
            oos_decay = float(weights[dated["_state_class"] == "OOS"].sum()) if not dated.empty else 0.0
            de_decay = float(weights[dated["_state_class"] == "DE"].sum()) if not dated.empty else 0.0
            result[f"oos_decay_{half_life}d"] = oos_decay
            result[f"de_decay_{half_life}d"] = de_decay
            result[f"oos_decay_share_{half_life}d"] = (
                oos_decay / (oos_decay + de_decay) if (oos_decay + de_decay) else 0.0
            )

        transition_counts = defaultdict(int)
        latest_source_states = {}
        latest_source_dates = {}
        for source, source_frame in dated.groupby("_canonical_source"):
            ordered = source_frame[source_frame["_state_class"].isin(["DE", "OOS"])].sort_values("_event_dt")
            states = ordered["_state_class"].tolist()
            for previous, current in zip(states, states[1:]):
                if previous == "OOS" and current == "DE":
                    transition_counts["oos_to_de"] += 1
                elif previous == "DE" and current == "OOS":
                    transition_counts["de_to_oos"] += 1
            if states:
                latest_source_states[source] = states[-1]
                latest_source_dates[source] = ordered.iloc[-1]["_event_dt"]
        result["oos_to_de_transition_count"] = transition_counts["oos_to_de"]
        result["de_to_oos_transition_count"] = transition_counts["de_to_oos"]
        latest_de_sources = sum(value == "DE" for value in latest_source_states.values())
        latest_oos_sources = sum(value == "OOS" for value in latest_source_states.values())
        result["latest_source_de_count"] = latest_de_sources
        result["latest_source_oos_count"] = latest_oos_sources
        result["latest_source_oos_share"] = (
            latest_oos_sources / (latest_oos_sources + latest_de_sources)
            if latest_oos_sources + latest_de_sources
            else 0.0
        )
        result["cross_source_conflict_count"] = latest_de_sources * latest_oos_sources
        result["cross_source_conflict_present"] = int(latest_de_sources > 0 and latest_oos_sources > 0)
        recent_de_sources = sum(
            latest_source_states[source] == "DE" and (as_of - latest_source_dates[source]).days <= 365
            for source in latest_source_states
        )
        recent_oos_sources = sum(
            latest_source_states[source] == "OOS" and (as_of - latest_source_dates[source]).days <= 365
            for source in latest_source_states
        )
        historical_oos_sources = sum(
            latest_source_states[source] == "OOS" and (as_of - latest_source_dates[source]).days > 365
            for source in latest_source_states
        )
        result["recent_latest_source_de_count_365d"] = recent_de_sources
        result["recent_latest_source_oos_count_365d"] = recent_oos_sources
        result["historical_latest_source_oos_count"] = historical_oos_sources
        result["recent_cross_source_conflict_count_365d"] = recent_de_sources * recent_oos_sources
        result["recent_cross_source_conflict_present_365d"] = int(
            recent_de_sources > 0 and recent_oos_sources > 0
        )
        result["recency_weighted_latest_source_oos"] = float(
            sum(
                math.exp(-math.log(2) * (as_of - latest_source_dates[source]).days / 365)
                for source, state in latest_source_states.items()
                if state == "OOS"
            )
        )

        for source in CANONICAL_SOURCES:
            source_frame = events[events["_canonical_source"] == source]
            result[f"{source}_record_count"] = len(source_frame)
            result[f"{source}_de_count"] = int((source_frame["_state_class"] == "DE").sum())
            result[f"{source}_oos_count"] = int((source_frame["_state_class"] == "OOS").sum())
            result[f"{source}_state_missing_count"] = int((source_frame["_state_class"] == "MISSING").sum())
            latest = source_frame[source_frame["_event_dt"].notna()].sort_values("_event_dt")
            latest_class = latest.iloc[-1]["_state_class"] if not latest.empty else "MISSING"
            result[f"{source}_latest_is_de"] = int(latest_class == "DE")
            result[f"{source}_latest_is_oos"] = int(latest_class == "OOS")
            result[f"days_since_{source}_latest_record"] = (
                _safe_days(as_of, latest.iloc[-1]["_event_dt"]) if not latest.empty else np.nan
            )
            result[f"days_since_{source}_latest_oos"] = _safe_days(
                as_of, self._latest_timestamp(source_frame, "OOS")
            )
            result[f"days_since_{source}_latest_de"] = _safe_days(
                as_of, self._latest_timestamp(source_frame, "DE")
            )

        if count:
            confidence = events["_confidence"].dropna()
            result["link_confidence_mean"] = float(confidence.mean()) if not confidence.empty else np.nan
            result["link_confidence_min"] = float(confidence.min()) if not confidence.empty else np.nan
            result["low_confidence_link_share"] = float((confidence < self.low_confidence).mean()) if not confidence.empty else 0.0
            result["tier_c_link_share"] = float((events["link_tier"] == "C").mean())
        else:
            result.update(
                {
                    "link_confidence_mean": np.nan,
                    "link_confidence_min": np.nan,
                    "low_confidence_link_share": 0.0,
                    "tier_c_link_share": 0.0,
                }
            )

        address = events[events["_canonical_source"] == "address"]
        active_address = address[address["_end_dt"].isna() | (address["_end_dt"] >= as_of)]
        result["active_address_de_count"] = int((active_address["_state_class"] == "DE").sum())
        result["active_address_oos_count"] = int((active_address["_state_class"] == "OOS").sum())
        result["active_address_state_conflict"] = int(
            result["active_address_de_count"] > 0 and result["active_address_oos_count"] > 0
        )

        license_events = events[events["_canonical_source"] == "license"]
        result["active_oos_credential_count"] = int(
            ((license_events["credential_status"] == "active") & (license_events["_state_class"] == "OOS")).sum()
        )
        result["active_de_credential_count"] = int(
            ((license_events["credential_status"] == "active") & (license_events["_state_class"] == "DE")).sum()
        )
        result["expired_or_superseded_credential_count"] = int(
            license_events["credential_status"].isin(["expired", "superseded"]).sum()
        )
        title_events = events[events["_canonical_source"] == "title"]
        result["distinct_vehicle_count"] = int(title_events["vehicle_ref"].replace("", np.nan).nunique())
        result["ownership_change_count"] = int((title_events["event_type"] == "ownership_change").sum())
        external = events[events["_canonical_source"] == "external"]
        result["limited_quality_external_count"] = int((external["evidence_quality"] == "limited").sum())
        result["standard_quality_external_count"] = int((external["evidence_quality"] == "standard").sum())

        update_events = events[events["phase"] == "T1"].copy()
        if update_events.empty:
            update_lag = pd.Series(dtype=float)
        else:
            update_lag = (
                pd.to_datetime(update_events["_observed_dt"], errors="coerce")
                - pd.to_datetime(update_events["_event_dt"], errors="coerce")
            ).dt.days.dropna()
        result["t1_update_observation_lag_mean_days"] = float(update_lag.mean()) if not update_lag.empty else 0.0
        result["t1_update_observation_lag_max_days"] = int(update_lag.max()) if not update_lag.empty else 0
        result["t1_late_arriving_update_count_over_30d"] = int((update_lag > 30).sum())

        residence_features, proxy_date, prior_oos_state = self._residency_proxy(candidate, events, as_of)
        result.update(residence_features)
        result.update(self._post_move_features(events, proxy_date, prior_oos_state))
        result["de_residency_proxy_date"] = proxy_date.date().isoformat() if proxy_date is not None else ""
        result["prior_oos_address_state_proxy"] = prior_oos_state
        return result

    def build_phase(self, phase: str) -> pd.DataFrame:
        rows = []
        for _, candidate in self.candidates.iterrows():
            as_of = candidate["_candidate_dt"] if phase == "T0" else self.t1_as_of
            events, future_count = self._phase_events(candidate["candidate_record_id"], phase, as_of)
            rows.append(self._aggregate_candidate(candidate, phase, events, as_of, future_count))
        return pd.DataFrame(rows)

    def build_deltas(self, t0: pd.DataFrame, t1: pd.DataFrame) -> pd.DataFrame:
        left = t0.set_index("candidate_record_id")
        right = t1.set_index("candidate_record_id")
        selected = [
            "linked_record_count", "oos_count_total", "de_count_total", "distinct_oos_states",
            "days_since_latest_oos", "days_since_latest_de", "source_diversity", "state_missing_ratio",
            "cross_source_conflict_count", "link_confidence_mean", "low_confidence_link_share",
            "oos_decay_90d", "de_decay_90d", "oos_decay_365d", "de_decay_365d",
            "latest_source_oos_count", "latest_source_de_count", "active_address_oos_count",
            "recent_latest_source_oos_count_365d", "recent_latest_source_de_count_365d",
            "historical_latest_source_oos_count", "recent_cross_source_conflict_count_365d",
            "recency_weighted_latest_source_oos", "oos_evidence_after_latest_de_count",
            "de_evidence_after_latest_oos_count", "recent_oos_after_latest_de_present",
            "recent_oos_independent_source_count_90d", "recent_de_independent_source_count_90d",
            "recent_oos_cross_source_confirmation_present_90d",
            "active_address_de_count", "oos_to_de_transition_count", "de_to_oos_transition_count",
            "post_60d_oos_vehicle_signal_count", "current_within_60_day_grace_proxy",
            "current_past_60_day_grace_proxy", "days_past_60_day_grace_proxy",
            "post_60d_active_oos_credential_signal_count",
            "post_60d_combined_title_active_credential_conflict_present",
            "post_60d_prior_state_persistence_any_present",
            "post_60d_prior_state_persistence_cross_source_present",
            "new_meaningful_oos_title_signal_post_60d_present",
            "t1_update_observation_lag_mean_days", "t1_late_arriving_update_count_over_30d",
        ]
        delta = pd.DataFrame(index=left.index)
        for column in selected:
            delta[f"delta_{column}"] = right[column] - left[column]

        updates = self.events[self.events["phase"] == "T1"].copy()
        updates = updates[
            (updates["_event_dt"].isna() | (updates["_event_dt"] <= self.t1_as_of))
            & (updates["_observed_dt"].isna() | (updates["_observed_dt"] <= self.t1_as_of))
        ]
        grouped = updates.groupby("candidate_record_id")
        direct = pd.DataFrame(index=left.index)
        direct["new_t1_record_count"] = grouped.size().reindex(left.index, fill_value=0)
        direct["new_t1_oos_count"] = grouped["_state_class"].apply(lambda values: int((values == "OOS").sum())).reindex(left.index, fill_value=0)
        direct["new_t1_de_count"] = grouped["_state_class"].apply(lambda values: int((values == "DE").sum())).reindex(left.index, fill_value=0)
        direct["new_t1_state_missing_count"] = grouped["_state_class"].apply(lambda values: int((values == "MISSING").sum())).reindex(left.index, fill_value=0)
        direct["new_t1_distinct_oos_states"] = grouped.apply(
            lambda frame: int(frame.loc[frame["_state_class"] == "OOS", "state"].replace("", np.nan).nunique()),
            include_groups=False,
        ).reindex(left.index, fill_value=0)
        direct["new_t1_source_domain_diversity"] = grouped["source_domain"].nunique().reindex(left.index, fill_value=0)
        for action in ("new_record", "record_correction", "status_update"):
            direct[f"new_t1_{action}_count"] = grouped["record_action"].apply(
                lambda values, action=action: int((values == action).sum())
            ).reindex(left.index, fill_value=0)

        t0_events = self.events[self.events["phase"] == "T0"].merge(
            self.candidates[["candidate_record_id", "_candidate_dt"]], on="candidate_record_id", how="left"
        )
        newly_effective = t0_events[
            (t0_events["_event_dt"] > t0_events["_candidate_dt"])
            & (t0_events["_event_dt"] <= self.t1_as_of)
        ].groupby("candidate_record_id").size()
        direct["newly_effective_t0_record_count"] = newly_effective.reindex(left.index, fill_value=0)
        direct["t1_has_linked_update"] = (direct["new_t1_record_count"] > 0).astype(int)
        output = direct.join(delta).reset_index()
        integer_prefixes = ("new_t1_", "newly_effective_", "t1_has_")
        for column in output.columns:
            if column.startswith(integer_prefixes):
                output[column] = output[column].astype(int)
        return output

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        t0 = self.build_phase("T0")
        t1 = self.build_phase("T1")
        delta = self.build_deltas(t0, t1)
        return t0, t1, delta
