from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import DE_STATE
from src.features.entity_resolution import EntityResolver, resolution_summary


DOMAINS = ("candidate", "address", "license", "title", "work", "external")
WINDOWS = (30, 90, 180, 365)


@dataclass
class FeatureArtifacts:
    t0: pd.DataFrame
    t1: pd.DataFrame
    timeline: pd.DataFrame
    resolution: pd.DataFrame


def _timeline_frame(
    frame: pd.DataFrame,
    domain: str,
    state_column: str,
    date_column: str,
    event_type_column: str | None = None,
    status_column: str | None = None,
    quality_column: str | None = None,
    vehicle_column: str | None = None,
    action_column: str | None = None,
    phase_available: str = "T0",
) -> pd.DataFrame:
    matched = frame.loc[frame["candidate_record_id"].notna()].copy()
    result = pd.DataFrame(
        {
            "candidate_record_id": matched["candidate_record_id"],
            "phase_available": phase_available,
            "source_domain": domain,
            "source_record_id": matched.get("source_record_id", ""),
            "event_date": pd.to_datetime(matched[date_column], errors="coerce"),
            "observed_date": pd.to_datetime(matched.get("observed_date"), errors="coerce")
            if "observed_date" in matched
            else pd.NaT,
            "state": matched[state_column].fillna("").astype(str).str.upper(),
            "event_type": matched[event_type_column].fillna("").astype(str)
            if event_type_column
            else "",
            "status": matched[status_column].fillna("").astype(str) if status_column else "",
            "quality": matched[quality_column].fillna("").astype(str) if quality_column else "",
            "vehicle_ref": matched[vehicle_column].fillna("").astype(str) if vehicle_column else "",
            "record_action": matched[action_column].fillna("").astype(str) if action_column else "",
            "match_confidence": matched.get("match_confidence", 1.0),
            "match_method": matched.get("match_method", "direct_candidate_record"),
            "identity_consistency": matched.get("identity_consistency", 1.0),
        }
    )
    return result.reset_index(drop=True)


def build_timeline(
    data: dict[str, pd.DataFrame], resolved: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    candidates = data["candidates"].copy()
    candidates["source_record_id"] = candidates["candidate_record_id"]
    candidates["match_confidence"] = 1.0
    candidates["match_method"] = "direct_candidate_record"
    candidates["identity_consistency"] = 1.0
    pieces = [
        _timeline_frame(
            candidates,
            "candidate",
            "observed_state",
            "candidate_observed_date",
            status_column="review_status",
        ),
        _timeline_frame(
            resolved["address"],
            "address",
            "state",
            "effective_start_date",
            event_type_column="source_type",
        ),
        _timeline_frame(
            resolved["license"],
            "license",
            "credential_state",
            "event_date",
            event_type_column="event_type",
            status_column="credential_status",
        ),
        _timeline_frame(
            resolved["title"],
            "title",
            "event_state",
            "event_date",
            event_type_column="event_type",
            vehicle_column="vehicle_ref",
        ),
        _timeline_frame(
            resolved["work"],
            "work",
            "work_state",
            "observed_date",
            event_type_column="source_type",
        ),
        _timeline_frame(
            resolved["external"],
            "external",
            "signal_state",
            "effective_date",
            event_type_column="signal_type",
            quality_column="evidence_quality",
        ),
        _timeline_frame(
            resolved["updates"],
            "update",
            "state",
            "effective_date",
            event_type_column="source_description",
            vehicle_column="vehicle_ref",
            action_column="record_action",
            phase_available="T1",
        ),
    ]
    timeline = pd.concat(pieces, ignore_index=True)
    timeline["state"] = timeline["state"].replace({"NAN": "", "NONE": ""})
    timeline["effective_source_domain"] = timeline["source_domain"]
    update_indices = timeline["source_domain"].eq("update")
    update_domains = resolved["updates"].loc[
        resolved["updates"]["candidate_record_id"].notna(), "source_domain"
    ].reset_index(drop=True)
    timeline.loc[update_indices, "effective_source_domain"] = update_domains.to_numpy()
    return timeline.sort_values(
        ["candidate_record_id", "event_date", "source_record_id"], na_position="last"
    ).reset_index(drop=True)


def _latest_state(events: pd.DataFrame, candidate_ids: pd.Index, prefix: str) -> pd.DataFrame:
    valid = events.loc[events["state"].ne("") & events["event_date"].notna()]
    if valid.empty:
        return pd.DataFrame({f"{prefix}_latest_state": "UNKNOWN"}, index=candidate_ids)
    latest = (
        valid.sort_values(["candidate_record_id", "event_date"])
        .groupby("candidate_record_id", sort=False)
        .tail(1)
        .set_index("candidate_record_id")["state"]
    )
    return latest.reindex(candidate_ids).fillna("UNKNOWN").rename(f"{prefix}_latest_state").to_frame()


def _aggregate_snapshot(
    candidates: pd.DataFrame, events: pd.DataFrame, phase: str, cutoff: pd.Timestamp
) -> pd.DataFrame:
    candidate_ids = pd.Index(candidates["candidate_record_id"], name="candidate_record_id")
    output = pd.DataFrame(index=candidate_ids)
    candidate_lookup = candidates.set_index("candidate_record_id")
    output["phase"] = phase
    output["phase_t1"] = 1 if phase == "T1" else 0
    output["candidate_observed_state"] = candidate_lookup["observed_state"].reindex(candidate_ids).fillna("UNKNOWN")
    output["candidate_observed_state_is_de"] = output["candidate_observed_state"].eq(DE_STATE).astype(int)

    events = events.copy()
    events["days_ago"] = (cutoff - events["event_date"]).dt.days.clip(lower=0)
    stateful = events.loc[events["state"].ne("")].copy()

    for domain in DOMAINS:
        domain_events = events.loc[events["effective_source_domain"].eq(domain)]
        group = domain_events.groupby("candidate_record_id")
        output[f"{domain}_record_count"] = group.size().reindex(candidate_ids, fill_value=0).astype(int)
        valid_state = domain_events.loc[domain_events["state"].ne("")]
        state_group = valid_state.groupby("candidate_record_id")
        output[f"{domain}_de_count"] = (
            valid_state["state"].eq(DE_STATE).groupby(valid_state["candidate_record_id"]).sum().reindex(candidate_ids, fill_value=0).astype(int)
        )
        output[f"{domain}_non_de_count"] = (
            valid_state["state"].ne(DE_STATE).groupby(valid_state["candidate_record_id"]).sum().reindex(candidate_ids, fill_value=0).astype(int)
        )
        output[f"{domain}_unique_non_de_states"] = (
            valid_state.loc[valid_state["state"].ne(DE_STATE)]
            .groupby("candidate_record_id")["state"]
            .nunique()
            .reindex(candidate_ids, fill_value=0)
            .astype(int)
        )
        latest_dates = group["event_date"].max().reindex(candidate_ids)
        output[f"days_since_latest_{domain}_evidence"] = (
            (cutoff - latest_dates).dt.days.clip(lower=0).fillna(9999).astype(int)
        )
        output = output.join(_latest_state(domain_events, candidate_ids, domain))
        output[f"{domain}_missing_state_count"] = (
            domain_events["state"].eq("")
            .groupby(domain_events["candidate_record_id"])
            .sum()
            .reindex(candidate_ids, fill_value=0)
            .astype(int)
        )
        output[f"{domain}_mean_identity_consistency"] = (
            group["identity_consistency"].mean().reindex(candidate_ids).fillna(0.0)
        )
        output[f"{domain}_low_identity_consistency_count"] = (
            domain_events["identity_consistency"].lt(0.65)
            .groupby(domain_events["candidate_record_id"])
            .sum()
            .reindex(candidate_ids, fill_value=0)
            .astype(int)
        )
        weighted = valid_state.assign(
            _weight=np.exp(-valid_state["days_ago"].clip(upper=3650) / 365.0)
        )
        output[f"{domain}_recency_weighted_de"] = (
            weighted.loc[weighted["state"].eq(DE_STATE)]
            .groupby("candidate_record_id")["_weight"]
            .sum()
            .reindex(candidate_ids, fill_value=0.0)
        )
        output[f"{domain}_recency_weighted_non_de"] = (
            weighted.loc[weighted["state"].ne(DE_STATE)]
            .groupby("candidate_record_id")["_weight"]
            .sum()
            .reindex(candidate_ids, fill_value=0.0)
        )
        output[f"{domain}_recency_weighted_state_margin"] = (
            output[f"{domain}_recency_weighted_non_de"]
            - output[f"{domain}_recency_weighted_de"]
        )
        for window in WINDOWS:
            recent = domain_events.loc[domain_events["days_ago"].le(window)]
            output[f"{domain}_records_{window}d"] = (
                recent.groupby("candidate_record_id").size().reindex(candidate_ids, fill_value=0).astype(int)
            )
            output[f"{domain}_de_records_{window}d"] = (
                recent.loc[recent["state"].eq(DE_STATE)]
                .groupby("candidate_record_id")
                .size()
                .reindex(candidate_ids, fill_value=0)
                .astype(int)
            )
            output[f"{domain}_non_de_records_{window}d"] = (
                recent.loc[recent["state"].ne("") & recent["state"].ne(DE_STATE)]
                .groupby("candidate_record_id")
                .size()
                .reindex(candidate_ids, fill_value=0)
                .astype(int)
            )

    state_counts = (
        stateful.groupby(["candidate_record_id", "state"]).size().rename("count").reset_index()
    )
    if not state_counts.empty:
        dominant_rows = state_counts.sort_values(
            ["candidate_record_id", "count", "state"], ascending=[True, False, True]
        ).drop_duplicates("candidate_record_id")
        dominant = dominant_rows.set_index("candidate_record_id")["state"]
        max_count = dominant_rows.set_index("candidate_record_id")["count"]
    else:
        dominant = pd.Series(dtype=object)
        max_count = pd.Series(dtype=float)
    output["dominant_evidence_state"] = dominant.reindex(candidate_ids).fillna("UNKNOWN")
    output["de_signal_count"] = (
        stateful["state"].eq(DE_STATE).groupby(stateful["candidate_record_id"]).sum().reindex(candidate_ids, fill_value=0).astype(int)
    )
    output["non_de_signal_count"] = (
        stateful["state"].ne(DE_STATE).groupby(stateful["candidate_record_id"]).sum().reindex(candidate_ids, fill_value=0).astype(int)
    )
    output["unique_non_de_states"] = (
        stateful.loc[stateful["state"].ne(DE_STATE)]
        .groupby("candidate_record_id")["state"]
        .nunique()
        .reindex(candidate_ids, fill_value=0)
        .astype(int)
    )
    output["total_state_signal_count"] = output["de_signal_count"] + output["non_de_signal_count"]
    output["event_state_agreement_score"] = (
        max_count.reindex(candidate_ids).fillna(0)
        / output["total_state_signal_count"].replace(0, np.nan)
    ).fillna(0.0)

    source_state = stateful.drop_duplicates(
        ["candidate_record_id", "effective_source_domain", "state"]
    )
    source_support = (
        source_state.groupby(["candidate_record_id", "state"])["effective_source_domain"]
        .nunique()
        .rename("sources")
        .reset_index()
    )
    max_source_agreement = source_support.groupby("candidate_record_id")["sources"].max()
    sources_present = (
        events.groupby("candidate_record_id")["effective_source_domain"]
        .nunique()
        .reindex(candidate_ids, fill_value=0)
    )
    output["independent_evidence_source_types"] = sources_present.astype(int)
    output["sources_supporting_de"] = (
        source_support.loc[source_support["state"].eq(DE_STATE)]
        .groupby("candidate_record_id")["sources"]
        .max()
        .reindex(candidate_ids, fill_value=0)
        .astype(int)
    )
    output["sources_supporting_non_de"] = (
        source_state.loc[source_state["state"].ne(DE_STATE)]
        .groupby("candidate_record_id")["effective_source_domain"]
        .nunique()
        .reindex(candidate_ids, fill_value=0)
        .astype(int)
    )
    non_de_state_sources = source_support.loc[source_support["state"].ne(DE_STATE)]
    max_non_de_agreement = non_de_state_sources.groupby("candidate_record_id")["sources"].max()
    output["max_sources_same_non_de_state"] = max_non_de_agreement.reindex(candidate_ids, fill_value=0).astype(int)
    output["multiple_sources_same_non_de_state"] = output["max_sources_same_non_de_state"].ge(2).astype(int)
    output["cross_source_agreement_score"] = (
        max_source_agreement.reindex(candidate_ids).fillna(0)
        / sources_present.replace(0, np.nan)
    ).fillna(0.0)
    output["conflicting_evidence_score"] = (
        np.minimum(output["sources_supporting_de"], output["sources_supporting_non_de"])
        / np.maximum(output["sources_supporting_de"], output["sources_supporting_non_de"]).replace(0, np.nan)
    ).fillna(0.0)
    output["de_non_de_conflict"] = (
        output["sources_supporting_de"].gt(0) & output["sources_supporting_non_de"].gt(0)
    ).astype(int)
    output["evidence_completeness_score"] = (sources_present / len(DOMAINS)).clip(0, 1)
    output["missing_evidence_categories"] = (len(DOMAINS) - sources_present).clip(lower=0).astype(int)
    output["total_missing_state_records"] = (
        sum(output[f"{domain}_missing_state_count"] for domain in DOMAINS)
    )
    output["mean_identity_consistency"] = (
        events.groupby("candidate_record_id")["identity_consistency"]
        .mean()
        .reindex(candidate_ids)
        .fillna(0.0)
    )
    output["low_identity_consistency_count"] = (
        events["identity_consistency"].lt(0.65)
        .groupby(events["candidate_record_id"])
        .sum()
        .reindex(candidate_ids, fill_value=0)
        .astype(int)
    )

    latest_any = events.groupby("candidate_record_id")["event_date"].max().reindex(candidate_ids)
    output["days_since_most_recent_evidence"] = (
        (cutoff - latest_any).dt.days.clip(lower=0).fillna(9999).astype(int)
    )
    output["evidence_recency_score"] = np.exp(
        -output["days_since_most_recent_evidence"].clip(upper=3650) / 365.0
    )

    license_events = events.loc[events["effective_source_domain"].eq("license")]
    for status in ("active", "expired", "superseded", "unknown"):
        mask = license_events["status"].eq(status)
        output[f"credential_{status}_count"] = (
            license_events.loc[mask].groupby("candidate_record_id").size().reindex(candidate_ids, fill_value=0).astype(int)
        )
    output["active_non_de_credential"] = (
        license_events["status"].eq("active") & license_events["state"].ne("") & license_events["state"].ne(DE_STATE)
    ).groupby(license_events["candidate_record_id"]).any().reindex(candidate_ids, fill_value=False).astype(int)
    active_valid = license_events.loc[
        license_events["status"].eq("active")
        & license_events["state"].ne("")
        & license_events["event_date"].notna()
    ]
    active_latest = (
        active_valid.sort_values(["candidate_record_id", "event_date"])
        .groupby("candidate_record_id")
        .tail(1)
        .set_index("candidate_record_id")["state"]
    )
    output["active_credential_latest_state"] = active_latest.reindex(candidate_ids).fillna("UNKNOWN")
    output["active_credential_latest_non_de"] = (
        output["active_credential_latest_state"].ne("UNKNOWN")
        & output["active_credential_latest_state"].ne(DE_STATE)
    ).astype(int)

    title_events = events.loc[events["effective_source_domain"].eq("title")]
    output["non_de_title_event_indicator"] = (
        title_events["state"].ne("") & title_events["state"].ne(DE_STATE)
    ).groupby(title_events["candidate_record_id"]).any().reindex(candidate_ids, fill_value=False).astype(int)
    output["distinct_vehicle_count"] = (
        title_events.loc[title_events["vehicle_ref"].ne("")]
        .groupby("candidate_record_id")["vehicle_ref"]
        .nunique()
        .reindex(candidate_ids, fill_value=0)
        .astype(int)
    )

    address_events = events.loc[events["effective_source_domain"].eq("address")].sort_values(
        ["candidate_record_id", "event_date"]
    )
    previous = address_events.groupby("candidate_record_id")["state"].shift()
    transition = address_events["state"].ne("") & previous.notna() & address_events["state"].ne(previous)
    output["address_state_transitions"] = (
        transition.groupby(address_events["candidate_record_id"]).sum().reindex(candidate_ids, fill_value=0).astype(int)
    )
    output["repeated_non_de_address_evidence"] = output["address_non_de_count"].ge(2).astype(int)

    external_events = events.loc[events["effective_source_domain"].eq("external")]
    for quality in ("standard", "limited"):
        output[f"external_{quality}_quality_count"] = (
            external_events.loc[external_events["quality"].eq(quality)]
            .groupby("candidate_record_id")
            .size()
            .reindex(candidate_ids, fill_value=0)
            .astype(int)
        )
        output[f"external_{quality}_non_de_count"] = (
            external_events.loc[
                external_events["quality"].eq(quality)
                & external_events["state"].ne("")
                & external_events["state"].ne(DE_STATE)
            ]
            .groupby("candidate_record_id")
            .size()
            .reindex(candidate_ids, fill_value=0)
            .astype(int)
        )
    ext_total = output["external_standard_quality_count"] + output["external_limited_quality_count"]
    quality_score = (
        (output["external_standard_quality_count"] + 0.5 * output["external_limited_quality_count"])
        / ext_total.replace(0, np.nan)
    ).fillna(0.5)
    output["evidence_strength_score"] = (
        0.38 * output["evidence_completeness_score"]
        + 0.26 * output["cross_source_agreement_score"]
        + 0.21 * output["evidence_recency_score"]
        + 0.15 * quality_score
    ).clip(0, 1)

    output["independent_non_de_sources_same_state"] = output["max_sources_same_non_de_state"]
    output["work_non_de_support_only"] = (
        output["work_non_de_count"].gt(0) & output["sources_supporting_non_de"].le(1)
    ).astype(int)
    latest_columns = [f"{domain}_latest_state" for domain in DOMAINS]
    output["latest_source_states_de"] = output[latest_columns].eq(DE_STATE).sum(axis=1)
    output["latest_source_states_non_de"] = (
        output[latest_columns].ne("UNKNOWN") & output[latest_columns].ne(DE_STATE)
    ).sum(axis=1)
    output["latest_source_states_matching_dominant"] = output[latest_columns].eq(
        output["dominant_evidence_state"], axis=0
    ).sum(axis=1)
    output["candidate_matches_dominant_state"] = output["candidate_observed_state"].eq(
        output["dominant_evidence_state"]
    ).astype(int)
    output["non_de_state_source_concentration"] = (
        output["max_sources_same_non_de_state"]
        / output["sources_supporting_non_de"].replace(0, np.nan)
    ).fillna(0.0)
    return output.reset_index()


def build_features(data: dict[str, pd.DataFrame]) -> FeatureArtifacts:
    resolver = EntityResolver(data["candidates"])
    resolved = resolver.resolve_all(data)
    timeline = build_timeline(data, resolved)
    t0_events = timeline.loc[timeline["phase_available"].eq("T0")].copy()
    t1_events = timeline.copy()
    t0_cutoff = t0_events["event_date"].max()
    observed_t1_max = timeline.loc[timeline["phase_available"].eq("T1"), "observed_date"].max()
    t1_cutoff = observed_t1_max if pd.notna(observed_t1_max) else t1_events["event_date"].max()
    t0 = _aggregate_snapshot(data["candidates"], t0_events, "T0", t0_cutoff)
    t1 = _aggregate_snapshot(data["candidates"], t1_events, "T1", t1_cutoff)

    candidate_ids = pd.Index(data["candidates"]["candidate_record_id"], name="candidate_record_id")
    updates = timeline.loc[timeline["phase_available"].eq("T1")].copy()
    update_group = updates.groupby("candidate_record_id")
    t0_idx = t0.set_index("candidate_record_id")
    t1_idx = t1.set_index("candidate_record_id")
    for frame, is_t1 in ((t0_idx, False), (t1_idx, True)):
        frame["new_t1_record_count"] = (
            update_group.size().reindex(candidate_ids, fill_value=0).astype(int) if is_t1 else 0
        )
        for action, column in (
            ("new_record", "t1_new_records"),
            ("record_correction", "t1_corrections"),
            ("status_update", "t1_status_updates"),
        ):
            counts = (
                updates.loc[updates["record_action"].eq(action)]
                .groupby("candidate_record_id")
                .size()
                .reindex(candidate_ids, fill_value=0)
                .astype(int)
            )
            frame[column] = counts if is_t1 else 0

    valid_updates = updates.loc[updates["state"].ne("")].copy()
    t0_dominant = t0_idx["dominant_evidence_state"]
    valid_updates["t0_dominant"] = valid_updates["candidate_record_id"].map(t0_dominant)
    valid_updates["reinforces"] = valid_updates["state"].eq(valid_updates["t0_dominant"])
    valid_updates["contradicts"] = valid_updates["t0_dominant"].ne("UNKNOWN") & ~valid_updates["reinforces"]
    reinforces = valid_updates.groupby("candidate_record_id")["reinforces"].any().reindex(candidate_ids, fill_value=False)
    contradicts = valid_updates.groupby("candidate_record_id")["contradicts"].any().reindex(candidate_ids, fill_value=False)
    t0_idx["t1_reinforces_t0"] = 0
    t0_idx["t1_contradicts_t0"] = 0
    t1_idx["t1_reinforces_t0"] = reinforces.astype(int)
    t1_idx["t1_contradicts_t0"] = contradicts.astype(int)
    t0_idx["t1_resolves_uncertainty"] = 0
    t1_idx["t1_resolves_uncertainty"] = (
        (t0_idx["conflicting_evidence_score"] >= 0.5)
        & (t1_idx["conflicting_evidence_score"] <= t0_idx["conflicting_evidence_score"] - 0.15)
    ).astype(int)
    t0_idx["change_in_evidence_strength"] = 0.0
    t0_idx["change_in_state_agreement"] = 0.0
    t1_idx["change_in_evidence_strength"] = (
        t1_idx["evidence_strength_score"] - t0_idx["evidence_strength_score"]
    )
    t1_idx["change_in_state_agreement"] = (
        t1_idx["cross_source_agreement_score"] - t0_idx["cross_source_agreement_score"]
    )
    return FeatureArtifacts(
        t0=t0_idx.reset_index(),
        t1=t1_idx.reset_index(),
        timeline=timeline,
        resolution=resolution_summary(resolved),
    )
