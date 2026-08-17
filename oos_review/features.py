"""Per-case feature table at T0 and T1.

Why these features
------------------
Raw row counts and overall DE-share do not separate the three review classes.
On the 300 labeled cases, **which state is more recent** does:

* ``review_warranted`` — Delaware address / license / title facts are newer
* ``review_not_warranted`` — out-of-state facts are newer
* ``insufficient_evidence`` — the two sides are mixed or thin

Each source therefore contributes a recency vote (DE-newer = +1, OOS-newer = -1,
tie/missing = 0). Votes are weighted by how directly the source speaks to
residency or the vehicle, then combined with whether the *current* address
is in Delaware. The resulting ``de_oos_score`` is the input to the transparent
rule baseline in ``baseline.py``.

T0 vs T1
--------
T0 uses only ``Data_T0`` linked tables. T1 appends ``evidence_update_stream``
rows into the matching domain (address, license, title, external) and treats
the latest T1 address, if any, as the current address. Work has no T1 domain.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DOMAIN_WEIGHTS = {
    "address": 2.0,
    "license": 1.5,
    "title": 2.0,
    "work": 1.0,
    "external": 0.5,
}
OPEN_ADDRESS_WEIGHT = 1.5

_T1_DOMAIN = {
    "address": "address",
    "license": "license",
    "title": "title",
    "external": "external",
}


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _linked(frame: pd.DataFrame) -> pd.DataFrame:
    if "candidate_record_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    out = frame.loc[frame["candidate_record_id"].notna()].copy()
    out["candidate_record_id"] = out["candidate_record_id"].astype(str)
    return out


def _t1_domain_events(
    t1: pd.DataFrame,
    domain: str,
    date_col: str,
    state_col: str,
) -> pd.DataFrame:
    """Normalize T1 rows for one source_domain onto T0 column names."""
    rows = t1.loc[t1["source_domain"] == domain].copy()
    if rows.empty:
        return pd.DataFrame(columns=["candidate_record_id", date_col, state_col])
    event_date = _to_datetime(rows["effective_date"])
    observed = _to_datetime(rows["observed_date"])
    rows[date_col] = event_date.fillna(observed)
    rows[state_col] = rows["state"]
    return rows[["candidate_record_id", date_col, state_col]]


def _with_t1(
    t0: pd.DataFrame,
    t1: Optional[pd.DataFrame],
    *,
    domain: str,
    date_col: str,
    state_col: str,
) -> pd.DataFrame:
    cols = ["candidate_record_id", date_col, state_col]
    base = t0[cols].copy() if not t0.empty else pd.DataFrame(columns=cols)
    if t1 is None or domain not in _T1_DOMAIN:
        return base
    extra = _t1_domain_events(t1, _T1_DOMAIN[domain], date_col, state_col)
    return pd.concat([base, extra], ignore_index=True)


def _recency_block(
    events: pd.DataFrame,
    *,
    date_col: str,
    state_col: str,
    prefix: str,
    candidates: pd.Index,
) -> pd.DataFrame:
    """Latest state, last DE/OOS dates, and a -1/0/+1 recency vote."""
    empty = pd.DataFrame(index=candidates)
    empty[f"n_{prefix}"] = 0
    empty[f"latest_{prefix}_state"] = pd.NA
    empty[f"last_de_{prefix}_date"] = pd.NaT
    empty[f"last_oos_{prefix}_date"] = pd.NaT
    empty[f"{prefix}_recency_vote"] = 0
    if events.empty:
        return empty

    ev = events.dropna(subset=["candidate_record_id", state_col]).copy()
    ev[date_col] = _to_datetime(ev[date_col])
    ev = ev.dropna(subset=[date_col])
    if ev.empty:
        return empty

    n = ev.groupby("candidate_record_id").size()
    latest = (
        ev.sort_values(date_col)
        .groupby("candidate_record_id")
        .tail(1)
        .set_index("candidate_record_id")[state_col]
    )
    last_de = (
        ev.loc[ev[state_col] == "DE"]
        .sort_values(date_col)
        .groupby("candidate_record_id")
        .tail(1)
        .set_index("candidate_record_id")[date_col]
    )
    last_oos = (
        ev.loc[ev[state_col] != "DE"]
        .sort_values(date_col)
        .groupby("candidate_record_id")
        .tail(1)
        .set_index("candidate_record_id")[date_col]
    )

    out = pd.DataFrame(index=candidates)
    out[f"n_{prefix}"] = n.reindex(candidates).fillna(0).astype(int)
    out[f"latest_{prefix}_state"] = latest.reindex(candidates)
    out[f"last_de_{prefix}_date"] = last_de.reindex(candidates)
    out[f"last_oos_{prefix}_date"] = last_oos.reindex(candidates)

    de_d = out[f"last_de_{prefix}_date"]
    oos_d = out[f"last_oos_{prefix}_date"]
    vote = pd.Series(0, index=candidates, dtype=int)
    vote[de_d.notna() & (oos_d.isna() | (de_d > oos_d))] = 1
    vote[oos_d.notna() & (de_d.isna() | (oos_d > de_d))] = -1
    out[f"{prefix}_recency_vote"] = vote
    return out


def _current_address(
    addr: pd.DataFrame,
    t1: Optional[pd.DataFrame],
    *,
    phase: str,
    candidates: pd.Index,
) -> pd.Series:
    """Best current residential state.

    T0: latest open T0 address (null end date).
    T1: latest T1 address row if the case has one, otherwise the T0 open address.
    """
    open_state = pd.Series(pd.NA, index=candidates, dtype="object")
    if not addr.empty and "effective_end_date" in addr.columns:
        open_rows = addr.loc[_to_datetime(addr["effective_end_date"]).isna()].copy()
        if not open_rows.empty:
            open_rows["effective_start_date"] = _to_datetime(
                open_rows["effective_start_date"]
            )
            latest_open = (
                open_rows.sort_values("effective_start_date")
                .groupby("candidate_record_id")
                .tail(1)
                .set_index("candidate_record_id")["state"]
            )
            open_state = latest_open.reindex(candidates)

    if phase != "T1" or t1 is None:
        return open_state

    t1_addr = t1.loc[t1["source_domain"] == "address"].copy()
    if t1_addr.empty:
        return open_state
    t1_addr["_when"] = _to_datetime(t1_addr["effective_date"]).fillna(
        _to_datetime(t1_addr["observed_date"])
    )
    latest_t1 = (
        t1_addr.sort_values("_when")
        .groupby("candidate_record_id")
        .tail(1)
        .set_index("candidate_record_id")["state"]
    )
    latest_t1.index = latest_t1.index.astype(str)
    combined = open_state.copy()
    overlap = latest_t1.index.intersection(combined.index)
    combined.loc[overlap] = latest_t1.loc[overlap]
    return combined.reindex(candidates)


def _latest_license_status(
    lic: pd.DataFrame,
    candidates: pd.Index,
) -> pd.Series:
    if lic.empty or "credential_status" not in lic.columns:
        return pd.Series(pd.NA, index=candidates, dtype="object")
    rows = lic.dropna(subset=["candidate_record_id"]).copy()
    rows["event_date"] = _to_datetime(rows["event_date"])
    rows = rows.dropna(subset=["event_date"])
    if rows.empty:
        return pd.Series(pd.NA, index=candidates, dtype="object")
    latest = (
        rows.sort_values("event_date")
        .groupby("candidate_record_id")
        .tail(1)
        .set_index("candidate_record_id")["credential_status"]
    )
    return latest.reindex(candidates)


def build_case_features(
    candidates: pd.DataFrame,
    linked_sources: dict[str, pd.DataFrame],
    *,
    phase: str,
    t1_stream: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """One feature row per candidate for ``phase`` ``T0`` or ``T1``."""
    if phase not in {"T0", "T1"}:
        raise ValueError("phase must be T0 or T1")

    ids = pd.Index(candidates["candidate_record_id"].astype(str), name="candidate_record_id")
    out = pd.DataFrame(index=ids)
    out["phase"] = phase
    out["observed_state"] = candidates.set_index("candidate_record_id")[
        "observed_state"
    ].reindex(ids)
    out["observed_is_de"] = out["observed_state"].eq("DE")
    out["candidate_observed_date"] = _to_datetime(
        candidates.set_index("candidate_record_id")["candidate_observed_date"]
    ).reindex(ids)

    addr = _linked(linked_sources.get("address_history", pd.DataFrame()))
    lic = _linked(linked_sources.get("license_id_events", pd.DataFrame()))
    ttl = _linked(linked_sources.get("vehicle_title_events", pd.DataFrame()))
    wrk = _linked(linked_sources.get("work_location_signals", pd.DataFrame()))
    ext = _linked(linked_sources.get("external_context_signals", pd.DataFrame()))
    t1 = None
    if phase == "T1":
        raw_t1 = t1_stream if t1_stream is not None else linked_sources.get(
            "evidence_update_stream"
        )
        t1 = _linked(raw_t1 if raw_t1 is not None else pd.DataFrame())

    addr_events = _with_t1(
        addr, t1, domain="address", date_col="effective_start_date", state_col="state"
    )
    lic_events = _with_t1(
        lic, t1, domain="license", date_col="event_date", state_col="credential_state"
    )
    ttl_events = _with_t1(
        ttl, t1, domain="title", date_col="event_date", state_col="event_state"
    )
    wrk_events = wrk
    ext_events = _with_t1(
        ext, t1, domain="external", date_col="effective_date", state_col="signal_state"
    )

    blocks = [
        _recency_block(
            addr_events, date_col="effective_start_date", state_col="state",
            prefix="address", candidates=ids,
        ),
        _recency_block(
            lic_events, date_col="event_date", state_col="credential_state",
            prefix="license", candidates=ids,
        ),
        _recency_block(
            ttl_events, date_col="event_date", state_col="event_state",
            prefix="title", candidates=ids,
        ),
        _recency_block(
            wrk_events, date_col="observed_date", state_col="work_state",
            prefix="work", candidates=ids,
        ),
        _recency_block(
            ext_events, date_col="effective_date", state_col="signal_state",
            prefix="external", candidates=ids,
        ),
    ]
    for block in blocks:
        out = out.join(block)

    open_addr = _current_address(addr, t1, phase=phase, candidates=ids)
    out["open_address_state"] = open_addr
    out["has_open_address"] = open_addr.notna()
    out["open_address_is_de"] = open_addr.eq("DE")

    out["latest_license_status"] = _latest_license_status(lic, ids)
    if phase == "T1" and t1 is not None:
        t1_lic_ids = t1.loc[t1["source_domain"] == "license", "candidate_record_id"]
        out.loc[out.index.isin(t1_lic_ids.astype(str)), "latest_license_status"] = pd.NA

    out["latest_license_is_de"] = out["latest_license_state"].eq("DE")
    out["has_active_de_license"] = (
        out["latest_license_is_de"] & out["latest_license_status"].eq("active")
    )
    out["latest_title_is_de"] = out["latest_title_state"].eq("DE")
    out["latest_work_is_de"] = out["latest_work_state"].eq("DE")
    out["latest_external_is_de"] = out["latest_external_state"].eq("DE")

    out["n_current_de_ties"] = (
        out["open_address_is_de"].astype(int)
        + out["has_active_de_license"].astype(int)
        + out["latest_title_is_de"].astype(int)
        + out["latest_work_is_de"].astype(int)
    )
    out["n_current_oos_ties"] = (
        (out["has_open_address"] & ~out["open_address_is_de"]).astype(int)
        + (
            out["latest_license_state"].notna()
            & ~out["latest_license_is_de"]
            & out["latest_license_status"].eq("active")
        ).astype(int)
        + (out["latest_title_state"].notna() & ~out["latest_title_is_de"]).astype(int)
        + (out["latest_work_state"].notna() & ~out["latest_work_is_de"]).astype(int)
    )
    out["has_state_conflict"] = (out["n_current_de_ties"] > 0) & (
        out["n_current_oos_ties"] > 0
    )

    current_states = out[
        [
            "open_address_state",
            "latest_license_state",
            "latest_title_state",
            "latest_work_state",
            "observed_state",
        ]
    ]
    out["n_distinct_current_states"] = current_states.apply(
        lambda row: pd.Series(row.dropna().unique()).nunique(), axis=1
    )
    out["oos_observed_open_de"] = (~out["observed_is_de"]) & out["open_address_is_de"]

    out["n_de_newer_sources"] = (
        (out["address_recency_vote"] == 1).astype(int)
        + (out["license_recency_vote"] == 1).astype(int)
        + (out["title_recency_vote"] == 1).astype(int)
        + (out["work_recency_vote"] == 1).astype(int)
        + (out["external_recency_vote"] == 1).astype(int)
    )
    out["n_oos_newer_sources"] = (
        (out["address_recency_vote"] == -1).astype(int)
        + (out["license_recency_vote"] == -1).astype(int)
        + (out["title_recency_vote"] == -1).astype(int)
        + (out["work_recency_vote"] == -1).astype(int)
        + (out["external_recency_vote"] == -1).astype(int)
    )
    out["n_recency_votes"] = out["n_de_newer_sources"] + out["n_oos_newer_sources"]
    out["n_sources_present"] = (
        (out["n_address"] > 0).astype(int)
        + (out["n_license"] > 0).astype(int)
        + (out["n_title"] > 0).astype(int)
        + (out["n_work"] > 0).astype(int)
        + (out["n_external"] > 0).astype(int)
    )

    if t1 is None or t1.empty:
        out["n_t1"] = 0
    else:
        out["n_t1"] = (
            t1.groupby("candidate_record_id").size().reindex(ids).fillna(0).astype(int)
        )

    match_frames = []
    for frame in (addr, lic, ttl, wrk, ext):
        if not frame.empty and "match_score" in frame.columns:
            match_frames.append(frame[["candidate_record_id", "match_score"]])
    if match_frames:
        scores = pd.concat(match_frames, ignore_index=True)
        out["mean_match_score"] = (
            scores.groupby("candidate_record_id")["match_score"].mean().reindex(ids)
        )
    else:
        out["mean_match_score"] = np.nan

    score = (
        out["address_recency_vote"] * DOMAIN_WEIGHTS["address"]
        + out["license_recency_vote"] * DOMAIN_WEIGHTS["license"]
        + out["title_recency_vote"] * DOMAIN_WEIGHTS["title"]
        + out["work_recency_vote"] * DOMAIN_WEIGHTS["work"]
        + out["external_recency_vote"] * DOMAIN_WEIGHTS["external"]
        + out["open_address_is_de"].astype(float) * OPEN_ADDRESS_WEIGHT
        - (out["has_open_address"] & ~out["open_address_is_de"]).astype(float)
        * OPEN_ADDRESS_WEIGHT
    )
    out["de_oos_score"] = score.round(4)

    return out.reset_index()


def build_t0_t1_features(
    candidates: pd.DataFrame,
    linked_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Stack T0 and T1 feature rows (two rows per candidate)."""
    t0 = build_case_features(candidates, linked_sources, phase="T0")
    t1 = build_case_features(
        candidates,
        linked_sources,
        phase="T1",
        t1_stream=linked_sources.get("evidence_update_stream"),
    )
    return pd.concat([t0, t1], ignore_index=True)
