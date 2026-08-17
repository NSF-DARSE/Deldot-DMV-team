"""Data loading, entity resolution and the tidy evidence table."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from alias import learn_aliases, link_by_alias
from resolve import Resolver, link_by_name, link_titles_by_vehicle, vehicle_owner_map

_HERE = Path(__file__).resolve().parent
_SEARCH = [
    os.environ.get("OOS_DATA_DIR"),
    _HERE / "Identify_Out_of_State_Tag_Holders",
    _HERE.parent / "Identify_Out_of_State_Tag_Holders",
    Path.home() / "Downloads/Identify_Out_of_State_Tag_Holders_extracted"
    / "Identify_Out_of_State_Tag_Holders",
]


def _find_data_dir():
    for p in _SEARCH:
        if p and (Path(p) / "Data_T0/candidate_records.csv").exists():
            return str(Path(p)) + "/"
    raise FileNotFoundError(
        "Challenge data not found. Put the Identify_Out_of_State_Tag_Holders folder next "
        "to this file, or set OOS_DATA_DIR to its path.")


BASE = _find_data_dir()

DOMAIN_TO_STREAM = {"address": "adr", "license": "lic", "title": "ttl", "external": "ext"}


def load(base=BASE):
    r = lambda p: pd.read_csv(base + p)
    return dict(
        cand=r("Data_T0/candidate_records.csv"),
        adr=r("Data_T0/address_history.csv"),
        lic=r("Data_T0/license_id_events.csv"),
        ttl=r("Data_T0/vehicle_title_events.csv"),
        ext=r("Data_T0/external_context_signals.csv"),
        wrk=r("Data_T0/work_location_signals.csv"),
        up=r("Data_T1/evidence_update_stream.csv"),
    )


def _frame(sub, ci, wt, stream, statecol, datecol, is_t1, extras=None):
    t = pd.DataFrame({
        "cand": ci,
        "w": wt,
        "stream": stream,
        "state": sub[statecol].values,
        "date": pd.to_datetime(sub[datecol], errors="coerce").values,
        "is_t1": is_t1,
    })
    for k, v in (extras or {}).items():
        t[k] = sub[v].values if isinstance(v, str) else v
    return t


def build_evidence(D, verbose=True):
    """Return the tidy evidence table: one row per (source record, candidate) link."""
    cand = D["cand"].reset_index(drop=True)
    res = Resolver(cand)
    frames = []

    # Titles resolve first: vehicle_ref both recovers their own replaced-name rows and
    # exposes each candidate's alias, which the keyless feeds then use (see alias.py).
    ttl = D["ttl"]
    t_ri, t_ci, t_wt = link_titles_by_vehicle(res, ttl)
    alias_map = learn_aliases(res, ttl)
    if verbose:
        print(f"  aliases learned from vehicle_ref: {len(alias_map)}")

    name_specs = [
        ("adr", "first_name", "last_name", None, "state", "effective_start_date",
         {"kind": "source_type"}),
        ("lic", "first_name", "last_name", "date_of_birth", "credential_state", "event_date",
         {"kind": "event_type", "status": "credential_status"}),
        ("ext", "first_name", "last_name", None, "signal_state", "effective_date",
         {"kind": "signal_type", "quality": "evidence_quality", "desc": "source_description"}),
        ("wrk", "first_name", "last_name", None, "work_state", "observed_date",
         {"kind": "source_type"}),
    ]
    for stream, fcol, lcol, dcol, scol, dtcol, extras in name_specs:
        df = D[stream]
        ri, ci, wt = link_by_name(res, df, fcol, lcol, dcol)
        claimed = np.zeros(len(df), bool)
        claimed[ri] = True
        a_ri, a_ci, a_wt = link_by_alias(alias_map, df, fcol, lcol, claimed)
        ri = np.concatenate([ri, a_ri])
        ci = np.concatenate([ci, a_ci])
        wt = np.concatenate([wt, a_wt])
        sub = df.iloc[ri].reset_index(drop=True)
        t = _frame(sub, ci, wt, stream, scol, dtcol, False, extras)
        if stream == "adr":
            t["open_ended"] = sub.effective_end_date.isna().values
        frames.append(t)
        if verbose:
            print(f"  {stream}: {len(set(ri))}/{len(df)} source rows linked "
                  f"({len(a_ri)} via alias)")

    ri, ci, wt = t_ri, t_ci, t_wt
    sub = ttl.iloc[ri].reset_index(drop=True)
    frames.append(_frame(sub, ci, wt, "ttl", "event_state", "event_date", False,
                         {"kind": "event_type", "vehicle_ref": "vehicle_ref"}))
    vmap = vehicle_owner_map(ri, ci, wt, ttl)
    if verbose:
        print(f"  ttl: {len(set(ri))}/{len(ttl)} source rows linked (via vehicle_ref)")

    # ---- T1 update stream -------------------------------------------------
    up = D["up"].reset_index(drop=True)
    has_veh = up.vehicle_ref.notna().values & up.vehicle_ref.map(lambda v: v in vmap).values
    ri, ci, wt = [], [], []
    for r in np.nonzero(has_veh)[0]:                       # vehicle-anchored updates
        i, conf = vmap[up.vehicle_ref.values[r]]
        ri.append(r); ci.append(i); wt.append(min(1.0, 0.6 + 0.4 * conf))
    rest = np.nonzero(~has_veh)[0]
    r2, c2, w2 = link_by_name(res, up.iloc[rest], "first_name", "last_name")
    ri = np.concatenate([np.array(ri, int), rest[r2]])
    ci = np.concatenate([np.array(ci, int), c2])
    wt = np.concatenate([np.array(wt, float), w2])
    claimed = np.zeros(len(up), bool)
    claimed[ri] = True
    a_ri, a_ci, a_wt = link_by_alias(alias_map, up, "first_name", "last_name", claimed)
    ri = np.concatenate([ri, a_ri])
    ci = np.concatenate([ci, a_ci])
    wt = np.concatenate([wt, a_wt])
    sub = up.iloc[ri].reset_index(drop=True)
    t = _frame(sub, ci, wt, None, "state", "effective_date", True,
               {"kind": "record_action", "vehicle_ref": "vehicle_ref"})
    t["stream"] = sub.source_domain.map(DOMAIN_TO_STREAM).values
    frames.append(t)
    if verbose:
        print(f"  up : {len(set(ri))}/{len(up)} source rows linked "
              f"({int(has_veh.sum())} via vehicle_ref)")

    ev = pd.concat(frames, ignore_index=True)
    for c in ("quality", "status", "open_ended", "vehicle_ref", "kind", "desc"):
        if c not in ev:
            ev[c] = np.nan
    ev["date"] = pd.to_datetime(ev["date"])
    return ev


if __name__ == "__main__":
    D = load()
    ev = build_evidence(D)
    ev.to_pickle("evidence.pkl")
    print("evidence rows:", len(ev))
