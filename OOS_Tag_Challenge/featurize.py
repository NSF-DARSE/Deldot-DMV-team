"""Per-candidate, per-phase features built from the resolved evidence table.

Every feature is an aggregate of one candidate's linked evidence.  Naming:

    {stream}_sh{H}   recency-weighted share of records whose state is DE,
                     using an exponential half-life of H days
    {stream}_last_*  the most recent state-bearing record in that stream
    ev_*             evidence sufficiency / quality, independent of direction

`core` pools the four state-bearing feeds that carry signal (address, licence,
title, external).  The work-location feed is kept as its own block but excluded
from `core`: it is clean, complete data with no relationship to the labels.
"""
import numpy as np
import pandas as pd

HOME = "DE"
CORE = ["adr", "lic", "ttl", "ext"]
ALL_STREAMS = CORE + ["wrk"]
HALFLIVES = (90, 150, 365)


def _bc(idx, w, n):
    return np.bincount(idx, weights=w, minlength=n)


def _share(sub, n, H=None):
    """Weighted DE share; recency-weighted when H is given."""
    w = sub.w.values if H is None else sub.w.values * 0.5 ** (sub.age.values / H)
    num = _bc(sub.cand.values, w * sub.de.values, n)
    den = _bc(sub.cand.values, w, n)
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)


def build_features(ev, cand, phase, t1_ref):
    n = len(cand)
    if phase == "T0":
        e = ev[~ev.is_t1].copy()
        ref = pd.to_datetime(cand.candidate_observed_date).values
    else:
        e = ev.copy()
        ref = np.repeat(np.datetime64(t1_ref), n)

    e["age"] = ((ref[e.cand.values] - e.date.values) / np.timedelta64(1, "D")).clip(min=0)
    e["has_state"] = e.state.notna().values
    st = e[e.has_state].copy()
    st["de"] = (st.state.values == HOME).astype(float)

    F = pd.DataFrame(index=range(n))
    F["obs_state_oos"] = (cand.observed_state.values != HOME).astype(float)

    def block(sub, pre, halflives=HALFLIVES):
        for H in halflives:
            F[f"{pre}_sh{H}"] = _share(sub, n, H)
        F[f"{pre}_shflat"] = _share(sub, n)
        F[f"{pre}_n"] = _bc(sub.cand.values, sub.w.values, n)
        s = sub.sort_values(["cand", "date"])
        last = s.groupby("cand").tail(1).set_index("cand")
        F[f"{pre}_last_de"] = last.de.reindex(F.index).values
        F[f"{pre}_last_age"] = last.age.reindex(F.index).values

    for s in ALL_STREAMS:
        block(st[st.stream == s], s)
    block(st[st.stream.isin(CORE)], "core")
    block(st, "glob")

    # The address row with no end date is where the person lives *now*. That single
    # record predicts the label better than the whole address feed does (rho +0.39 vs
    # +0.29), so it gets its own block rather than being averaged in with the history.
    op = st[(st.stream == "adr") & (st.open_ended == True)]
    F["adr_open_de"] = _share(op, n, 150)
    F["adr_open_n"] = _bc(op.cand.values, op.w.values, n)
    op_last = op.sort_values(["cand", "date"]).groupby("cand").tail(1).set_index("cand")
    F["adr_open_last_de"] = op_last.de.reindex(F.index).values
    F["adr_open_age"] = op_last.age.reindex(F.index).values

    # Record categories that carry two rows per candidate rather than one: a less noisy
    # read of the same feed.
    for stream, kind, name in (("ttl", "record_update", "ttl_recupd"),
                               ("lic", "credential_update", "lic_credupd"),
                               ("ext", "reference_update", "ext_refupd")):
        F[f"{name}_de"] = _share(st[(st.stream == stream) & (st.kind == kind)], n, 150)

    F["core_last_mean"] = F[[f"{s}_last_de" for s in CORE]].mean(axis=1)
    F["core_last_std"] = F[[f"{s}_last_de" for s in CORE]].std(axis=1)
    F["core_sh150_std"] = F[[f"{s}_sh150" for s in CORE]].std(axis=1)

    # evidence sufficiency / quality (direction-free)
    idx, w = e.cand.values, e.w.values
    F["ev_n"] = _bc(idx, w, n)
    denom = np.maximum(F["ev_n"].values, 1e-9)
    F["ev_nullshare"] = _bc(idx, w * ~e.has_state.values, n) / denom
    F["ev_limited"] = _bc(idx, w * e["quality"].eq("limited").fillna(False).values, n) / denom
    F["ev_badstatus"] = _bc(
        idx, w * e["status"].isin(["superseded", "expired"]).fillna(False).values, n) / denom
    for win in (180, 365):
        m = e.age.values <= win
        F[f"ev_n{win}"] = _bc(idx[m], w[m], n)
    F["ev_linkconf"] = pd.Series(e.groupby("cand").w.max()).reindex(F.index).values
    F["ev_nstates"] = pd.Series(st.groupby("cand").state.nunique()).reindex(F.index).fillna(0).values
    F["ev_minage"] = pd.Series(st.groupby("cand").age.min()).reindex(F.index).values

    # T1 update stream.  Encoded as signed deviations so that every T1 feature is
    # exactly 0 in the T0 phase (and when a candidate has no update of that kind):
    # the block then contributes nothing at T0 instead of relying on imputation.
    t1 = st[st.is_t1]
    F["t1_n"] = _bc(t1.cand.values, t1.w.values, n) if (phase == "T1" and len(t1)) else 0.0
    if phase == "T1" and len(t1):
        F["t1_de"] = np.nan_to_num(_share(t1, n) - 0.5)
        last = t1.sort_values(["cand", "date"]).groupby("cand").tail(1).set_index("cand")
        F["t1_last_de"] = np.nan_to_num(last.de.reindex(F.index).values - 0.5)
        for s in ("ttl", "adr"):
            F[f"t1_{s}_de"] = np.nan_to_num(_share(t1[t1.stream == s], n) - 0.5)
        base = _share(st[(~st.is_t1) & st.stream.isin(CORE)], n, 365)
        F["t1_delta"] = np.nan_to_num(np.nan_to_num(_share(t1, n), nan=0.5) - base)
    else:
        for c in ("t1_de", "t1_last_de", "t1_ttl_de", "t1_adr_de", "t1_delta"):
            F[c] = 0.0

    F["phase_t1"] = 1.0 if phase == "T1" else 0.0
    return F


FEATURES = (
    ["obs_state_oos"]
    + [f"{s}_{k}" for s in ALL_STREAMS + ["core", "glob"]
       for k in ["sh90", "sh150", "sh365", "shflat", "n", "last_de", "last_age"]]
    + ["adr_open_de", "adr_open_n", "adr_open_last_de", "adr_open_age",
       "ttl_recupd_de", "lic_credupd_de", "ext_refupd_de",
       "core_last_mean", "core_last_std", "core_sh150_std",
       "ev_n", "ev_nullshare", "ev_limited", "ev_badstatus", "ev_n180", "ev_n365",
       "ev_linkconf", "ev_nstates", "ev_minage",
       "t1_n", "t1_de", "t1_last_de", "t1_ttl_de", "t1_adr_de", "t1_delta", "phase_t1"]
)
