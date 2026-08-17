"""Feature ideas surfaced by the process audit: the current-address flag, an
explicit old-versus-new contrast, and the record categories that hold two rows
per candidate. Kept separate so both audit scripts can import without side effects."""
import numpy as np
import pandas as pd

from featurize import CORE

OPEN = ["adr_open_de", "adr_open_n", "adr_open_last_de", "adr_open_age"]
TREND = [f"{s}_{k}" for s in CORE for k in ("first_de", "trend")] + ["core_trend"]
CATEGORY = ["ttl_recupd_de", "lic_credupd_de", "ext_refupd_de"]


def make_extra(ev, cand, phase, t1_ref):
    n = len(cand)
    """The untested ideas, built the same way featurize.py builds everything."""
    e = ev[~ev.is_t1].copy() if phase=="T0" else ev.copy()
    ref = (pd.to_datetime(cand.candidate_observed_date).values if phase=="T0"
           else np.repeat(np.datetime64(t1_ref), n))
    e["age"]=((ref[e.cand.values]-e.date.values)/np.timedelta64(1,"D")).clip(min=0)
    e=e[e.state.notna()].copy(); e["de"]=(e.state=="DE").astype(float)
    G=pd.DataFrame(index=range(n))
    def sh(sub,H=150):
        if not len(sub): return np.full(n,np.nan)
        w=sub.w.values*0.5**(sub.age.values/H)
        num=np.bincount(sub.cand.values,weights=w*sub.de.values,minlength=n)
        den=np.bincount(sub.cand.values,weights=w,minlength=n)
        return np.where(den>1e-9,num/np.maximum(den,1e-9),np.nan)
    # 1. current address = the one with no end date
    op=e[(e.stream=="adr")&(e.open_ended==True)]
    G["adr_open_de"]=sh(op)
    G["adr_open_n"]=np.bincount(op.cand.values,weights=op.w.values,minlength=n)
    last=op.sort_values(["cand","date"]).groupby("cand").tail(1).set_index("cand")
    G["adr_open_last_de"]=last.de.reindex(range(n)).values
    G["adr_open_age"]=last.age.reindex(range(n)).values
    # 2. explicit old-versus-new contrast per feed
    for s in CORE:
        sub=e[e.stream==s].sort_values(["cand","date"])
        first=sub.groupby("cand").head(1).set_index("cand").de.reindex(range(n))
        lastd=sub.groupby("cand").tail(1).set_index("cand").de.reindex(range(n))
        G[f"{s}_first_de"]=first.values
        G[f"{s}_trend"]=(lastd-first).values
    G["core_trend"]=G[[f"{s}_trend" for s in CORE]].mean(axis=1)
    # 3. the record categories that hold two rows per candidate
    for s,k,name in [("ttl","record_update","ttl_recupd"),("lic","credential_update","lic_credupd"),
                     ("ext","reference_update","ext_refupd")]:
        G[f"{name}_de"]=sh(e[(e.stream==s)&(e.kind==k)])
    return G

