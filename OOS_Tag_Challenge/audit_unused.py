"""AUDIT: do the record-type / feed-name / current-address fields carry signal?
None of them was used by the shipped model. This asks whether that was a mistake."""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from pipeline import load, BASE, build_evidence
from model import CLASS_TO_ORD
D=load(); cand=D['cand'].reset_index(drop=True); n=len(cand)
ev=build_evidence(D,verbose=False)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
y0=lab.label_t0.map(CLASS_TO_ORD).values
ref=pd.to_datetime(cand.candidate_observed_date).values
e=ev[(~ev.is_t1)&ev.state.notna()].copy()
e["age"]=((ref[e.cand.values]-e.date.values)/np.timedelta64(1,"D")).clip(min=0)
e["de"]=(e.state=="DE").astype(float)
def rho(v):
    v=np.asarray(v,float)[li]; m=~np.isnan(v)
    return (spearmanr(v[m],y0[m]).statistic, int(m.sum()))
def score(sub,H=150):
    w=sub.w*0.5**(sub.age/H)
    num=np.bincount(sub.cand,weights=w*sub.de,minlength=n); den=np.bincount(sub.cand,weights=w,minlength=n)
    return np.where(den>1e-9,num/np.maximum(den,1e-9),np.nan)
CORE=["adr","lic","ttl","ext"]
r,c = rho(score(e[e.stream.isin(CORE)]))
print(f"REFERENCE  pooled core feeds: rho={r:+.3f}\n")
print("Signal carried by each record category on its own (rho, and how many candidates it covers):")
for stream,field in [("adr","kind"),("lic","kind"),("ttl","kind"),("ext","kind"),("ext","desc"),("wrk","kind")]:
    sub=e[e.stream==stream]
    if sub[field].nunique()<2: print(f"  {stream}.{field}: constant, nothing to test"); continue
    print(f"  {stream}.{field}")
    for k,g in sub.groupby(field):
        rr,cc=rho(score(g))
        print(f"      {str(k):26s} rho={rr:+.3f}  n={cc:4d}  rows={len(g):6d}")
print("\nCurrent address (effective_end_date is blank) vs any address:")
a=e[e.stream=="adr"]
for tag,g in [("open-ended (current) only",a[a.open_ended==True]),("all address rows",a)]:
    rr,cc=rho(score(g)); print(f"  {tag:28s} rho={rr:+.3f}  n={cc}")

print("\n" + "="*78)
print("Are the record types just a proxy for age? (mean age in days, and position in sequence)")
for stream in ["ttl","lic","ext","adr"]:
    sub=e[e.stream==stream].sort_values(["cand","date"]).copy()
    sub["rank_from_newest"]=sub.groupby("cand").cumcount(ascending=False)
    t=sub.groupby("kind").agg(mean_age=("age","mean"), mean_rank=("rank_from_newest","mean"),
                              de_share=("de","mean"), rows=("de","size"))
    print(f"  {stream}:"); print(t.round(2).to_string().replace("\n","\n    "))
