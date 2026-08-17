"""AUDIT: what does alias recovery buy, measured the same way as everything else?"""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from pipeline import load, BASE
from resolve import Resolver, link_by_name, link_titles_by_vehicle
from alias import learn_aliases, link_by_alias
from model import CLASS_TO_ORD
D=load(); cand=D['cand'].reset_index(drop=True); n=len(cand)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id]); y0=lab.label_t0.map(CLASS_TO_ORD).values
ref=pd.to_datetime(cand.candidate_observed_date).values
res=Resolver(cand)
amap=learn_aliases(res, D['ttl'])
print(f"aliases learned from the title feed: {len(amap)}  (covering {len(set(amap.values()))} distinct candidates)")

SPEC={"adr":("first_name","last_name","state","effective_start_date"),
      "lic":("first_name","last_name","credential_state","event_date"),
      "ext":("first_name","last_name","signal_state","effective_date"),
      "wrk":("first_name","last_name","work_state","observed_date")}
def rho(v):
    v=np.asarray(v,float)[li]; m=~np.isnan(v); return spearmanr(v[m],y0[m]).statistic
def feed_score(df,ci,wt,ri,scol,dcol):
    sub=df.iloc[ri]
    d=pd.DataFrame({"cand":ci,"w":wt,"state":sub[scol].values,
                    "date":pd.to_datetime(sub[dcol]).values})
    d=d[d.state.notna()]
    age=((ref[d.cand.values]-d.date.values)/np.timedelta64(1,"D")).clip(min=0)
    w=d.w*0.5**(age/150); de=(d.state=="DE").astype(float)
    num=np.bincount(d.cand,weights=w*de,minlength=n); den=np.bincount(d.cand,weights=w,minlength=n)
    return np.where(den>1e-9,num/np.maximum(den,1e-9),np.nan)
print(f"\n{'feed':5s} {'rows now':>9s} {'rows +alias':>12s} {'rho now':>9s} {'rho +alias':>11s}")
pooled_a={}; pooled_b={}
for k,(fc,lc,scol,dcol) in SPEC.items():
    df=D[k]
    ri,ci,wt=link_by_name(res,df,fc,lc)
    linked=np.zeros(len(df),bool); linked[ri]=True
    ri2,ci2,wt2=link_by_alias(amap,df,fc,lc,linked)
    A=feed_score(df,ci,wt,ri,scol,dcol)
    RI=np.concatenate([ri,ri2]); CI=np.concatenate([ci,ci2]); WT=np.concatenate([wt,wt2])
    B=feed_score(df,CI,WT,RI,scol,dcol)
    pooled_a[k]=(ri,ci,wt); pooled_b[k]=(RI,CI,WT)
    print(f"{k:5s} {len(set(ri)):9d} {len(set(RI)):12d} {rho(A):+9.3f} {rho(B):+11.3f}")
