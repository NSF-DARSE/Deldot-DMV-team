"""AUDIT: candidate features the shipped model never had. Judged on the same
grouped CV protocol, so results are comparable to predict.py."""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from model import OrdinalLogit, CLASS_TO_ORD
from featurize import build_features, CORE
from pipeline import load, BASE, build_evidence
from predict import COLS, priority

D=load(); cand=D['cand'].reset_index(drop=True); n=len(cand)
ev=build_evidence(D,verbose=False); t1ref=pd.to_datetime(D['up'].observed_date).max()
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
grp=np.concatenate([li,li])

def extra(phase):
    """The untested ideas, built the same way featurize.py builds everything."""
    e = ev[~ev.is_t1].copy() if phase=="T0" else ev.copy()
    ref = (pd.to_datetime(cand.candidate_observed_date).values if phase=="T0"
           else np.repeat(np.datetime64(t1ref), n))
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
X0=pd.concat([build_features(ev,cand,"T0",t1ref),extra("T0")],axis=1)
X1=pd.concat([build_features(ev,cand,"T1",t1ref),extra("T1")],axis=1)
FULL=pd.concat([X0.iloc[li],X1.iloc[li]],ignore_index=True)

def cv(cols,C=0.03,nrep=20):
    X=FULL[cols]; oof=np.zeros((len(y),3))
    for rep in range(nrep):
        rs=np.random.RandomState(100+rep); u=np.unique(grp)
        gm=dict(zip(u,rs.permutation(len(u)))); gg=np.array([gm[g] for g in grp])
        for tr,te in GroupKFold(n_splits=5).split(X,y,gg):
            imp=SimpleImputer(strategy="median"); sc=StandardScaler()
            A=sc.fit_transform(imp.fit_transform(X.iloc[tr]))
            oof[te]+=OrdinalLogit(C=C).fit(A,y[tr]).predict_proba(sc.transform(imp.transform(X.iloc[te])))
    oof/=nrep
    return (log_loss(y,oof,labels=[0,1,2]), accuracy_score(y,oof.argmax(1)),
            np.mean([roc_auc_score((y==k).astype(int),oof[:,k]) for k in range(3)]),
            roc_auc_score((y==2).astype(int),priority(oof)))
CUR=[f"{s}_{k}" for s in CORE for k in ("first_de","trend")]+["core_trend"]
OPEN=["adr_open_de","adr_open_n","adr_open_last_de","adr_open_age"]
CAT=["ttl_recupd_de","lic_credupd_de","ext_refupd_de"]
b=cv(COLS); print(f"  {'shipped model':38s} ll={b[0]:.4f} acc={b[1]:.4f} auc={b[2]:.4f} prio={b[3]:.4f}")
for tag,add in [("+ current address (open-ended)",OPEN),("+ old-vs-new trend",CUR),
                ("+ two-row record categories",CAT),
                ("+ current address & trend",OPEN+CUR),("+ all three",OPEN+CUR+CAT)]:
    r=cv(COLS+add)
    print(f"  {tag:38s} ll={r[0]:.4f} acc={r[1]:.4f} auc={r[2]:.4f} prio={r[3]:.4f}   Δll {r[0]-b[0]:+.4f}")

print("\nBest combinations, with a regularisation sweep:")
for tag,add in [("current address only",OPEN),("current address + categories",OPEN+CAT),("all three",OPEN+CUR+CAT)]:
    for C in (0.02,0.03,0.05,0.08):
        r=cv(COLS+add,C=C)
        print(f"  {tag:30s} C={C:<5} ll={r[0]:.4f} acc={r[1]:.4f} auc={r[2]:.4f} prio={r[3]:.4f}")
