"""Timeboxed push for more accuracy. Each idea is judged on the SAME grouped CV
protocol as the shipped model; anything that does not clearly beat it is dropped."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
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
F0=build_features(ev,cand,"T0",t1ref); F1=build_features(ev,cand,"T1",t1ref)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
grp=np.concatenate([li,li])

def cv(X, nrep=20, C=0.03, extra_train=None):
    oof=np.zeros((len(y),3))
    for rep in range(nrep):
        rs=np.random.RandomState(100+rep); u=np.unique(grp)
        gm=dict(zip(u,rs.permutation(len(u)))); gg=np.array([gm[g] for g in grp])
        for tr,te in GroupKFold(n_splits=5).split(X,y,gg):
            imp=SimpleImputer(strategy='median'); sc=StandardScaler()
            A=sc.fit_transform(imp.fit_transform(X.iloc[tr])); yy=y[tr]
            sw=None
            if extra_train is not None:
                Xu,yu,wu=extra_train
                A=np.vstack([A,sc.transform(imp.transform(Xu))]); yy=np.concatenate([yy,yu])
                sw=np.concatenate([np.ones(len(tr)),wu])
            B=sc.transform(imp.transform(X.iloc[te]))
            oof[te]+=OrdinalLogit(C=C).fit(A,yy,sample_weight=sw).predict_proba(B)
    oof/=nrep
    return (log_loss(y,oof,labels=[0,1,2]), accuracy_score(y,oof.argmax(1)),
            np.mean([roc_auc_score((y==k).astype(int),oof[:,k]) for k in range(3)]),
            roc_auc_score((y==2).astype(int),priority(oof)))

def show(tag,r,base=None):
    d = f"   (Δll {r[0]-base[0]:+.4f})" if base else ""
    print(f"  {tag:38s} ll={r[0]:.4f} acc={r[1]:.4f} auc={r[2]:.4f} prio={r[3]:.4f}{d}")
    return r

def mat(cols, add0=None, add1=None):
    a=F0.iloc[li].copy(); b=F1.iloc[li].copy()
    if add0 is not None: a=pd.concat([a,add0.iloc[li].reset_index(drop=True).set_index(a.index)],axis=1)
    if add1 is not None: b=pd.concat([b,add1.iloc[li].reset_index(drop=True).set_index(b.index)],axis=1)
    return pd.concat([a,b],ignore_index=True)[cols]

print("BASELINE (shipped model)")
base=show("shipped 34 features", cv(mat(COLS)))

# ---- A. per-feed short and long half-lives -------------------------------
A_COLS=COLS+[f"{s}_{k}" for s in CORE for k in ("sh90","sh365")]
show("A. + per-feed sh90 & sh365", cv(mat(A_COLS)), base)

# ---- B. T1 record_action semantics ---------------------------------------
def t1_action(phase):
    G=pd.DataFrame(index=range(n))
    if phase=="T0":
        for c in ("t1_corr_n","t1_corr_de","t1_new_de"): G[c]=0.0
        return G
    t=ev[ev.is_t1&ev.state.notna()].copy(); t["de"]=(t.state=="DE").astype(float)
    for tag,kinds in [("corr",["record_correction"]),("new",["new_record","status_update"])]:
        s=t[t.kind.isin(kinds)]
        cnt=np.bincount(s.cand,weights=s.w,minlength=n)
        de=np.bincount(s.cand,weights=s.w*s.de,minlength=n)
        if tag=="corr": G["t1_corr_n"]=cnt
        G[f"t1_{tag}_de"]=np.where(cnt>1e-9, de/np.maximum(cnt,1e-9)-0.5, 0.0)
    return G
B0,B1=t1_action("T0"),t1_action("T1")
B_COLS=COLS+["t1_corr_n","t1_corr_de","t1_new_de"]
show("B. + T1 record_action split", cv(mat(B_COLS,B0,B1)), base)

# ---- C. nonlinearity in the direction index -------------------------------
def nl(F):
    G=pd.DataFrame(index=range(n))
    x=F.core_sh150.fillna(0.5)
    G["core_sh150_sq"]=(x-0.5)**2
    G["core_sh150_abs"]=(x-0.5).abs()
    return G
C_COLS=COLS+["core_sh150_sq","core_sh150_abs"]
show("C. + |dev| and squared direction", cv(mat(C_COLS,nl(F0),nl(F1))), base)

# ---- E. did the T1 title update change the state? -------------------------
def chg(phase):
    G=pd.DataFrame(index=range(n))
    if phase=="T0":
        G["t1_ttl_changed"]=0.0; return G
    t0=ev[(~ev.is_t1)&(ev.stream=="ttl")&ev.state.notna()].sort_values("date").groupby("cand").tail(1).set_index("cand")
    t1=ev[ev.is_t1&(ev.stream=="ttl")&ev.state.notna()].sort_values("date").groupby("cand").tail(1).set_index("cand")
    a=t0.state.reindex(range(n)); b=t1.state.reindex(range(n))
    G["t1_ttl_changed"]=np.where(b.isna()|a.isna(),0.0,np.where(a.values!=b.values,1.0,-1.0))
    return G
E_COLS=COLS+["t1_ttl_changed"]
show("E. + T1 title state changed flag", cv(mat(E_COLS,chg("T0"),chg("T1"))), base)

# ---- F. tag x direction interaction ---------------------------------------
def inter(F):
    G=pd.DataFrame(index=range(n))
    G["tag_x_dir"]=(F.obs_state_oos.values)*(F.core_sh150.fillna(0.5).values-0.5)
    return G
F_COLS=COLS+["tag_x_dir"]
show("F. + tag x direction interaction", cv(mat(F_COLS,inter(F0),inter(F1))), base)
