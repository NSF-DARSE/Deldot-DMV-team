"""AUDIT: an unbiased number for the whole procedure, not just for its winner.

Nested selection. The choice of feature blocks and regularisation strength is made
inside a training split; the chosen model is then scored once on candidates it never
saw. The gap between this and the ordinary cross-validated figure is the selection
bias built up by comparing many options against the same 300 labels.
"""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from model import OrdinalLogit, CLASS_TO_ORD
from featurize import build_features
from pipeline import load, BASE, build_evidence
from predict import COLS, CURRENT_ADDR_F, CATEGORY_F, priority

D=load(); cand=D['cand'].reset_index(drop=True)
ev=build_evidence(D,verbose=False); t1ref=pd.to_datetime(D['up'].observed_date).max()
F0=build_features(ev,cand,"T0",t1ref); F1=build_features(ev,cand,"T1",t1ref)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
grp=np.concatenate([li,li])
FULL=pd.concat([F0.iloc[li],F1.iloc[li]],ignore_index=True)

BASE_F=[c for c in COLS if c not in CURRENT_ADDR_F+CATEGORY_F]
MENU={"base":BASE_F,"base+address":BASE_F+CURRENT_ADDR_F,
      "base+categories":BASE_F+CATEGORY_F,"full":COLS}
CS=(0.02,0.03,0.05)

def fp(cols,C,tr,te):
    imp=SimpleImputer(strategy="median"); sc=StandardScaler()
    A=sc.fit_transform(imp.fit_transform(FULL[cols].iloc[tr]))
    return OrdinalLogit(C=C).fit(A,y[tr]).predict_proba(sc.transform(imp.transform(FULL[cols].iloc[te])))

def report(tag,oof):
    print(f"   {tag:26s} ll={log_loss(y,oof,labels=[0,1,2]):.4f} "
          f"acc={accuracy_score(y,oof.argmax(1)):.4f} "
          f"auc={np.mean([roc_auc_score((y==k).astype(int),oof[:,k]) for k in range(3)]):.4f} "
          f"prio={roc_auc_score((y==2).astype(int),priority(oof)):.4f}")

print("Ordinary cross-validation (what predict.py reports):")
o=np.zeros((len(y),3))
for rep in range(20):
    rs=np.random.RandomState(100+rep); u=np.unique(grp)
    gm=dict(zip(u,rs.permutation(len(u)))); gg=np.array([gm[g] for g in grp])
    for tr,te in GroupKFold(n_splits=5).split(FULL,y,gg): o[te]+=fp(COLS,0.02,tr,te)
report("chosen model",o/20)

print("\nNested selection (unbiased for the whole procedure):")
rows=[]
for seed in range(6):
    rs=np.random.RandomState(seed); u=np.unique(grp)
    gm=dict(zip(u,rs.permutation(len(u)))); gg=np.array([gm[g] for g in grp])
    oof=np.zeros((len(y),3)); picks=[]
    for tr,te in GroupKFold(n_splits=5).split(FULL,y,gg):
        best=(9e9,None); gtr=gg[tr]
        for name,cols in MENU.items():
            for C in CS:
                inner=np.zeros((len(tr),3))
                for itr,ite in GroupKFold(n_splits=4).split(FULL.iloc[tr],y[tr],gtr):
                    inner[ite]=fp(cols,C,tr[itr],tr[ite])
                sc_=log_loss(y[tr],inner,labels=[0,1,2])
                if sc_<best[0]: best=(sc_,(name,cols,C))
        _,(name,cols,C)=best; picks.append(f"{name}/{C}")
        oof[te]=fp(cols,C,tr,te)
    rows.append([log_loss(y,oof,labels=[0,1,2]),accuracy_score(y,oof.argmax(1)),
                 np.mean([roc_auc_score((y==k).astype(int),oof[:,k]) for k in range(3)]),
                 roc_auc_score((y==2).astype(int),priority(oof))])
    print(f"   seed {seed}: ll={rows[-1][0]:.4f} acc={rows[-1][1]:.4f} auc={rows[-1][2]:.4f}  picks: {picks}")
r=np.array(rows).mean(0)
print(f"\n   NESTED MEAN  ll={r[0]:.4f} acc={r[1]:.4f} auc={r[2]:.4f} prio={r[3]:.4f}")
