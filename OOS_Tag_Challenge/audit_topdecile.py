"""Exactly what is in the top of the queue: how do those cases' TRUE labels break down?"""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from featurize import build_features
from model import CLASS_TO_ORD
from pipeline import load, BASE, build_evidence
from predict import COLS, cross_validate, priority
D=load(); cand=D['cand'].reset_index(drop=True)
ev=build_evidence(D,verbose=False); t1=pd.to_datetime(D['up'].observed_date).max()
F0=build_features(ev,cand,"T0",t1); F1=build_features(ev,cand,"T1",t1)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
X=pd.concat([F0.iloc[li],F1.iloc[li]],ignore_index=True)[COLS]
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
oof=cross_validate(X,y,np.concatenate([li,li]))
# T0 only = the realistic first pass over a backlog
P,yy = oof[:300], y[:300]
order=np.argsort(-priority(P))
NAMES=["not worth reviewing","cannot tell","worth reviewing"]
print(f"All 300 held-out cases, true labels: "
      f"{ {NAMES[k]: int((yy==k).sum()) for k in range(3)} }\n")
print(f"{'slice of queue':>16} {'cases':>6} | {'worth reviewing':>16} {'cannot tell':>12} {'not worth it':>13}")
for frac in (0.05,0.10,0.20,0.50,1.00):
    k=int(round(frac*len(yy))); sel=yy[order[:k]]
    c=[int((sel==j).sum()) for j in range(3)]
    print(f"{f'top {frac:.0%}':>16} {k:>6} | {c[2]:>7} ({c[2]/k:4.0%}) {c[1]:>6} ({c[1]/k:4.0%}) {c[0]:>6} ({c[0]/k:4.0%})")
k=30; sel=yy[order[:k]]
w=int((sel==2).sum()); u=int((sel==1).sum()); n=int((sel==0).sum())
print(f"\nTOP 10% ({k} cases) in detail:")
print(f"  truly worth reviewing : {w}")
print(f"  truly cannot tell     : {u}   <- still needs a human, just not a clear-cut one")
print(f"  truly not worth it    : {n}   <- these are the wasted opens")
print(f"\n  'worth reviewing' only              : {w}/{k} = {w/k:.0%}")
print(f"  'worth reviewing' OR 'cannot tell'  : {(w+u)}/{k} = {(w+u)/k:.0%}  (both need a person)")
print(f"  genuinely wasted                    : {n}/{k} = {n/k:.0%}")
# honest uncertainty on a 30-case slice
from scipy.stats import beta
lo,hi = beta.ppf([.025,.975], w+0.5, k-w+0.5)
print(f"\n  CAVEAT: this is measured on only {k} cases. 95% interval on the {w/k:.0%} figure")
print(f"  is roughly {lo:.0%} to {hi:.0%}. The direction is solid; the exact number is not.")
