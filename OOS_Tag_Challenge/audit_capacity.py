"""The operational question a judge actually cares about: if staff can only work N
cases, how many of the ones that warranted review do they reach?"""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from predict import COLS, cross_validate, priority
from featurize import build_features
from model import CLASS_TO_ORD
from pipeline import load, BASE, build_evidence
D=load(); cand=D['cand'].reset_index(drop=True)
ev=build_evidence(D,verbose=False); t1=pd.to_datetime(D['up'].observed_date).max()
F0=build_features(ev,cand,"T0",t1); F1=build_features(ev,cand,"T1",t1)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
X=pd.concat([F0.iloc[li],F1.iloc[li]],ignore_index=True)[COLS]
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
grp=np.concatenate([li,li])
oof=cross_validate(X,y,grp)
# T0 only: the realistic "first pass over the backlog" scenario
m=np.arange(300); P=oof[:300]; yy=y[:300]
pr=priority(P); order=np.argsort(-pr)
warranted=(yy==2).astype(int); total=warranted.sum()
print(f"held-out cases: {len(yy)}   of which review_warranted: {total} ({total/len(yy):.1%})\n")
print(f"{'staff reviews':>14} {'warranted found':>16} {'vs random':>11} {'lift':>7} {'precision':>10}")
for frac in (0.05,0.10,0.20,0.30,0.40,0.50):
    k=int(round(frac*len(yy))); got=warranted[order[:k]].sum()
    rand=total*frac
    print(f"{f'top {frac:.0%} ({k})':>14} {f'{got} of {total}':>16} {rand:11.1f} {got/rand:6.2f}x {got/k:10.1%}")
half=np.searchsorted(np.cumsum(warranted[order]), total*0.5)+1
print(f"\n  half of all warranted cases are found in the top {half} of {len(yy)} "
      f"({half/len(yy):.0%}) of the queue")
eighty=np.searchsorted(np.cumsum(warranted[order]), total*0.8)+1
print(f"  80% of them in the top {eighty} ({eighty/len(yy):.0%})")
