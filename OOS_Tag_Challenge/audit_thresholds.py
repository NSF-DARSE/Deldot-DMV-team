"""Accuracy is scored, but so is calibration. What does tuning the decision rule
purely for accuracy actually buy, and what does it cost?"""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, f1_score, log_loss
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
print(f"argmax (what we ship):        acc={accuracy_score(y,oof.argmax(1)):.4f} "
      f"f1={f1_score(y,oof.argmax(1),average='macro'):.4f} ll={log_loss(y,oof,labels=[0,1,2]):.4f}")
# tune class-specific multipliers on the probabilities, chosen ON the data (optimistic)
best=(0,None)
for a in np.linspace(0.6,1.8,25):
    for b in np.linspace(0.6,1.8,25):
        pred=(oof*np.array([a,b,1.0])).argmax(1)
        s=accuracy_score(y,pred)
        if s>best[0]: best=(s,(a,b,pred))
s,(a,b,pred)=best
print(f"weights tuned FOR accuracy:   acc={s:.4f} "
      f"f1={f1_score(y,pred,average='macro'):.4f}  (weights {a:.2f},{b:.2f},1.00)")
print(f"  -> buys {s-accuracy_score(y,oof.argmax(1)):+.4f} accuracy, and the weights were")
print(f"     chosen on the same 300 cases, so the real gain is smaller still.")
# what if we simply never predict the middle class?
two=np.where(oof[:,2]>=oof[:,0],2,0)
print(f"\nif we never say 'insufficient': acc={accuracy_score(y,two):.4f} "
      f"(forced into two classes; every truly-unclear case is now wrong)")
