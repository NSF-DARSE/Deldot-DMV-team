"""How much is another label worth? Train on subsets of the 300 development
candidates and watch the held-out score move. Answers whether asking the organisers
for a larger label set is worth doing, or whether we are limited by the data itself."""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from model import OrdinalLogit, CLASS_TO_ORD
from featurize import build_features
from pipeline import load, BASE, build_evidence
from predict import COLS, C_REG, priority

D=load(); cand=D['cand'].reset_index(drop=True)
ev=build_evidence(D,verbose=False); t1=pd.to_datetime(D['up'].observed_date).max()
F0=build_features(ev,cand,"T0",t1); F1=build_features(ev,cand,"T1",t1)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
X=pd.concat([F0.iloc[li],F1.iloc[li]],ignore_index=True)[COLS]
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
cid=np.concatenate([np.arange(300),np.arange(300)])     # candidate index within the dev set

print(f"{'train candidates':>17} {'log loss':>9} {'accuracy':>9} {'macro AUC':>10}")
res={}
HOLDOUT=75
for ntrain in (40, 70, 105, 145, 190, 225):
    ll=[];ac=[];au=[]
    for rep in range(40):
        rs=np.random.RandomState(rep)
        order=rs.permutation(300)
        te_c=order[:HOLDOUT]                       # same-sized held-out set every time
        tr_c=order[HOLDOUT:HOLDOUT+ntrain]
        tr=np.nonzero(np.isin(cid,tr_c))[0]; te=np.nonzero(np.isin(cid,te_c))[0]
        imp=SimpleImputer(strategy="median"); sc=StandardScaler()
        A=sc.fit_transform(imp.fit_transform(X.iloc[tr]))
        P=OrdinalLogit(C=C_REG).fit(A,y[tr]).predict_proba(sc.transform(imp.transform(X.iloc[te])))
        ll.append(log_loss(y[te],P,labels=[0,1,2])); ac.append(accuracy_score(y[te],P.argmax(1)))
        au.append(np.mean([roc_auc_score((y[te]==k).astype(int),P[:,k]) for k in range(3)]))
    res[ntrain]=(np.mean(ll),np.mean(ac),np.mean(au))
    print(f"{ntrain:>17} {np.mean(ll):9.4f} {np.mean(ac):9.4f} {np.mean(au):10.4f}")
ks=sorted(res)
print("\n  marginal value of labels (log loss gained per extra 50 candidates):")
for a,b in zip(ks,ks[1:]):
    per50=(res[a][0]-res[b][0])/(b-a)*50
    print(f"    {a:3d} -> {b:3d}:  {per50:+.4f}")
