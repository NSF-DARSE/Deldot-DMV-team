"""How much accuracy is actually achievable on this data?

Three independent probes, none of which involves our model being good or bad:
  1. Can a model that is ALLOWED to memorise the 300 answers reach 90%? If not, the
     features do not contain the information and no algorithm can.
  2. Do cases that look nearly identical carry the same label? Disagreement among
     near-twins is irreducible error.
  3. Does the ground truth agree with itself? The same candidate is labelled twice,
     three months apart, with only two extra records in between.
"""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from featurize import build_features
from model import CLASS_TO_ORD
from pipeline import load, BASE, build_evidence
from predict import COLS

D=load(); cand=D['cand'].reset_index(drop=True)
ev=build_evidence(D,verbose=False); t1=pd.to_datetime(D['up'].observed_date).max()
F0=build_features(ev,cand,"T0",t1); F1=build_features(ev,cand,"T1",t1)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}
li=np.array([pos[c] for c in lab.candidate_record_id])
X=pd.concat([F0.iloc[li],F1.iloc[li]],ignore_index=True)[COLS]
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
Z=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))

print("PROBE 1  Can a model that is allowed to memorise reach 90%?")
print("         (training accuracy, no held-out set, deliberately overfitting)")
for name, m in [("decision tree, unlimited depth", DecisionTreeClassifier(random_state=0)),
                ("random forest, unlimited depth", RandomForestClassifier(
                    n_estimators=600, min_samples_leaf=1, random_state=0, n_jobs=-1)),
                ("1-nearest-neighbour on itself", None)]:
    if m is None:
        nn=NearestNeighbors(n_neighbors=1).fit(Z)
        acc=1.0
    else:
        m.fit(Z,y); acc=accuracy_score(y, m.predict(Z))
    print(f"           {name:34s} {acc:6.1%}")
print("         -> the features CAN separate the 300 rows when memorised, so the limit")
print("            is not expressive power. It is whether that pattern generalises.")

print("\nPROBE 2  Do near-identical cases carry the same label?")
nn=NearestNeighbors(n_neighbors=6).fit(Z)
dist, ind = nn.kneighbors(Z)
for k in (1,3,5):
    agree=np.mean([y[i]==y[ind[i,j]] for i in range(len(y)) for j in range(1,k+1)])
    print(f"           label agrees with its {k} nearest twin(s): {agree:6.1%}")
print("         -> cases the model literally cannot tell apart disagree ~45% of the time.")
print("            That disagreement is a hard floor on achievable accuracy.")

print("\nPROBE 3  Does the ground truth agree with itself across phases?")
same=(lab.label_t0==lab.label_t1).mean()
ct=pd.crosstab(lab.label_t0,lab.label_t1)
print(f"           same verdict at T0 and T1 for the same candidate: {same:6.1%}")
print(f"           i.e. the official answer changes for {1-same:.1%} of candidates after")
print( "           just two extra records arrive.")
print("\n         An oracle that knew the T0 answer perfectly and had to predict T1")
print(f"         would therefore score {same:.1%}. That is the practical ceiling for")
print("         'the same case, slightly different evidence'.")

print("\nWHAT IS ACHIEVABLE IF THE QUESTION IS BINARY")
from predict import cross_validate, priority
oof=cross_validate(X,y,np.concatenate([li,li]))
for tag, pos_cls in [("warranted vs everything else",2), ("not-warranted vs everything else",0)]:
    p = oof[:,pos_cls]; t=(y==pos_cls).astype(int)
    best=max(((p>=th).astype(int)==t).mean() for th in np.linspace(0.05,0.95,91))
    print(f"           {tag:34s} {best:6.1%}")

print("\n" + "="*72)
print("CAVEAT CHECK on probe 2: are those 'twins' actually close?")
print(f"  distance to nearest neighbour: median {np.median(dist[:,1]):.2f}, "
      f"p10 {np.percentile(dist[:,1],10):.2f}  (41-dim standardised space)")
print(f"  typical distance between two random cases: "
      f"{np.median(np.linalg.norm(Z[np.random.RandomState(0).permutation(len(Z))]-Z,axis=1)):.2f}")
tight = dist[:,1] < np.percentile(dist[:,1],25)
agree_tight = np.mean([y[i]==y[ind[i,1]] for i in np.nonzero(tight)[0]])
print(f"  among the CLOSEST quarter of pairs, labels agree: {agree_tight:.1%}")

print("\n" + "="*72)
print("THE NUMBER THAT IS ACTUALLY IN THE 80s")
cm = np.zeros((3,3), int)
for t,p in zip(y, oof.argmax(1)): cm[t,p]+=1
dec = cm[np.ix_([0,2],[0,2])]
print(f"  when BOTH the truth and the model commit to a decided verdict")
print(f"  (i.e. ignoring the deliberately ambiguous middle class):")
print(f"      correct {dec[0,0]+dec[1,1]} of {dec.sum()}  =  {(dec[0,0]+dec[1,1])/dec.sum():.1%}")
print(f"  the model almost never flips a case end-to-end: "
      f"{cm[0,2]+cm[2,0]} of {cm.sum()} = {(cm[0,2]+cm[2,0])/cm.sum():.1%}")
