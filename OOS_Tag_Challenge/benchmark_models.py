"""Every model family on identical folds, tuned, with the trees given a fair shot.

Fairness matters here because the headline claim is that a plain ordinal logistic beats
the boosters. So the boosters get: a hyperparameter sweep, the option of running under the
same ordinal decomposition the winner uses, and a genuine categorical feature (the
observed tag state, nine levels) that CatBoost can exploit and the linear model cannot.
"""
import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from model import OrdinalLogit, CLASS_TO_ORD
from featurize import build_features
from pipeline import load, BASE, build_evidence
from predict import COLS, C_REG

D = load(); cand = D["cand"].reset_index(drop=True)
ev = build_evidence(D, verbose=False); t1 = pd.to_datetime(D["up"].observed_date).max()
F0 = build_features(ev, cand, "T0", t1); F1 = build_features(ev, cand, "T1", t1)
lab = pd.read_csv(BASE + "Development_Labels/Development_Labels.csv")
pos = {c: i for i, c in enumerate(cand.candidate_record_id)}
li = np.array([pos[c] for c in lab.candidate_record_id])
X = pd.concat([F0.iloc[li], F1.iloc[li]], ignore_index=True)[COLS].copy()
# a real categorical, coded as an integer, for the tree models
state_codes = pd.Categorical(cand.observed_state).codes
X["tag_state"] = np.concatenate([state_codes[li], state_codes[li]])
y = np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0) + list(lab.label_t1)])
grp = np.concatenate([li, li])
CAT_IDX = [X.columns.get_loc("tag_state")]


def ordinalise(make, Xtr, ytr, Xte):
    """Give any binary classifier the ordinal structure: P(y>=1) and P(y>=2)."""
    pa = make().fit(Xtr, (ytr >= 1).astype(int)).predict_proba(Xte)[:, 1]
    pb = make().fit(Xtr, (ytr >= 2).astype(int)).predict_proba(Xte)[:, 1]
    pb = np.minimum(pa, pb)                       # keep the cumulative order
    P = np.column_stack([1 - pa, pa - pb, pb])
    return np.clip(P, 1e-6, 1) / np.clip(P, 1e-6, 1).sum(1, keepdims=True)


def as_cat(M):
    """CatBoost rejects float categoricals; hand it strings for that one column."""
    O = M.astype(object)
    for j in CAT_IDX:
        O[:, j] = [str(int(v)) for v in M[:, j]]
    return O


def run(name, fit_predict, scale=False, nrep=6):
    LL = AC = AU = F1S = 0.0
    for rep in range(nrep):
        rs = np.random.RandomState(rep); u = np.unique(grp)
        gm = dict(zip(u, rs.permutation(len(u)))); gg = np.array([gm[g] for g in grp])
        oof = np.zeros((len(y), 3))
        for tr, te in GroupKFold(n_splits=5).split(X, y, gg):
            A, B = X.iloc[tr].values, X.iloc[te].values
            imp = SimpleImputer(strategy="median")
            A = imp.fit_transform(A); B = imp.transform(B)
            if scale:
                sc = StandardScaler(); A = sc.fit_transform(A); B = sc.transform(B)
            oof[te] = fit_predict(A, y[tr], B)
        LL += log_loss(y, oof, labels=[0, 1, 2]) / nrep
        AC += accuracy_score(y, oof.argmax(1)) / nrep
        F1S += f1_score(y, oof.argmax(1), average="macro") / nrep
        AU += np.mean([roc_auc_score((y == k).astype(int), oof[:, k]) for k in range(3)]) / nrep
    print(f"  {name:44s} ll={LL:.4f}  acc={AC:.4f}  f1={F1S:.4f}  auc={AU:.4f}", flush=True)
    return LL, name


results = []
print("BASELINE")
prior = np.bincount(y) / len(y)
print(f"  {'class prior':44s} ll={log_loss(y, np.tile(prior, (len(y),1)), labels=[0,1,2]):.4f}"
      f"  acc={prior.max():.4f}")

print("\nLINEAR")
results.append(run("ordinal logistic (shipped)",
    lambda A, yt, B: OrdinalLogit(C=C_REG).fit(A, yt).predict_proba(B), scale=True))
for C in (0.005, 0.01, 0.05):
    results.append(run(f"multinomial logistic  C={C}",
        lambda A, yt, B, C=C: LogisticRegression(C=C, max_iter=4000).fit(A, yt).predict_proba(B),
        scale=True))

print("\nRANDOM FOREST")
for leaf in (5, 15, 30):
    results.append(run(f"random forest  min_leaf={leaf}",
        lambda A, yt, B, l=leaf: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=l, random_state=0, n_jobs=-1
        ).fit(A, yt).predict_proba(B)))

print("\nXGBOOST")
for d, lr, n in ((2, .05, 250), (3, .05, 200), (2, .02, 500), (4, .1, 120)):
    results.append(run(f"xgboost  depth={d} lr={lr} n={n}",
        lambda A, yt, B, d=d, lr=lr, n=n: XGBClassifier(
            max_depth=d, learning_rate=lr, n_estimators=n, subsample=.8,
            colsample_bytree=.8, reg_lambda=5.0, min_child_weight=5,
            objective="multi:softprob", num_class=3, tree_method="hist",
            verbosity=0, n_jobs=-1).fit(A, yt).predict_proba(B)))
results.append(run("xgboost  depth=2 + ORDINAL decomposition",
    lambda A, yt, B: ordinalise(lambda: XGBClassifier(
        max_depth=2, learning_rate=.05, n_estimators=250, subsample=.8,
        colsample_bytree=.8, reg_lambda=5.0, min_child_weight=5,
        verbosity=0, n_jobs=-1), A, yt, B)))

print("\nLIGHTGBM")
for leaves, n in ((4, 300), (8, 200)):
    results.append(run(f"lightgbm  leaves={leaves} n={n}",
        lambda A, yt, B, lv=leaves, n=n: LGBMClassifier(
            num_leaves=lv, n_estimators=n, learning_rate=.05, min_child_samples=25,
            reg_lambda=5.0, subsample=.8, colsample_bytree=.8, verbose=-1, n_jobs=-1
        ).fit(A, yt).predict_proba(B)))

print("\nCATBOOST  (tag_state passed as a genuine categorical)")
for d, n in ((3, 400), (4, 300), (2, 600)):
    results.append(run(f"catboost  depth={d} n={n}",
        lambda A, yt, B, d=d, n=n: CatBoostClassifier(
            depth=d, iterations=n, learning_rate=.05, l2_leaf_reg=8,
            loss_function="MultiClass", cat_features=CAT_IDX, verbose=0,
            allow_writing_files=False
        ).fit(as_cat(A), yt).predict_proba(as_cat(B))))
results.append(run("catboost  depth=3 + ORDINAL decomposition",
    lambda A, yt, B: ordinalise(lambda: CatBoostClassifier(
        depth=3, iterations=400, learning_rate=.05, l2_leaf_reg=8,
        cat_features=CAT_IDX, verbose=0, allow_writing_files=False),
        as_cat(A), yt, as_cat(B))))

print("\nHIST GRADIENT BOOSTING")
results.append(run("hist gradient boosting  depth=2",
    lambda A, yt, B: HistGradientBoostingClassifier(
        max_depth=2, max_iter=150, learning_rate=.05, l2_regularization=5.0,
        min_samples_leaf=25).fit(A, yt).predict_proba(B)))

results.sort()
print("\n" + "=" * 74 + "\nRANKED BY LOG LOSS (lower is better)")
for i, (ll, nm) in enumerate(results, 1):
    print(f"  {i:2d}. {nm:44s} {ll:.4f}")
