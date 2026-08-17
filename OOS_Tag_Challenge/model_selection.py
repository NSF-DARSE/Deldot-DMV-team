"""Reproduces the model- and feature-selection comparisons quoted in METHOD.md.

Run: python model_selection.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from featurize import CORE, build_features
from model import CLASS_TO_ORD, OrdinalLogit
from pipeline import BASE, build_evidence, load

warnings.filterwarnings("ignore")

CORE_F = [f"core_{k}" for k in ("sh90", "sh150", "sh365", "shflat", "last_de", "last_age")] + \
         ["core_last_mean", "core_last_std", "core_sh150_std"]
PER = [f"{s}_{k}" for s in CORE for k in ("sh150", "last_de")]
SUF = ["ev_n", "ev_nullshare", "ev_limited", "ev_badstatus", "ev_n180", "ev_n365",
       "ev_linkconf", "ev_nstates", "ev_minage"]
T1F = ["t1_n", "t1_de", "t1_last_de", "t1_ttl_de", "t1_adr_de", "t1_delta", "phase_t1"]
CHOSEN = CORE_F + PER + SUF + T1F + ["obs_state_oos"]
SLIM = ["core_sh150", "core_sh365", "core_last_mean", "core_last_std", "core_last_age"] + \
       [f"{s}_sh150" for s in CORE] + \
       ["ev_n", "ev_nullshare", "ev_limited", "ev_badstatus", "ev_n180", "ev_linkconf",
        "ev_nstates", "ev_minage"] + \
       ["t1_n", "t1_de", "t1_delta", "phase_t1", "obs_state_oos"]


def cv(X, y, grp, make, nrep=12):
    ll, ac, au, f1 = [], [], [], []
    for rep in range(nrep):
        rs = np.random.RandomState(rep)
        u = np.unique(grp)
        gm = dict(zip(u, rs.permutation(len(u))))
        gg = np.array([gm[g] for g in grp])
        oof = np.zeros((len(y), 3))
        for tr, te in GroupKFold(n_splits=5).split(X, y, gg):
            imp, sc = SimpleImputer(strategy="median"), StandardScaler()
            A = sc.fit_transform(imp.fit_transform(X.iloc[tr]))
            B = sc.transform(imp.transform(X.iloc[te]))
            oof[te] = make().fit(A, y[tr]).predict_proba(B)
        ll.append(log_loss(y, oof, labels=[0, 1, 2]))
        ac.append(accuracy_score(y, oof.argmax(1)))
        f1.append(f1_score(y, oof.argmax(1), average="macro"))
        au.append(np.mean([roc_auc_score((y == k).astype(int), oof[:, k]) for k in range(3)]))
    return np.mean(ll), np.mean(ac), np.mean(au), np.mean(f1)


def main():
    D = load()
    cand = D["cand"].reset_index(drop=True)
    n = len(cand)
    ev = build_evidence(D, verbose=False)
    t1_ref = pd.to_datetime(D["up"].observed_date).max()
    F0 = build_features(ev, cand, "T0", t1_ref)
    F1 = build_features(ev, cand, "T1", t1_ref)
    lab = pd.read_csv(BASE + "Development_Labels/Development_Labels.csv")
    pos = {c: i for i, c in enumerate(cand.candidate_record_id)}
    li = np.array([pos[c] for c in lab.candidate_record_id])
    FULL = pd.concat([F0.iloc[li], F1.iloc[li]], ignore_index=True)
    y = np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0) + list(lab.label_t1)])
    grp = np.concatenate([li, li])

    print("=" * 78 + "\nA. Model family, on the chosen feature set\n" + "=" * 78)
    prior = np.bincount(y) / len(y)
    print(f"  class-prior baseline      ll={log_loss(y, np.tile(prior, (len(y), 1)), labels=[0,1,2]):.4f}"
          f"  acc={prior.max():.4f}"
          f"  f1={f1_score(y, np.full(len(y), prior.argmax()), average='macro'):.4f}")
    X = FULL[CHOSEN]
    for name, make in [
        ("ordinal logit C=0.03", lambda: OrdinalLogit(C=0.03)),
        ("multinomial LR C=0.01", lambda: LogisticRegression(C=0.01, max_iter=3000)),
        ("random forest", lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=15,
                                                         random_state=0)),
        ("hist gradient boosting", lambda: HistGradientBoostingClassifier(
            max_depth=2, max_iter=120, learning_rate=0.05, l2_regularization=5.0,
            min_samples_leaf=30)),
    ]:
        r = cv(X, y, grp, make)
        print(f"  {name:24s} ll={r[0]:.4f}  acc={r[1]:.4f}  macroAUC={r[2]:.4f}  macroF1={r[3]:.4f}")

    print("\n" + "=" * 78 + "\nB. Feature set and regularisation\n" + "=" * 78)
    for label, cols in [("chosen (34)", CHOSEN), ("slim (22)", SLIM)]:
        for C in (0.02, 0.03, 0.05, 0.1):
            r = cv(FULL[cols], y, grp, lambda C=C: OrdinalLogit(C=C))
            print(f"  {label:12s} C={C:<5} ll={r[0]:.4f}  acc={r[1]:.4f}  macroAUC={r[2]:.4f}")

    print("\n" + "=" * 78 + "\nC. Aggregating the direction signal (Spearman rho vs ordered T0 label)\n" + "=" * 78)
    e = ev[(~ev.is_t1) & ev.state.notna()].copy()
    ref = pd.to_datetime(cand.candidate_observed_date).values
    e["age"] = ((ref[e.cand.values] - e.date.values) / np.timedelta64(1, "D")).clip(min=0)
    e["de"] = (e.state == "DE").astype(float)
    e = e[e.stream.isin(CORE)]
    y0 = lab.label_t0.map(CLASS_TO_ORD).values

    def rho(v):
        v = np.asarray(v, float)[li]
        m = ~np.isnan(v)
        return spearmanr(v[m], y0[m]).statistic

    def ratio(sub, w):
        num = np.bincount(sub.cand, weights=w * sub.de, minlength=n)
        den = np.bincount(sub.cand, weights=w, minlength=n)
        return np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)

    print(f"  exponential decay, 150-day half-life : {rho(ratio(e, e.w * 0.5 ** (e.age / 150))):+.3f}")
    print(f"  newest record per feed, averaged     : "
          f"{rho(pd.DataFrame({s: e[e.stream == s].sort_values('date').groupby('cand').tail(1).set_index('cand').de.reindex(range(n)) for s in CORE}).mean(axis=1).values):+.3f}")
    for W in (180, 365, 730):
        sub = e[e.age <= W]
        print(f"  hard window, last {W:3d} days          : {rho(ratio(sub, sub.w)):+.3f}")
    for d in (0.2, 0.35, 0.5):
        s2 = e.sort_values(["cand", "date"]).copy()
        s2["rk"] = s2.groupby("cand").cumcount(ascending=False)
        print(f"  rank decay {d}                      : {rho(ratio(s2, s2.w * d ** s2.rk)):+.3f}")
    # per-candidate linear-in-time probability, read off at the reference date
    x = -e.age.values / 365.0
    S1 = np.bincount(e.cand, minlength=n).astype(float)
    Sx = np.bincount(e.cand, weights=x, minlength=n)
    Sxx = np.bincount(e.cand, weights=x * x, minlength=n)
    Sy = np.bincount(e.cand, weights=e.de.values, minlength=n)
    Sxy = np.bincount(e.cand, weights=x * e.de.values, minlength=n)
    det = S1 * (Sxx + 1.0) - Sx * Sx
    a = ((Sxx + 1.0) * Sy - Sx * Sxy) / np.where(np.abs(det) < 1e-9, np.nan, det)
    print(f"  linear-in-time extrapolation         : {rho(a):+.3f}")

    def changepoint(g):
        d = g.sort_values("date").de.values
        if len(d) < 2:
            return d.mean() if len(d) else np.nan
        best = (-1e9, d.mean())
        for k in range(len(d)):
            sc_ = abs(d[k:].mean() - 0.5) * len(d[k:])
            if k:
                sc_ += abs(d[:k].mean() - 0.5) * k
            if sc_ > best[0]:
                best = (sc_, d[k:].mean())
        return best[1]

    print(f"  change-point, post-segment mean      : "
          f"{rho(e.groupby('cand').apply(changepoint).reindex(range(n)).values):+.3f}")


if __name__ == "__main__":
    main()
