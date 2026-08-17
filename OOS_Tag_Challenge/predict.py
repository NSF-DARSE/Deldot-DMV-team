"""End-to-end run: resolve -> featurise -> fit -> score every candidate/phase.

Outputs
-------
case_predictions.csv   the submission file (exact template schema)
case_audit.csv         per-case evidence trail for staff review (supporting file)
model_card.json        fitted coefficients and CV metrics
"""
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from featurize import CORE, build_features
from model import CLASSES, CLASS_TO_ORD, OrdinalLogit
from pipeline import BASE, build_evidence, load

C_REG = 0.02
CV_REPEATS = 20

CORE_F = [f"core_{k}" for k in ("sh90", "sh150", "sh365", "shflat", "last_de", "last_age")] + \
         ["core_last_mean", "core_last_std", "core_sh150_std"]
PER_STREAM_F = [f"{s}_{k}" for s in CORE for k in ("sh150", "last_de")]
CURRENT_ADDR_F = ["adr_open_de", "adr_open_n", "adr_open_last_de", "adr_open_age"]
CATEGORY_F = ["ttl_recupd_de", "lic_credupd_de", "ext_refupd_de"]
SUFFICIENCY_F = ["ev_n", "ev_nullshare", "ev_limited", "ev_badstatus", "ev_n180",
                 "ev_n365", "ev_linkconf", "ev_nstates", "ev_minage"]
T1_F = ["t1_n", "t1_de", "t1_last_de", "t1_ttl_de", "t1_adr_de", "t1_delta", "phase_t1"]
COLS = (CORE_F + PER_STREAM_F + CURRENT_ADDR_F + CATEGORY_F
        + SUFFICIENCY_F + T1_F + ["obs_state_oos"])


def priority(P):
    """Review priority in [0, 1].

    A case that warrants review takes full priority; a case the evidence cannot
    decide still needs a person to look at it, so it takes half; a case that
    clearly does not warrant review takes almost none.
    """
    return P[:, 2] + 0.5 * P[:, 1]


STREAM_LABEL = {"adr": "address", "lic": "licence", "ttl": "title",
                "ext": "external", "wrk": "work"}

# Individual coefficients are shrunk and the direction features are collinear, so
# feature-level contributions can look contradictory.  Block sums are stable, and
# are what the audit trail leads with.
DRIVER_BLOCKS = {
    "evidence direction": CORE_F + PER_STREAM_F + CATEGORY_F,
    "current address": CURRENT_ADDR_F,
    "evidence sufficiency": SUFFICIENCY_F,
    "T1 updates": T1_F,
    "observed tag": ["obs_state_oos"],
}

# plain-English names so the audit trail reads without the code next to it
FEATURE_LABEL = {
    "core_sh90": "DE share of evidence, last ~3 months",
    "core_sh150": "DE share of evidence, last ~5 months",
    "core_sh365": "DE share of evidence, last ~12 months",
    "core_shflat": "DE share of evidence, all time",
    "core_last_de": "state on the newest record overall",
    "core_last_age": "age of the newest record",
    "core_last_mean": "how many feeds end in DE",
    "adr_open_de": "current address is in DE",
    "adr_open_n": "has a current address on file",
    "adr_open_last_de": "newest current-address record is DE",
    "adr_open_age": "age of the current-address record",
    "ttl_recupd_de": "title updates lean DE",
    "lic_credupd_de": "licence updates lean DE",
    "ext_refupd_de": "external reference updates lean DE",
    "core_last_std": "disagreement between feeds",
    "core_sh150_std": "recent disagreement between feeds",
    "adr_sh150": "address feed direction", "adr_last_de": "state on the newest address",
    "lic_sh150": "licence feed direction", "lic_last_de": "state on the newest licence",
    "ttl_sh150": "title feed direction", "ttl_last_de": "state on the newest title",
    "ext_sh150": "external feed direction", "ext_last_de": "state on the newest external signal",
    "ev_n": "volume of linked evidence", "ev_nullshare": "share of records with no state",
    "ev_limited": "share of limited-quality records",
    "ev_badstatus": "share of expired/superseded records",
    "ev_n180": "records in the last 6 months", "ev_n365": "records in the last 12 months",
    "ev_linkconf": "identity-match confidence", "ev_nstates": "number of distinct states seen",
    "ev_minage": "days since the newest record",
    "t1_n": "volume of T1 updates", "t1_de": "direction of the T1 updates",
    "t1_last_de": "direction of the newest T1 update",
    "t1_ttl_de": "direction of the T1 title update",
    "t1_adr_de": "direction of the T1 address update",
    "t1_delta": "T1 updates vs prior evidence",
    "phase_t1": "phase is T1", "obs_state_oos": "observed tag is out of state"}


def _latest_state(ev, cand, stream, is_t1=None):
    """Most recent state-bearing record per candidate for one stream."""
    sub = ev[(ev.stream == stream) & ev.state.notna()]
    if is_t1 is not None:
        sub = sub[sub.is_t1 == is_t1]
    last = sub.sort_values(["cand", "date"]).groupby("cand").tail(1).set_index("cand")
    idx = range(len(cand))
    return last.state.reindex(idx).values, last.date.reindex(idx).dt.date.astype("string").values


def write_audit(cand, ev, F, pred, clf, imp, sc, path="case_audit.csv"):
    """Per-case evidence trail so a reviewer can see why a case scored as it did."""
    rows = []
    contrib_names = np.array(COLS)
    for phase in ("T0", "T1"):
        Fp = F[phase]
        Z = sc.transform(imp.transform(Fp[COLS]))
        contrib = Z * clf.coef_                     # standardised contribution to the index
        order = np.argsort(-np.abs(contrib), axis=1)[:, :3]
        block_of = {c: b for b, cols in DRIVER_BLOCKS.items() for c in cols}
        block_sum = {
            b: contrib[:, [COLS.index(c) for c in cols]].sum(axis=1)
            for b, cols in DRIVER_BLOCKS.items()
        }
        shown = [b for b in DRIVER_BLOCKS if not (phase == "T0" and b == "T1 updates")]
        summary = [
            "; ".join(f"{b} {block_sum[b][i]:+.2f}"
                      for b in sorted(shown, key=lambda b: -abs(block_sum[b][i])))
            for i in range(len(Fp))
        ]
        drivers = [
            "; ".join(
                f"{FEATURE_LABEL.get(contrib_names[j], contrib_names[j])} "
                f"({'+' if contrib[i, j] >= 0 else '-'}{abs(contrib[i, j]):.2f})"
                for j in order[i])
            for i in range(len(Fp))
        ]
        t = pd.DataFrame({
            "candidate_record_id": cand.candidate_record_id.values,
            "phase": phase,
            "observed_tag_state": cand.observed_state.values,
            "evidence_records_linked": Fp["ev_n"].round(1).values,
            "de_share_recent": Fp["core_sh150"].round(3).values,
            "de_share_all_time": Fp["core_shflat"].round(3).values,
            "days_since_newest_record": Fp["ev_minage"].values,
            "evidence_index": clf.decision_function(Z).round(3),
            "driver_summary": summary,
            "top_drivers": drivers,
        })
        for st in ("adr", "lic", "ttl", "ext"):
            s, d = _latest_state(ev if phase == "T1" else ev[~ev.is_t1], cand, st)
            t[f"latest_{STREAM_LABEL[st]}_state"] = s
            t[f"latest_{STREAM_LABEL[st]}_date"] = d
        if phase == "T1":
            t["t1_update_direction"] = np.where(
                Fp["t1_n"].values <= 0, "no update",
                np.where(Fp["t1_de"].values > 0, "toward DE",
                         np.where(Fp["t1_de"].values < 0, "away from DE", "mixed")))
        else:
            t["t1_update_direction"] = "T0 (no updates yet)"
        rows.append(t)
    audit = pd.concat(rows, ignore_index=True)
    audit = pred[["candidate_record_id", "phase", "predicted_class", "p_review_warranted",
                  "p_review_not_warranted", "p_insufficient_evidence", "review_priority"]].merge(
        audit, on=["candidate_record_id", "phase"], how="left")
    audit.to_csv(path, index=False)
    return audit


def cross_validate(X, y, groups, repeats=CV_REPEATS):
    oof = np.zeros((len(y), 3))
    for rep in range(repeats):
        rs = np.random.RandomState(100 + rep)
        u = np.unique(groups)
        gmap = dict(zip(u, rs.permutation(len(u))))
        gg = np.array([gmap[g] for g in groups])
        for tr, te in GroupKFold(n_splits=5).split(X, y, gg):
            imp, sc = SimpleImputer(strategy="median"), StandardScaler()
            A = sc.fit_transform(imp.fit_transform(X.iloc[tr]))
            B = sc.transform(imp.transform(X.iloc[te]))
            oof[te] += OrdinalLogit(C=C_REG).fit(A, y[tr]).predict_proba(B)
    return oof / repeats


def main():
    print("1/5 loading + resolving")
    D = load()
    cand = D["cand"].reset_index(drop=True)
    ev = build_evidence(D)

    print("2/5 featurising")
    t1_ref = pd.to_datetime(D["up"].observed_date).max()
    F = {"T0": build_features(ev, cand, "T0", t1_ref),
         "T1": build_features(ev, cand, "T1", t1_ref)}

    lab = pd.read_csv(BASE + "Development_Labels/Development_Labels.csv")
    pos = {c: i for i, c in enumerate(cand.candidate_record_id)}
    li = np.array([pos[c] for c in lab.candidate_record_id])
    Xtr = pd.concat([F["T0"].iloc[li], F["T1"].iloc[li]], ignore_index=True)[COLS]
    ytr = np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0) + list(lab.label_t1)])
    groups = np.concatenate([li, li])

    print("3/5 cross-validating")
    oof = cross_validate(Xtr, ytr, groups)
    metrics = {
        "n_labelled_rows": int(len(ytr)),
        "log_loss": float(log_loss(ytr, oof, labels=[0, 1, 2])),
        "log_loss_prior_baseline": float(
            log_loss(ytr, np.tile(np.bincount(ytr) / len(ytr), (len(ytr), 1)), labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(ytr, oof.argmax(1))),
        "macro_f1": float(f1_score(ytr, oof.argmax(1), average="macro")),
        "macro_auc": float(np.mean([roc_auc_score((ytr == k).astype(int), oof[:, k])
                                    for k in range(3)])),
        "priority_auc_review_warranted": float(
            roc_auc_score((ytr == 2).astype(int), priority(oof))),
    }
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("4/5 fitting final model on all labelled rows")
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    A = sc.fit_transform(imp.fit_transform(Xtr))
    clf = OrdinalLogit(C=C_REG).fit(A, ytr)

    print("5/5 scoring all candidates")
    out = []
    for phase in ("T0", "T1"):
        Xp = sc.transform(imp.transform(F[phase][COLS]))
        P = clf.predict_proba(Xp)
        P = P / P.sum(axis=1, keepdims=True)
        out.append(pd.DataFrame({
            "candidate_record_id": cand.candidate_record_id.values,
            "phase": phase,
            "predicted_class": [CLASSES[i] for i in P.argmax(1)],
            "p_review_warranted": P[:, 2],
            "p_review_not_warranted": P[:, 0],
            "p_insufficient_evidence": P[:, 1],
            "review_priority": priority(P),
            "_index": clf.decision_function(Xp),
        }))
    pred = pd.concat(out, ignore_index=True)

    # match the template's row order exactly
    tpl = pd.read_csv(BASE + "Submission_Template.csv")[["candidate_record_id", "phase"]]
    pred = tpl.merge(pred, on=["candidate_record_id", "phase"], how="left")
    assert pred.p_review_warranted.notna().all(), "unscored template rows"

    sub = pred.drop(columns=["_index"]).copy()
    prob_cols = ["p_review_warranted", "p_review_not_warranted", "p_insufficient_evidence"]
    sub[prob_cols] = sub[prob_cols].round(6)
    # absorb rounding drift into the largest probability so each row sums to exactly 1
    drift = 1.0 - sub[prob_cols].sum(axis=1)
    big = sub[prob_cols].values.argmax(1)
    vals = sub[prob_cols].values
    vals[np.arange(len(vals)), big] = (vals[np.arange(len(vals)), big] + drift.values).round(6)
    sub[prob_cols] = vals
    sub["review_priority"] = sub["review_priority"].clip(0, 1).round(6)
    sub.to_csv("case_predictions.csv", index=False)

    write_audit(cand, ev, F, pred, clf, imp, sc)

    coefs = dict(sorted(zip(COLS, clf.coef_.tolist()), key=lambda kv: -abs(kv[1])))
    json.dump({"regularisation_C": C_REG, "cv_metrics": metrics,
               "thresholds": [float(clf.theta0_), float(clf.theta1_)],
               "standardised_coefficients": coefs},
              open("model_card.json", "w"), indent=2)
    return D, cand, ev, F, pred


if __name__ == "__main__":
    main()
    print("wrote case_predictions.csv, model_card.json")
