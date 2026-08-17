"""Export everything the reviewer console needs as one compact JSON blob.

The model is linear, so a case's score decomposes exactly into
    z = rest + sum over adjustable groups of (A_g * v_g - B_g)
where `rest` folds in every feature the console does not expose. That lets the page
recompute the real verdict live from six controls instead of shipping 41 numbers a case.
"""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from featurize import build_features
from model import CLASSES, CLASS_TO_ORD, OrdinalLogit
from pipeline import BASE, build_evidence, load
from predict import COLS, C_REG, priority

# adjustable control -> the features it moves, and the natural range of the control
GROUPS = [
    ("Recent evidence direction", ["core_sh90", "core_sh150", "core_shflat",
                                   "adr_sh150", "ext_sh150", "ttl_sh150", "lic_sh150"],
     "away from DE", "toward DE"),
    ("Current address",  ["adr_open_de", "adr_open_last_de"], "another state", "Delaware"),
    ("Latest title",     ["ttl_last_de", "ttl_recupd_de"],    "another state", "Delaware"),
    ("Latest licence",   ["lic_last_de", "lic_credupd_de"],   "another state", "Delaware"),
    ("T1 update direction", ["t1_de", "t1_last_de"],          "away from DE", "toward DE"),
    ("Observed tag",     ["obs_state_oos"],                   "in state", "out of state"),
]
T1_CENTRED = {"t1_de", "t1_last_de"}     # these are stored as deviations from 0.5


def main():
    D = load()
    cand = D["cand"].reset_index(drop=True)
    ev = build_evidence(D, verbose=False)
    t1_ref = pd.to_datetime(D["up"].observed_date).max()
    F = {p: build_features(ev, cand, p, t1_ref) for p in ("T0", "T1")}

    lab = pd.read_csv(BASE + "Development_Labels/Development_Labels.csv")
    pos = {c: i for i, c in enumerate(cand.candidate_record_id)}
    li = np.array([pos[c] for c in lab.candidate_record_id])
    Xtr = pd.concat([F["T0"].iloc[li], F["T1"].iloc[li]], ignore_index=True)[COLS]
    ytr = np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0) + list(lab.label_t1)])
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    A = sc.fit_transform(imp.fit_transform(Xtr))
    clf = OrdinalLogit(C=C_REG).fit(A, ytr)

    idx = {c: i for i, c in enumerate(COLS)}
    beta, mean, std = clf.coef_, sc.mean_, sc.scale_

    # per-group constants: contribution(v) = A_g * v - B_g
    gA, gB, gnames = [], [], []
    for name, feats, lo_lab, hi_lab in GROUPS:
        a = sum(beta[idx[f]] / std[idx[f]] for f in feats)
        b = sum(beta[idx[f]] * mean[idx[f]] / std[idx[f]] for f in feats)
        gA.append(a); gB.append(b); gnames.append([name, lo_lab, hi_lab])

    audit = pd.read_csv("case_audit.csv")
    aud = {p: audit[audit.phase == p].set_index("candidate_record_id") for p in ("T0", "T1")}
    labels = dict(zip(lab.candidate_record_id, zip(lab.label_t0, lab.label_t1)))

    cases = []
    for phase in ("T0", "T1"):
        Xp = pd.DataFrame(imp.transform(F[phase][COLS]), columns=COLS)
        Z = sc.transform(Xp)
        z_total = Z @ beta
        # current control value per group = the natural value of its first member feature
        vals = []
        for name, feats, *_ in GROUPS:
            f0 = feats[0]
            v = Xp[f0].values + (0.5 if f0 in T1_CENTRED else 0.0)
            vals.append(np.clip(v, 0, 1))
        contrib = sum(np.asarray(gA[g]) * vals[g] - gB[g] for g in range(len(GROUPS)))
        rest = z_total - contrib
        P = clf.predict_proba(Z)
        cases.append(dict(z=z_total, rest=rest, vals=vals, P=P,
                          pri=priority(P), aud=aud[phase]))

    # positional arrays: object keys repeated 12,000 times are most of the payload
    # case  = [id, tag, seen, truth|0, t0, t1]
    # phase = [pWarr, pInsuf, pri, rest, [v...], n, freshDays, [[feed,state,date]...], upd]
    FEEDS = ("address", "licence", "title", "external")
    UPD = {"toward DE": 1, "away from DE": 2, "mixed": 3, "no update": 0}
    out = []
    for i, cid in enumerate(cand.candidate_record_id):
        rec = [cid.replace("CAN-", ""), cand.observed_state.values[i],
               cand.candidate_observed_date.values[i][2:],          # yy-mm-dd
               list(labels[cid] and [CLASS_TO_ORD[labels[cid][0]],
                                     CLASS_TO_ORD[labels[cid][1]]]) if cid in labels else 0]
        for k in (0, 1):
            c, a = cases[k], cases[k]["aud"].loc[cid]
            feeds = []
            for f in FEEDS:
                st, dt = a[f"latest_{f}_state"], a[f"latest_{f}_date"]
                feeds.append([st if isinstance(st, str) else 0,
                              dt[2:] if isinstance(dt, str) else 0])
            rec.append([
                round(float(c["P"][i][2]), 3), round(float(c["P"][i][1]), 3),
                round(float(c["pri"][i]), 3), round(float(c["rest"][i]), 4),
                [round(float(c["vals"][g][i]), 3) for g in range(len(GROUPS))],
                round(float(a.evidence_records_linked), 1),
                -1 if pd.isna(a.days_since_newest_record) else int(a.days_since_newest_record),
                feeds,
                0 if k == 0 else UPD.get(str(a.t1_update_direction), 0),
            ])
        out.append(rec)

    blob = {"classes": CLASSES, "groups": gnames, "feeds": list(FEEDS),
            "upd": ["no update", "toward DE", "away from DE", "mixed"],
            "gA": [round(float(x), 6) for x in gA], "gB": [round(float(x), 6) for x in gB],
            "th": [round(float(clf.theta0_), 6), round(float(clf.theta1_), 6)],
            "cases": out}
    js = json.dumps(blob, separators=(",", ":"))
    open("dist/console_data.json", "w").write(js)
    print(f"wrote dist/console_data.json  {len(js)/1e6:.2f} MB  {len(out)} cases")

    # verify the live recomputation reproduces the shipped probabilities exactly
    import random
    random.seed(0)
    worst = 0.0
    for r in random.sample(out, 400):
        for ph in (4, 5):
            d = r[ph]
            z = d[3] + sum(gA[g] * d[4][g] - gB[g] for g in range(len(GROUPS)))
            c0 = 1 / (1 + np.exp(-(clf.theta0_ - z)))
            c1 = 1 / (1 + np.exp(-(clf.theta1_ - z)))
            worst = max(worst, abs((1 - c1) - d[0]), abs((c1 - c0) - d[1]))
    print(f"live-recompute check: largest probability error over 400 cases = {worst:.2e}")


if __name__ == "__main__":
    main()
