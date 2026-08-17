"""Entity resolution, second design.

What the data actually looks like (verified on the vehicle-title feed, where
`vehicle_ref` gives an independent grouping key):

* Each candidate owns exactly one vehicle and has four title rows for it.
* In three of those four rows the owner name is the candidate's, possibly lightly
  corrupted (truncation, one substitution, an adjacent transposition, a stray
  space).  **In the fourth row the name is replaced outright** -- clustering names
  inside a vehicle gives a 3 + 1 split for 98% of vehicles.

The same 3-good-plus-1-garbled pattern shows up in the licence feed when rows are
grouped by date of birth, so it is the generator's noise model rather than a
quirk of one table.

Consequences:

* For the title feed the garbled row is still recoverable -- `vehicle_ref` says
  which vehicle it belongs to, so it can be attached to whoever owns the other
  three rows.  Doing this lifts the title signal substantially.
* For the address, licence, external and work feeds there is no such key.  The
  garbled row is unrecoverable, and forcing an assignment onto it just injects a
  wrong state into the candidate's evidence.  Those feeds therefore use
  confidence-weighted name matching and simply accept ~75% recall.
"""
import numpy as np
import pandas as pd
from collections import defaultdict

from er import norm, sim, NameMatcher

NAME_CLUSTER_SIM = 0.62
MIN_LINK = 0.60
SHARPEN = 8            # exponent used to turn match scores into soft weights
INFERRED_WEIGHT = 0.85  # weight for a title row recovered purely via vehicle_ref


class Resolver:
    """Fuzzy name -> candidate lookup with soft (probabilistic) assignment."""

    def __init__(self, cand):
        self.cand = cand.reset_index(drop=True)
        self.n = len(self.cand)
        self.ln = np.array([norm(x) for x in self.cand.last_name])
        self.fn = np.array([norm(x) for x in self.cand.first_name])
        self.dob = self.cand.date_of_birth.astype(str).values
        self.by_ln, self.by_fn, self.by_dob = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(self.n):
            self.by_ln[self.ln[i]].append(i)
            self.by_fn[self.fn[i]].append(i)
            self.by_dob[self.dob[i]].append(i)
        self.M_ln = NameMatcher(self.by_ln.keys())
        self.M_fn = NameMatcher(self.by_fn.keys())
        self._pair, self._lnc, self._fnc = {}, {}, {}

    def _ln_match(self, l):
        if l not in self._lnc:
            self._lnc[l] = self.M_ln.match(l, min_sim=0.55, top=8)
        return self._lnc[l]

    def _fn_match(self, f):
        if f not in self._fnc:
            self._fnc[f] = self.M_fn.match(f, min_sim=0.70, top=8)
        return self._fnc[f]

    def score(self, fn_raw, ln_raw, dob_raw=None, top=4):
        """[(candidate_index, raw_score)] above MIN_LINK, best first."""
        f, l = norm(fn_raw), norm(ln_raw)
        dob = dob_raw if isinstance(dob_raw, str) and dob_raw.strip() else None
        key = (f, l, dob)
        if key in self._pair:
            return self._pair[key]

        lnm = dict(self._ln_match(l))
        pool = set()
        for name in lnm:
            pool.update(self.by_ln[name])
        for name, _ in self._fn_match(f):
            pool.update(self.by_fn[name])
        if dob:
            pool.update(self.by_dob.get(dob, ()))

        out = []
        for i in pool:
            ls = lnm.get(self.ln[i], sim(l, self.ln[i]))
            fs = sim(f, self.fn[i])
            if ls <= 0 or fs <= 0:
                continue
            s = 0.5 * ls + 0.5 * fs
            if dob is not None and dob == self.dob[i]:
                s = min(1.0, s + 0.20)
            if s >= MIN_LINK:
                out.append((i, s))
        out.sort(key=lambda x: -x[1])
        out = out[:top]
        self._pair[key] = out
        return out

    def soft(self, fn_raw, ln_raw, dob_raw=None):
        """[(candidate_index, weight)] with weights summing to 1 (empty if no match)."""
        raw = self.score(fn_raw, ln_raw, dob_raw)
        if not raw:
            return []
        w = np.array([s for _, s in raw]) ** SHARPEN
        w /= w.sum()
        return [(i, float(x)) for (i, _), x in zip(raw, w)]


def link_by_name(res, df, fcol, lcol, dcol=None):
    """Soft name linking. Returns (row_idx, cand_idx, weight)."""
    ri, ci, wt = [], [], []
    f, l = df[fcol].values, df[lcol].values
    d = df[dcol].values if dcol else [None] * len(df)
    for r in range(len(df)):
        for i, w in res.soft(f[r], l[r], d[r]):
            ri.append(r); ci.append(i); wt.append(w)
    return np.array(ri, int), np.array(ci, int), np.array(wt, float)


def _name_clusters(names):
    """Greedy single-link clustering of (first, last) name pairs."""
    m = len(names)
    lab = [-1] * m
    k = 0
    for i in range(m):
        if lab[i] >= 0:
            continue
        lab[i] = k
        for j in range(i + 1, m):
            if lab[j] < 0:
                s = 0.5 * sim(names[i][0], names[j][0]) + 0.5 * sim(names[i][1], names[j][1])
                if s >= NAME_CLUSTER_SIM:
                    lab[j] = k
        k += 1
    return lab


def link_titles_by_vehicle(res, ttl, fcol="owner_first_name", lcol="owner_last_name"):
    """Attach every title row to a candidate using `vehicle_ref` as the group key.

    Rows are clustered by name inside each vehicle; each cluster is assigned to the
    candidate it matches best.  A singleton cluster (the garbled row) is attached to
    the vehicle's dominant owner when there is exactly one, at a reduced weight.
    """
    f = np.array([norm(x) for x in ttl[fcol]])
    l = np.array([norm(x) for x in ttl[lcol]])
    ri, ci, wt = [], [], []
    for _, idx in ttl.groupby("vehicle_ref").indices.items():
        idx = np.asarray(idx)
        lab = _name_clusters([(f[r], l[r]) for r in idx])
        groups = defaultdict(list)
        for r, k in zip(idx, lab):
            groups[k].append(r)
        owners = {}
        for k, rows in groups.items():
            votes = defaultdict(float)
            for r in rows:
                for i, s in res.score(ttl[fcol].values[r], ttl[lcol].values[r]):
                    votes[i] += s ** SHARPEN
            if votes:
                tot = sum(votes.values())
                i = max(votes, key=votes.get)
                owners[k] = (i, votes[i] / tot)
        big = [k for k, rows in groups.items() if len(rows) >= 2 and k in owners]
        for k, rows in groups.items():
            if k in owners and len(rows) >= 2:
                i, conf = owners[k]
                for r in rows:
                    ri.append(r); ci.append(i); wt.append(min(1.0, 0.6 + 0.4 * conf))
            elif len(big) == 1:                       # garbled row -> dominant owner
                i, conf = owners[big[0]]
                for r in rows:
                    ri.append(r); ci.append(i); wt.append(INFERRED_WEIGHT * min(1.0, 0.6 + 0.4 * conf))
            elif k in owners:                          # singleton with its own match
                i, conf = owners[k]
                for r in rows:
                    ri.append(r); ci.append(i); wt.append(0.5 * min(1.0, 0.6 + 0.4 * conf))
    return np.array(ri, int), np.array(ci, int), np.array(wt, float)


def vehicle_owner_map(ri, ci, wt, ttl):
    """vehicle_ref -> (candidate_index, weight) using the resolved title rows."""
    vr = ttl.vehicle_ref.values
    votes = defaultdict(lambda: defaultdict(float))
    for r, i, w in zip(ri, ci, wt):
        votes[vr[r]][i] += w
    out = {}
    for v, d in votes.items():
        tot = sum(d.values())
        i = max(d, key=d.get)
        out[v] = (i, d[i] / tot if tot else 0.0)
    return out
