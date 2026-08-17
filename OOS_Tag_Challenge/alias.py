"""Second-identity recovery.

The generator replaces the name on one record in four. Those replacements are not random
per row: a candidate is given a persistent fake name that recurs across feeds. The title
feed exposes it, because `vehicle_ref` groups a vehicle's records regardless of the name,
so the odd-one-out row in a 3+1 name split is that candidate's alias.

Learn the alias there, then use it to claim the matching rows in the feeds that have no
grouping key of their own. Aliases that point at more than one candidate are discarded.
"""
import numpy as np
from collections import Counter, defaultdict

from er import norm, sim

ALIAS_WEIGHT = 0.75      # an inferred identity, so it counts for less than a name match


def _clusters(names, thresh=0.62):
    m = len(names)
    lab = [-1] * m
    k = 0
    for i in range(m):
        if lab[i] >= 0:
            continue
        lab[i] = k
        for j in range(i + 1, m):
            if lab[j] < 0 and 0.5 * sim(names[i][0], names[j][0]) + \
                    0.5 * sim(names[i][1], names[j][1]) >= thresh:
                lab[j] = k
        k += 1
    return lab


def learn_aliases(res, ttl, fcol="owner_first_name", lcol="owner_last_name"):
    """{(first, last): candidate_index} for aliases that point at exactly one candidate."""
    f = np.array([norm(x) for x in ttl[fcol]])
    l = np.array([norm(x) for x in ttl[lcol]])
    claims = defaultdict(set)
    for _, idx in ttl.groupby("vehicle_ref").indices.items():
        idx = np.asarray(idx)
        if len(idx) != 4:
            continue
        names = [(f[r], l[r]) for r in idx]
        lab = _clusters(names)
        cnt = Counter(lab)
        big = [k for k, c in cnt.items() if c == 3]
        sing = [k for k, c in cnt.items() if c == 1]
        if len(big) != 1 or len(sing) != 1:
            continue
        owner = defaultdict(float)
        for r, k in zip(idx, lab):
            if k != big[0]:
                continue
            for i, s in res.score(ttl[fcol].values[r], ttl[lcol].values[r]):
                owner[i] += s ** 8
        if not owner:
            continue
        who = max(owner, key=owner.get)
        if owner[who] / sum(owner.values()) < 0.60:      # ambiguous owner, skip
            continue
        claims[names[lab.index(sing[0])]].add(who)
    return {a: next(iter(c)) for a, c in claims.items() if len(c) == 1}


def link_by_alias(alias_map, df, fcol, lcol, already_linked):
    """Claim rows whose name is a known alias and that nothing else has claimed."""
    f = np.array([norm(x) for x in df[fcol]])
    l = np.array([norm(x) for x in df[lcol]])
    ri, ci, wt = [], [], []
    for r in range(len(df)):
        if already_linked[r]:
            continue
        who = alias_map.get((f[r], l[r]))
        if who is not None:
            ri.append(r); ci.append(who); wt.append(ALIAS_WEIGHT)
    return np.array(ri, int), np.array(ci, int), np.array(wt, float)
