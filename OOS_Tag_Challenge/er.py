"""Entity resolution helpers: map noisy source-record names onto candidate names.

The synthetic data corrupts names by truncation, character substitution,
insertion/deletion, adjacent transposition and stray whitespace. Matching uses a
SymSpell-style delete neighbourhood (handles edit distance <= 2) plus an explicit
prefix index (handles truncation to an initial).
"""
import re
from collections import defaultdict

PFX = re.compile(r'^SYN(GIV|FAM|NAME)-', re.I)
NON = re.compile(r'[^A-Z0-9]')


def norm(s):
    s = str(s).upper().strip()
    return NON.sub('', PFX.sub('', s))


def dl(a, b, maxd=2):
    """Damerau-Levenshtein distance, capped at maxd (returns maxd+1 if beyond)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > maxd:
        return maxd + 1
    prev2, prev = None, list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [maxd + 1] * lb
        for j in range(max(1, i - maxd), min(lb, i + maxd) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
        if min(cur) > maxd:
            return maxd + 1
        prev2, prev = prev, cur
    return prev[lb]


def sim(a, b):
    """0..1 similarity. Truncation is scored by retained-length fraction."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    lo, hi = (len(a), len(b)) if len(a) < len(b) else (len(b), len(a))
    if a.startswith(b) or b.startswith(a):
        return 0.80 + 0.19 * (lo / hi)
    d = dl(a, b, 2)
    if d <= 2:
        return max(0.0, 1.0 - d / hi) * 0.97
    return 0.0


def _deletes(s, k=2):
    out = {s}
    frontier = {s}
    for _ in range(k):
        nxt = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1:])
        out |= nxt
        frontier = nxt
    return out


class NameMatcher:
    """Fuzzy lookup from an arbitrary name string to a fixed target vocabulary."""

    def __init__(self, targets):
        self.targets = set(targets)
        self.del_idx = defaultdict(set)
        self.pfx_idx = defaultdict(set)
        for t in self.targets:
            for d in _deletes(t, 2):
                self.del_idx[d].add(t)
            for k in range(1, len(t) + 1):
                self.pfx_idx[t[:k]].add(t)

    def candidates(self, s):
        c = set()
        for d in _deletes(s, 2):
            c |= self.del_idx.get(d, set())
        c |= self.pfx_idx.get(s, set())          # target extends source (truncation)
        for k in range(1, len(s)):                # target is a prefix of source
            if s[:k] in self.targets:
                c.add(s[:k])
        return c

    def match(self, s, min_sim=0.55, top=12):
        scored = [(t, sim(s, t)) for t in self.candidates(s)]
        scored = [x for x in scored if x[1] >= min_sim]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top]
