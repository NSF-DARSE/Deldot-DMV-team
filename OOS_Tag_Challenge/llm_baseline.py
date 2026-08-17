"""Does a frontier LLM classify these cases better than the ordinal model?

Gives Claude the same evidence the model sees, asks for the same three-way verdict with
probabilities, and scores it against the same held-out labels. The point is to answer the
"why didn't you use AI for this?" question with a measurement rather than an opinion.
"""
import json, subprocess, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from pipeline import BASE

BATCH, MODEL = 15, "claude-haiku-4-5-20251001"
CLASSES = ["review_not_warranted", "insufficient_evidence", "review_warranted"]

audit = pd.read_csv("case_audit.csv")
lab = pd.read_csv(BASE + "Development_Labels/Development_Labels.csv")
a0 = audit[audit.phase == "T0"].set_index("candidate_record_id")
ids = list(lab.candidate_record_id)
y = np.array([CLASSES.index(v) for v in lab.label_t0])

def brief(cid):
    r = a0.loc[cid]
    feeds = "; ".join(
        f"{f}: {r[f'latest_{f}_state']} on {r[f'latest_{f}_date']}"
        for f in ("address", "licence", "title", "external")
        if isinstance(r[f"latest_{f}_state"], str))
    return (f"case {cid} | observed tag: {r.observed_tag_state} | "
            f"linked records: {r.evidence_records_linked} | "
            f"share of recent evidence pointing to DE: {r.de_share_recent} "
            f"(all time {r.de_share_all_time}) | "
            f"days since newest record: {r.days_since_newest_record} | newest per feed - {feeds}")

PROMPT = """You are triaging Delaware motor-vehicle records. For each case decide whether a staff member should review it.

review_warranted    - recent evidence indicates the person is now operating in Delaware, so the vehicle may need Delaware registration
review_not_warranted - recent evidence indicates they are established in another state
insufficient_evidence - the records conflict or are too thin to decide

Base rates: roughly 30% warranted, 35% not warranted, 35% insufficient.

Return ONLY a JSON array, one object per case, in the same order:
[{"id":"...","p_not_warranted":0.0,"p_insufficient":0.0,"p_warranted":0.0}]
Probabilities must sum to 1. No prose, no code fences.

CASES:
"""

def ask(chunk):
    body = PROMPT + "\n".join(brief(c) for c in chunk)
    out = subprocess.run(["claude", "-p", body, "--model", MODEL],
                         capture_output=True, text=True, timeout=300).stdout.strip()
    out = out.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(out)

P, done = np.full((len(ids), 3), np.nan), 0
for i in range(0, len(ids), BATCH):
    chunk = ids[i:i + BATCH]
    try:
        for j, rec in enumerate(ask(chunk)):
            p = np.array([rec["p_not_warranted"], rec["p_insufficient"], rec["p_warranted"]], float)
            P[i + j] = p / p.sum()
        done += len(chunk)
    except Exception as e:
        print(f"  batch {i} failed: {type(e).__name__}", file=sys.stderr)
    print(f"  {done}/{len(ids)} scored", flush=True)

ok = ~np.isnan(P[:, 0])
P, yy = np.clip(P[ok], 1e-6, 1), y[ok]
P = P / P.sum(1, keepdims=True)
print(f"\nLLM classifier on {ok.sum()} held-out cases (model: {MODEL})")
print(f"  log loss  {log_loss(yy, P, labels=[0,1,2]):.4f}")
print(f"  accuracy  {accuracy_score(yy, P.argmax(1)):.4f}")
print(f"  macro AUC {np.mean([roc_auc_score((yy==k).astype(int), P[:,k]) for k in range(3)]):.4f}")
print(f"  predicted mix {np.bincount(P.argmax(1), minlength=3)} vs true {np.bincount(yy)}")
print("\nordinal model on the same task: log loss 0.906, accuracy 0.558, macro AUC 0.740")
np.save("llm_baseline_probs.npy", P)
