from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os, io, csv, logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
from collections import Counter

from data_loader import store, CLASS_MAP_INV, EVIDENCE_KEYS


class BulkTagPayload(BaseModel):
    candidate_ids: List[str]
    tag: str

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ.get('MONGO_URL', 'memory://local')
if mongo_url.startswith('memory://'):
    from memory_mongo import MemoryClient, MemoryDB
    client = MemoryClient()
    db = MemoryDB()
else:
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'hencheck')]

app = FastAPI(title="DelDOT DMV Casework API")
api_router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


class DecisionUpdate(BaseModel):
    reviewer_status: str
    reviewer_notes: Optional[str] = ""
    reviewer_id: Optional[str] = "analyst_demo"


class TagPayload(BaseModel):
    tag: str


# ---- Reviewer state (persisted in Mongo) ----
async def _get_state(cid: str) -> dict:
    doc = await db.reviewer_state.find_one({"candidate_id": cid}, {"_id": 0})
    if not doc:
        return {"reviewer_status": "unreviewed", "reviewer_notes": "", "reviewer_tags": [], "last_updated": None}
    return doc


async def _merge_state(c: dict) -> dict:
    st = await _get_state(c["candidate_id"])
    return {**c, **{k: v for k, v in st.items() if k != "candidate_id"}}


def _priority_band(p: float) -> str:
    if p >= 0.85: return "critical"
    if p >= 0.70: return "high"
    if p >= 0.40: return "medium"
    return "low"


def _phase_view(c: dict, phase: str) -> dict:
    """Return a case dict projecting the given phase's predictions to top-level fields."""
    if phase not in ("T0", "T1"):
        return c
    p = c["t0"] if phase == "T0" else c["t1"]
    if not p:
        return c
    out = {**c, "phase": phase, **p}
    return out


# ---- Endpoints ----
@api_router.get("/")
async def root():
    return {"service": "DelDOT DMV Casework API", "status": "operational", "loaded": store.loaded}


@api_router.get("/data/status")
async def data_status():
    v = store.validation
    class_labels = {
        "insufficient_evidence": "insufficient_information",
        "review_warranted": "review_warranted",
        "review_not_warranted": "review_not_warranted",
    }
    def _rename(d):
        return {class_labels.get(k, k): v for k, v in d.items()}
    return {
        "total_prediction_rows": v["total_prediction_rows"],
        "expected_rows": v["expected_rows"],
        "unique_candidates": v["unique_candidates"],
        "t0_records": v["t0_records"],
        "t1_records": v["t1_records"],
        "evidence_records": v["evidence_records"],
        "by_class_t1": _rename(v["by_class_t1"]),
        "by_class_t0": _rename(v["by_class_t0"]),
        "warnings": v["warnings"],
        "model_version": store.metadata.get("model_version", ""),
        "prediction_timestamp": store.metadata.get("prediction_timestamp", ""),
    }


@api_router.get("/stats")
async def get_stats(phase: str = "T1"):
    total = len(store.cases)
    by_class = Counter()
    priority = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    priority_bins = [0] * 10
    confidence_bins = [0] * 10
    total_conf = 0.0
    changes = {"warranted_to_not": 0, "not_to_warranted": 0, "ii_to_warranted": 0, "ii_to_not": 0, "to_ii": 0, "no_change": 0}
    evidence_coverage = {k: 0 for k in EVIDENCE_KEYS}
    ev_strength = {"Strong": 0, "Moderate": 0, "Weak": 0}
    state_counts = Counter()

    for c in store.cases.values():
        p = c["t0"] if phase == "T0" else c["t1"]
        if not p:
            continue
        by_class[p["predicted_class"]] += 1
        priority[_priority_band(p["review_priority"])] += 1
        priority_bins[min(9, int(p["review_priority"] * 10))] += 1
        conf = p["confidence"]
        confidence_bins[min(9, int(conf * 10))] += 1
        total_conf += conf
        ev_strength[c["evidence_strength"]] += 1
        state_counts[c["primary_state"]] += 1
        for k in EVIDENCE_KEYS:
            if c["evidence_counts"].get(k, 0) > 0:
                evidence_coverage[k] += 1
        prev = c["t0"]["predicted_class"] if c["t0"] else p["predicted_class"]
        cur = p["predicted_class"]
        if prev == cur:
            changes["no_change"] += 1
        elif prev == "review_warranted" and cur == "review_not_warranted":
            changes["warranted_to_not"] += 1
        elif prev == "review_not_warranted" and cur == "review_warranted":
            changes["not_to_warranted"] += 1
        elif prev == "insufficient_information" and cur == "review_warranted":
            changes["ii_to_warranted"] += 1
        elif prev == "insufficient_information" and cur == "review_not_warranted":
            changes["ii_to_not"] += 1
        elif cur == "insufficient_information":
            changes["to_ii"] += 1

    top_states = state_counts.most_common(8)
    return {
        "total": total,
        "phase": phase,
        "by_class": {k: by_class.get(k, 0) for k in ["review_warranted", "review_not_warranted", "insufficient_information"]},
        "priority": priority,
        "priority_bins": priority_bins,
        "confidence_bins": confidence_bins,
        "avg_confidence": round(total_conf / total, 3) if total else 0,
        "changes": changes,
        "evidence_coverage": evidence_coverage,
        "evidence_strength": ev_strength,
        "top_states": [{"state": s, "count": n} for s, n in top_states],
    }


@api_router.get("/cases")
async def list_cases(
    q: Optional[str] = None,
    predicted_class: Optional[str] = None,
    reviewer_status: Optional[str] = None,
    priority_band: Optional[str] = None,
    phase: Optional[str] = "T1",  # phase view (which prediction to project)
    evidence_strength: Optional[str] = None,
    tag: Optional[str] = None,
    sort_by: str = "review_priority",
    sort_dir: str = "desc",
    skip: int = 0,
    limit: int = 50,
):
    phase = phase if phase in ("T0", "T1") else "T1"
    q_lower = q.lower() if q else None

    # Load reviewer state map efficiently
    states_docs = await db.reviewer_state.find({}, {"_id": 0}).to_list(20000)
    state_map = {d["candidate_id"]: d for d in states_docs}

    filtered = []
    for c in store.cases.values():
        p = c["t0"] if phase == "T0" else c["t1"]
        if not p:
            continue
        if q_lower and q_lower not in c["candidate_id"].lower():
            continue
        if predicted_class and predicted_class != "all" and p["predicted_class"] != predicted_class:
            continue
        band = _priority_band(p["review_priority"])
        if priority_band and priority_band != "all" and band != priority_band:
            continue
        if evidence_strength and evidence_strength != "all" and c["evidence_strength"] != evidence_strength:
            continue
        st = state_map.get(c["candidate_id"], {})
        rs = st.get("reviewer_status", "unreviewed")
        if reviewer_status and reviewer_status != "all" and rs != reviewer_status:
            continue
        tags = st.get("reviewer_tags", [])
        if tag and tag != "all" and tag not in tags:
            continue
        # Merge phase view + state (return lightweight version)
        item = {
            "candidate_id": c["candidate_id"],
            "phase": phase,
            "predicted_class": p["predicted_class"],
            "prob_review_warranted": p["prob_review_warranted"],
            "prob_review_not_warranted": p["prob_review_not_warranted"],
            "prob_insufficient_information": p["prob_insufficient_information"],
            "review_priority": p["review_priority"],
            "confidence": p["confidence"],
            "evidence_strength": c["evidence_strength"],
            "evidence_recency": c["evidence_recency"],
            "agreement": c["agreement"],
            "new_evidence_count": c["new_evidence_count"],
            "primary_state": c["primary_state"],
            "reviewer_status": rs,
            "reviewer_tags": tags,
            "last_updated": st.get("last_updated"),
        }
        filtered.append(item)

    # Sort
    reverse = sort_dir == "desc"
    def _key(x):
        v = x.get(sort_by)
        return (v is None, v)
    filtered.sort(key=_key, reverse=reverse)

    total = len(filtered)
    page = filtered[skip:skip + limit] if limit else filtered
    return {"cases": page, "total": total, "skip": skip, "limit": limit}


@api_router.get("/cases/{candidate_id}")
async def get_case(candidate_id: str):
    c = store.cases.get(candidate_id)
    if not c:
        raise HTTPException(404, "Case not found")
    state = await _get_state(candidate_id)
    # Group evidence by kind
    grouped = {k: [] for k in EVIDENCE_KEYS}
    for r in store.evidence.get(candidate_id, []):
        grouped[r["kind"]].append(r)
    for k in grouped:
        grouped[k].sort(key=lambda x: x["date"] or "")
    out = {
        **c,
        "evidence": grouped,
        **{kk: vv for kk, vv in state.items() if kk != "candidate_id"},
    }
    return out


@api_router.patch("/cases/{candidate_id}")
async def update_case(candidate_id: str, update: DecisionUpdate):
    c = store.cases.get(candidate_id)
    if not c:
        raise HTTPException(404, "Case not found")
    prev = await _get_state(candidate_id)
    now = datetime.now(timezone.utc).isoformat()
    new_state = {
        "candidate_id": candidate_id,
        "reviewer_status": update.reviewer_status,
        "reviewer_notes": update.reviewer_notes or "",
        "reviewer_tags": prev.get("reviewer_tags", []),
        "last_updated": now,
    }
    await db.reviewer_state.update_one({"candidate_id": candidate_id}, {"$set": new_state}, upsert=True)
    # Audit
    audit = {
        "id": f"{candidate_id}-{datetime.now(timezone.utc).timestamp()}",
        "candidate_id": candidate_id,
        "phase": c["phase"],
        "model_version": store.metadata.get("model_version", ""),
        "action": update.reviewer_status,
        "reviewer_id": update.reviewer_id or "analyst_demo",
        "from_status": prev.get("reviewer_status", "unreviewed"),
        "to_status": update.reviewer_status,
        "notes": update.reviewer_notes or "",
        "timestamp": now,
        "snapshot": {
            "predicted_class": c["predicted_class"],
            "review_priority": c["review_priority"],
            "confidence": c["confidence"],
            "probs": {
                "review_warranted": c["prob_review_warranted"],
                "review_not_warranted": c["prob_review_not_warranted"],
                "insufficient_information": c["prob_insufficient_information"],
            },
            "evidence_strength": c["evidence_strength"],
            "evidence_summary": c["evidence_counts"],
        },
    }
    await db.audit_log.insert_one(audit)
    return {**c, **new_state}


# ---- Reviewer Tags ----
@api_router.post("/cases/{candidate_id}/tags")
async def add_tag(candidate_id: str, payload: TagPayload):
    if candidate_id not in store.cases:
        raise HTTPException(404, "Case not found")
    tag = payload.tag.strip()
    if not tag:
        raise HTTPException(400, "Empty tag")
    now = datetime.now(timezone.utc).isoformat()
    await db.reviewer_state.update_one(
        {"candidate_id": candidate_id},
        {"$addToSet": {"reviewer_tags": tag}, "$set": {"last_updated": now, "candidate_id": candidate_id}},
        upsert=True,
    )
    st = await _get_state(candidate_id)
    return st


@api_router.delete("/cases/{candidate_id}/tags/{tag}")
async def remove_tag(candidate_id: str, tag: str):
    now = datetime.now(timezone.utc).isoformat()
    await db.reviewer_state.update_one(
        {"candidate_id": candidate_id},
        {"$pull": {"reviewer_tags": tag}, "$set": {"last_updated": now}},
    )
    return await _get_state(candidate_id)


@api_router.get("/tags")
async def list_tags():
    pipeline = [
        {"$unwind": "$reviewer_tags"},
        {"$group": {"_id": "$reviewer_tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    result = await db.reviewer_state.aggregate(pipeline).to_list(500)
    return {"tags": [{"tag": r["_id"], "count": r["count"]} for r in result]}


# ---- Bulk Tagging ----
@api_router.post("/tags/bulk")
async def bulk_add_tag(payload: BulkTagPayload):
    tag = payload.tag.strip()
    if not tag:
        raise HTTPException(400, "Empty tag")
    now = datetime.now(timezone.utc).isoformat()
    valid = [cid for cid in payload.candidate_ids if cid in store.cases]
    if not valid:
        raise HTTPException(400, "No valid candidates")
    for cid in valid:
        await db.reviewer_state.update_one(
            {"candidate_id": cid},
            {"$addToSet": {"reviewer_tags": tag}, "$set": {"last_updated": now, "candidate_id": cid}},
            upsert=True,
        )
    await db.audit_log.insert_one({
        "id": f"BULK-{datetime.now(timezone.utc).timestamp()}",
        "candidate_id": "BULK",
        "phase": "-",
        "model_version": store.metadata.get("model_version", ""),
        "action": f"bulk_tag_add:{tag}",
        "reviewer_id": "analyst_demo",
        "from_status": "-",
        "to_status": f"tagged_{len(valid)}",
        "notes": f"Bulk added tag '{tag}' to {len(valid)} candidates",
        "timestamp": now,
        "snapshot": {"tag": tag, "count": len(valid)},
    })
    return {"tagged": len(valid), "tag": tag}


# ---- Evidence Search ----
STATE_NAME_TO_CODE = {
    "delaware": "DE", "pennsylvania": "PA", "maryland": "MD", "new jersey": "NJ",
    "virginia": "VA", "new york": "NY", "ohio": "OH", "west virginia": "WV",
    "north carolina": "NC", "florida": "FL", "texas": "TX", "district of columbia": "DC",
}


@api_router.get("/evidence/search")
async def evidence_search(q: str = "", state: Optional[str] = None, source: Optional[str] = None, limit: int = 50):
    ql = q.strip().lower() if q else ""
    # Expand common state names → codes so "Delaware" matches "DE"
    ql_alt = STATE_NAME_TO_CODE.get(ql)
    matched = []
    seen_candidates = set()
    for r in store.evidence_flat:
        if source and source != "all" and r["kind"] != source:
            continue
        if state and state != "all" and r["state"] != state:
            continue
        if ql:
            hay = f"{r['candidate_id']} {r['source_record_id']} {r['vehicle_ref']} {r['state']} {r['event_type']}".lower()
            if ql not in hay and (not ql_alt or ql_alt.lower() not in hay):
                continue
        matched.append(r)
        seen_candidates.add(r["candidate_id"])
        if len(matched) >= limit:
            break
    return {"records": matched, "total_matches": len(matched), "candidates_matched": len(seen_candidates)}


# ---- Feature Importance ----
@api_router.get("/cases/{candidate_id}/feature-importance")
async def feature_importance(candidate_id: str):
    if candidate_id not in store.cases:
        raise HTTPException(404, "Case not found")
    if not store.feature_importance:
        return {"features": [], "note": "Feature file not loaded"}
    c = store.cases[candidate_id]
    cls = c["predicted_class"]
    top = store.feature_importance["top_by_class"].get(cls, [])
    values = store.feature_importance["case_values"].get(candidate_id, {})
    labels = store.feature_importance["feature_labels"]
    out = []
    for t in top:
        f = t["feature"]
        v = values.get(f, 0)
        z = 0.0
        stats = store.feature_importance["feature_stats"].get(f, {})
        std = stats.get("overall_std", 1) or 1
        z = round((v - t["overall_mean"]) / std, 2)
        # Contribution direction: does this case's value push toward the class mean vs overall?
        pushes_toward = (v > t["overall_mean"] and t["direction"] == "higher") or (v < t["overall_mean"] and t["direction"] == "lower")
        out.append({
            "feature": f,
            "label": labels.get(f, f),
            "case_value": round(float(v), 3),
            "class_mean": t["class_mean"],
            "overall_mean": t["overall_mean"],
            "class_direction": t["direction"],
            "effect_size": t["effect_size"],
            "case_z_score": z,
            "pushes_toward_class": pushes_toward,
        })
    return {"predicted_class": cls, "features": out}


# ---- Case Comparison ----
@api_router.get("/cases/compare/{id_a}/{id_b}")
async def compare(id_a: str, id_b: str):
    a = store.cases.get(id_a)
    b = store.cases.get(id_b)
    if not a or not b:
        raise HTTPException(404, "One or both cases not found")
    async def full(cid, c):
        state = await _get_state(cid)
        grouped = {k: [] for k in EVIDENCE_KEYS}
        for r in store.evidence.get(cid, []):
            grouped[r["kind"]].append(r)
        return {**c, "evidence": grouped, **{kk: vv for kk, vv in state.items() if kk != "candidate_id"}}
    return {"a": await full(id_a, a), "b": await full(id_b, b)}


# ---- Audit ----
@api_router.get("/audit")
async def list_audit(limit: int = 300):
    docs = await db.audit_log.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
    return {"entries": docs}


@api_router.get("/audit/{entry_id}")
async def get_audit(entry_id: str):
    d = await db.audit_log.find_one({"id": entry_id}, {"_id": 0})
    if not d:
        raise HTTPException(404, "Not found")
    return d


# ---- Model Performance ----
@api_router.get("/model/performance")
async def model_perf():
    return store.metrics


# ---- CSV export in original schema ----
@api_router.get("/cases/export/csv")
async def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_record_id", "phase", "predicted_class",
                     "p_review_warranted", "p_review_not_warranted", "p_insufficient_evidence",
                     "review_priority"])
    for c in store.cases.values():
        for phase_key, phase in [("t0", "T0"), ("t1", "T1")]:
            p = c[phase_key]
            if not p:
                continue
            writer.writerow([
                c["candidate_id"], phase,
                CLASS_MAP_INV.get(p["predicted_class"], p["predicted_class"]),
                p["prob_review_warranted"], p["prob_review_not_warranted"],
                p["prob_insufficient_information"], p["review_priority"],
            ])
    output.seek(0)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=case_predictions_{ts}.csv"})


@api_router.post("/data/reload")
async def reload_data():
    store.cases.clear()
    store.evidence.clear()
    store.load()
    return {"status": "reloaded", "candidates": len(store.cases)}


app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"], allow_headers=["*"])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@app.on_event("startup")
async def startup():
    logger.info("Loading real dataset from case_predictions.csv…")
    store.load()
    logger.info(f"Loaded {len(store.cases)} candidates, {sum(len(v) for v in store.evidence.values())} evidence records")
    # Clear old mock cases collection to avoid confusion (kept reviewer_state + audit_log)
    await db.cases.drop()


@app.on_event("shutdown")
async def shutdown():
    client.close()
