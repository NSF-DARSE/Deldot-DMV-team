"""Minimal async Mongo stand-in for local dashboard runs without MongoDB."""
from copy import deepcopy


class _Cursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, key, direction=-1):
        reverse = direction == -1
        self._docs = sorted(self._docs, key=lambda d: d.get(key) or "", reverse=reverse)
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, n=None):
        docs = self._docs if n is None else self._docs[:n]
        return [deepcopy(d) for d in docs]


class _Collection:
    def __init__(self):
        self._docs = []

    def _matches(self, doc, filt):
        return all(doc.get(k) == v for k, v in (filt or {}).items())

    def _project(self, doc, projection):
        if not projection:
            return deepcopy(doc)
        out = deepcopy(doc)
        if projection.get("_id") == 0:
            out.pop("_id", None)
        include = {k for k, v in projection.items() if v and k != "_id"}
        if include:
            out = {k: deepcopy(doc[k]) for k in include if k in doc}
            if projection.get("_id") != 0 and "_id" in doc:
                out["_id"] = doc["_id"]
        return out

    async def find_one(self, filt=None, projection=None):
        for d in self._docs:
            if self._matches(d, filt):
                return self._project(d, projection)
        return None

    def find(self, filt=None, projection=None):
        docs = [self._project(d, projection) for d in self._docs if self._matches(d, filt)]
        return _Cursor(docs)

    async def insert_one(self, doc):
        self._docs.append(deepcopy(doc))

    async def update_one(self, filt, update, upsert=False):
        for d in self._docs:
            if self._matches(d, filt):
                self._apply(d, update)
                return
        if upsert:
            new = dict(filt or {})
            self._apply(new, update)
            self._docs.append(new)

    def _apply(self, doc, update):
        if "$set" in update:
            doc.update(update["$set"])
        if "$addToSet" in update:
            for k, v in update["$addToSet"].items():
                arr = doc.setdefault(k, [])
                if v not in arr:
                    arr.append(v)
        if "$pull" in update:
            for k, v in update["$pull"].items():
                if k in doc and isinstance(doc[k], list):
                    doc[k] = [x for x in doc[k] if x != v]
        for k, v in update.items():
            if not k.startswith("$"):
                doc[k] = v

    def aggregate(self, pipeline):
        docs = [deepcopy(d) for d in self._docs]
        for stage in pipeline:
            if "$unwind" in stage:
                field = stage["$unwind"].lstrip("$")
                unwound = []
                for d in docs:
                    for item in d.get(field, []) or []:
                        nd = deepcopy(d)
                        nd[field] = item
                        unwound.append(nd)
                docs = unwound
            elif "$group" in stage:
                grouped = {}
                gid = stage["$group"]["_id"]
                key_field = gid[1:] if isinstance(gid, str) and gid.startswith("$") else gid
                for d in docs:
                    key = d.get(key_field) if isinstance(key_field, str) else gid
                    g = grouped.setdefault(key, {"_id": key, "count": 0})
                    count_spec = stage["$group"].get("count", {})
                    if "$sum" in count_spec:
                        g["count"] += count_spec["$sum"]
                docs = list(grouped.values())
            elif "$sort" in stage:
                for k, direction in reversed(list(stage["$sort"].items())):
                    docs.sort(key=lambda d: d.get(k) or 0, reverse=direction == -1)
        return _Cursor(docs)

    async def drop(self):
        self._docs.clear()


class MemoryDB:
    def __init__(self):
        self.reviewer_state = _Collection()
        self.audit_log = _Collection()
        self.cases = _Collection()


class MemoryClient:
    def close(self):
        pass
