"""Tiny in-memory stand-in for the subset of PyMongo used in this project.

Only used automatically when no MongoDB server (or PyMongo) is available, so
the project can still be demonstrated. With a real MongoDB server running,
the real PyMongo client is used and this file is never touched.
"""
import copy


def _match(doc, flt):
    for k, cond in flt.items():
        v = doc.get(k)
        if isinstance(cond, dict):
            for op, x in cond.items():
                if v is None and op != "$ne":
                    return False
                if op == "$gt" and not v > x: return False
                if op == "$gte" and not v >= x: return False
                if op == "$lt" and not v < x: return False
                if op == "$lte" and not v <= x: return False
                if op == "$ne" and v == x: return False
                if op == "$in" and v not in x: return False
        elif v != cond:
            return False
    return True


class Collection:
    def __init__(self):
        self._docs, self._next, self.indexes = [], 1, []

    def insert_many(self, docs):
        for d in docs:
            d = dict(d); d["_id"] = self._next; self._next += 1
            self._docs.append(d)

    def create_index(self, keys, **kw):
        name = "_".join(f"{k}_{d}" for k, d in (keys if isinstance(keys, list) else [(keys, 1)]))
        self.indexes.append(name); return name

    def delete_many(self, flt):
        self._docs = [d for d in self._docs if not _match(d, flt)]

    def count_documents(self, flt):
        return sum(_match(d, flt) for d in self._docs)

    def find(self, flt=None, projection=None):
        out = []
        for d in self._docs:
            if _match(d, flt or {}):
                d = copy.copy(d)
                if projection:
                    if all(v == 0 for v in projection.values()):
                        for k in projection: d.pop(k, None)
                    else:
                        d = {k: d.get(k) for k in projection if projection[k]}
                out.append(d)
        return out

    def find_one(self, flt=None, projection=None):
        r = self.find(flt, projection)
        return r[0] if r else None

    def aggregate(self, pipeline):
        docs = [copy.copy(d) for d in self._docs]
        for st in pipeline:
            (op, arg), = st.items()
            if op == "$match":
                docs = [d for d in docs if _match(d, arg)]
            elif op == "$group":
                groups = {}
                for d in docs:
                    groups.setdefault(d.get(arg["_id"][1:]), []).append(d)
                res = []
                for key, g in groups.items():
                    r = {"_id": key}
                    for name, spec in arg.items():
                        if name == "_id": continue
                        (acc, f), = spec.items()
                        vals = [1] * len(g) if f == 1 else [x.get(f[1:]) for x in g if x.get(f[1:]) is not None]
                        r[name] = (sum(vals) / len(vals) if acc == "$avg" else sum(vals) if acc == "$sum"
                                   else min(vals) if acc == "$min" else max(vals))
                    res.append(r)
                docs = res
            elif op == "$sort":
                for k, direction in reversed(list(arg.items())):
                    docs.sort(key=lambda d: d[k], reverse=direction < 0)
            elif op == "$limit":
                docs = docs[:arg]
        return docs


class Database(dict):
    def __missing__(self, k):
        self[k] = Collection(); return self[k]
