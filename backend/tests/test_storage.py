"""Tests for the audit log store (SQLite and Firestore backends). Run from backend/: python -m pytest"""

import json
import sqlite3
import types
from datetime import datetime, timedelta, timezone

import pytest

from app.config import get_settings
from app.insights import store


@pytest.fixture
def sqlite_store(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "storage_backend", "sqlite")
    monkeypatch.setattr(s, "interactions_db", tmp_path / "interactions.db")
    monkeypatch.setattr(s, "interaction_retention_days", 90)
    store.get_backend.cache_clear()
    yield store
    store.get_backend.cache_clear()


def _record(st, question="What is KYC?", outcome="answered", **details):
    return st.record_interaction(session_id="s1", question=question, answer="KYC is … [1]", outcome=outcome,
                                 sources=[{"doc_id": "d", "filename": "kyc.pdf", "page": 1}], top_score=0.6,
                                 latency_ms=1200.0, details=details)


def test_sqlite_keeps_audit_fields_and_feedback(sqlite_store):
    mid = _record(sqlite_store, llm_model="gpt-oss-120b", prompt_version="abc123",
                  timings_ms={"retrieval": 12.0, "generate": 900.0}, decision="answer", grounded=True)
    assert sqlite_store.set_feedback(mid, "down", "wrong page")
    assert not sqlite_store.set_feedback("missing", "up", None)
    [r] = sqlite_store.recent()
    assert r["id"] == mid and r["sources"][0]["filename"] == "kyc.pdf"
    assert r["llm_model"] == "gpt-oss-120b" and r["prompt_version"] == "abc123"
    assert r["timings_ms"]["generate"] == 900.0 and r["grounded"] is True
    assert r["rating"] == "down" and r["comment"] == "wrong page" and r["feedback_at"]


def test_sqlite_upgrades_old_database(tmp_path, sqlite_store):
    path = get_settings().interactions_db
    conn = sqlite3.connect(path)  # an interactions table from before the audit fields existed
    conn.execute("CREATE TABLE interactions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, session_id TEXT, "
                 "question TEXT NOT NULL, answer TEXT NOT NULL, outcome TEXT NOT NULL, sources TEXT NOT NULL, "
                 "top_score REAL, latency_ms REAL, rating TEXT, comment TEXT)")
    conn.execute("INSERT INTO interactions VALUES ('old', ?, 's', 'q', 'a', 'answered', '[]', NULL, 5, NULL, NULL)",
                 (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    conn.close()
    _record(sqlite_store, prompt_version="v2")
    records = sqlite_store.recent()
    assert {r["id"] for r in records} >= {"old"} and len(records) == 2
    assert next(r for r in records if r["id"] != "old")["prompt_version"] == "v2"


def test_sqlite_retention_deletes_old_records(sqlite_store):
    backend = sqlite_store.get_backend()
    old = {"id": "old", "created_at": datetime.now(timezone.utc) - timedelta(days=200), "session_id": "s",
           "question": "q", "answer": "a", "outcome": "answered", "sources": []}
    backend.add(old)
    _record(sqlite_store)  # adding a new record purges records older than the retention period
    assert [r["id"] for r in sqlite_store.recent()] != ["old"] and len(sqlite_store.recent()) == 1


def test_insights_and_eval_candidates(sqlite_store):
    _record(sqlite_store, question="Home loan rate?", outcome="not_found")
    _record(sqlite_store, question="home loan  rate?", outcome="not_found")
    mid = _record(sqlite_store, question="What is KYC?")
    sqlite_store.set_feedback(mid, "down", None)
    ins = sqlite_store.insights()
    assert ins["total"] == 3 and ins["by_outcome"] == {"not_found": 2, "answered": 1}
    assert ins["unanswered"][0]["question"] == "home loan rate?" and ins["unanswered"][0]["count"] == 2
    assert ins["feedback"] == {"up": 0, "down": 1} and ins["downvoted"][0]["question"] == "What is KYC?"
    assert ins["storage"] == "sqlite" and ins["avg_latency_ms"] == 1200
    candidates = sqlite_store.eval_candidates()
    assert {c["question"] for c in candidates} == {"home loan rate?", "what is kyc?"}
    assert all(c["type"] == "needs_review" for c in candidates)


# --------------------------------------------------------------------- Firestore (simulated)

class FakeSnap:
    def __init__(self, doc_id, data):
        self.id, self._data = doc_id, data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data)


class FakeDocRef:
    def __init__(self, docs, doc_id):
        self.docs, self.doc_id = docs, doc_id

    def set(self, data):
        self.docs[self.doc_id] = dict(data)

    def get(self):
        return FakeSnap(self.doc_id, self.docs.get(self.doc_id))

    def update(self, data):
        self.docs[self.doc_id].update(data)


class FakeQuery:
    def __init__(self, docs, filters=(), order=None, limit_n=None):
        self.docs, self.filters, self.order, self.limit_n = docs, list(filters), order, limit_n

    def where(self, filter):  # noqa: A002 - mirrors the Firestore API
        return FakeQuery(self.docs, [*self.filters, filter], self.order, self.limit_n)

    def order_by(self, field, direction):
        return FakeQuery(self.docs, self.filters, (field, direction), self.limit_n)

    def limit(self, n):
        return FakeQuery(self.docs, self.filters, self.order, n)

    def stream(self):
        items = list(self.docs.items())
        for f in self.filters:
            assert f.op_string == ">="
            items = [(k, v) for k, v in items if v[f.field_path] >= f.value]
        if self.order:
            field, direction = self.order
            items.sort(key=lambda kv: kv[1][field], reverse=direction == "DESCENDING")
        for k, v in items[: self.limit_n]:
            yield FakeSnap(k, v)


class FakeCollection(FakeQuery):
    def __init__(self):
        super().__init__({})

    def document(self, doc_id):
        return FakeDocRef(self.docs, doc_id)


@pytest.fixture
def firestore_store(monkeypatch):
    backend = store.FirestoreBackend.__new__(store.FirestoreBackend)
    backend.collection = FakeCollection()
    backend.firestore = types.SimpleNamespace(Query=types.SimpleNamespace(DESCENDING="DESCENDING"))
    monkeypatch.setattr(get_settings(), "interaction_retention_days", 90)
    monkeypatch.setattr(store, "get_backend", lambda: backend)
    return backend


def test_firestore_backend_stores_expiry_feedback_and_queries(firestore_store):
    first = _record(store, question="q1", llm_model="m")
    second = _record(store, question="q2")
    doc = firestore_store.collection.docs[first]
    assert isinstance(doc["created_at"], datetime) and doc["expire_at"] - doc["created_at"] == timedelta(days=90)
    assert doc["llm_model"] == "m" and "id" not in doc
    assert store.set_feedback(second, "up", None) and not store.set_feedback("missing", "up", None)
    records = store.recent()
    assert [r["question"] for r in records] == ["q2", "q1"]  # newest first
    assert records[0]["rating"] == "up" and isinstance(records[0]["created_at"], str)
    later = datetime.now(timezone.utc) + timedelta(seconds=5)
    assert store.recent(since=later) == []
    json.dumps(store.insights())  # insights output is JSON-serialisable with Firestore records
