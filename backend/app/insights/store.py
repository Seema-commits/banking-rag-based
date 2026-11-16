"""Interaction (audit) log and feedback, for the admin Insights tab and later review.

One record per question: when it was asked, the PII-masked question, the exact answer shown, its
sources and outcome, the agent's decisions, time per step, tokens, the model and prompt version
that produced it, and any thumbs-up/down feedback. Used to find knowledge gaps, review answers,
and export failures as eval candidates.

Two backends, chosen by STORAGE_BACKEND:
  - "sqlite"    (default): a local file (INTERACTIONS_DB). Fine for development; on Cloud Run the
                disk is wiped on restart.
  - "firestore": Google Cloud Firestore, which keeps records permanently. Records carry an
                `expire_at` field; a Firestore TTL policy deletes them after
                INTERACTION_RETENTION_DAYS (see ops/storage/setup_storage.sh).

Insights are computed in Python over the most recent INSIGHTS_WINDOW records, which keeps the
queries simple and cheap on both backends at demo scale.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from app.config import get_settings

INSIGHTS_WINDOW = 2000
_SQLITE_MAX_ROWS = 5000
_CORE = ("id", "created_at", "session_id", "question", "answer", "outcome", "sources", "top_score",
         "latency_ms", "rating", "comment")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------- SQLite backend

class SQLiteBackend:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS interactions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        session_id TEXT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        outcome TEXT NOT NULL,
        sources TEXT NOT NULL,
        top_score REAL,
        latency_ms REAL,
        rating TEXT,
        comment TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_interactions_created ON interactions(created_at);
    """

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.executescript(self._SCHEMA)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(interactions)")}
        if "details" not in columns:  # databases created before the audit fields existed
            conn.execute("ALTER TABLE interactions ADD COLUMN details TEXT")
        return conn

    def add(self, record: dict) -> None:
        details = {k: v for k, v in record.items() if k not in _CORE}
        retention = get_settings().interaction_retention_days
        with self.lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO interactions (id, created_at, session_id, question, answer, outcome, sources,"
                " top_score, latency_ms, rating, comment, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record["id"], record["created_at"].isoformat(), record.get("session_id"), record["question"],
                 record["answer"], record["outcome"], json.dumps(record.get("sources") or []),
                 record.get("top_score"), record.get("latency_ms"), record.get("rating"), record.get("comment"),
                 json.dumps(details, default=str)),
            )
            if retention > 0:
                cutoff = (_now() - timedelta(days=retention)).isoformat()
                conn.execute("DELETE FROM interactions WHERE created_at < ?", (cutoff,))
            conn.execute("DELETE FROM interactions WHERE id NOT IN "
                         "(SELECT id FROM interactions ORDER BY created_at DESC LIMIT ?)", (_SQLITE_MAX_ROWS,))

    def set_feedback(self, message_id: str, rating: str, comment: str | None) -> bool:
        with self.lock, self._connect() as conn:
            row = conn.execute("SELECT details FROM interactions WHERE id = ?", (message_id,)).fetchone()
            if row is None:
                return False
            details = json.loads(row["details"] or "{}")
            details["feedback_at"] = _now().isoformat()
            conn.execute("UPDATE interactions SET rating = ?, comment = ?, details = ? WHERE id = ?",
                         (rating, comment, json.dumps(details), message_id))
            return True

    def get(self, message_id: str) -> dict | None:
        with self.lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM interactions WHERE id = ?", (message_id,)).fetchone()
        return self._to_record(row) if row else None

    @staticmethod
    def _to_record(r) -> dict:
        record = {k: r[k] for k in _CORE}
        record["sources"] = json.loads(r["sources"] or "[]")
        record.update(json.loads(r["details"] or "{}"))
        return record

    def recent(self, limit: int, since: datetime | None = None) -> list[dict]:
        query = "SELECT * FROM interactions"
        params: list = []
        if since is not None:
            query += " WHERE created_at >= ?"
            params.append(since.isoformat())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._to_record(r) for r in rows]


# ------------------------------------------------------------------------- Firestore backend

class FirestoreBackend:
    def __init__(self, project: str | None, database: str, collection: str):
        from google.cloud import firestore  # imported here: only needed on Cloud Run

        self.firestore = firestore
        self.collection = firestore.Client(project=project or None, database=database).collection(collection)

    def add(self, record: dict) -> None:
        retention = get_settings().interaction_retention_days
        doc = {k: v for k, v in record.items() if k != "id"}
        if retention > 0:
            doc["expire_at"] = record["created_at"] + timedelta(days=retention)
        self.collection.document(record["id"]).set(doc)

    def set_feedback(self, message_id: str, rating: str, comment: str | None) -> bool:
        ref = self.collection.document(message_id)
        if not ref.get().exists:
            return False
        ref.update({"rating": rating, "comment": comment, "feedback_at": _now()})
        return True

    @staticmethod
    def _to_record(snap) -> dict:
        record = {"id": snap.id, **snap.to_dict()}
        for key in ("created_at", "expire_at", "feedback_at"):
            if isinstance(record.get(key), datetime):
                record[key] = record[key].isoformat()
        return record

    def get(self, message_id: str) -> dict | None:
        snap = self.collection.document(message_id).get()
        return self._to_record(snap) if snap.exists else None

    def recent(self, limit: int, since: datetime | None = None) -> list[dict]:
        from google.cloud.firestore_v1.base_query import FieldFilter

        query = self.collection
        if since is not None:
            query = query.where(filter=FieldFilter("created_at", ">=", since))
        query = query.order_by("created_at", direction=self.firestore.Query.DESCENDING).limit(limit)
        return [self._to_record(snap) for snap in query.stream()]


@lru_cache
def get_backend():
    settings = get_settings()
    if settings.storage_backend == "firestore":
        return FirestoreBackend(settings.firestore_project, settings.firestore_database,
                                settings.firestore_collection)
    return SQLiteBackend(settings.interactions_db)


# ------------------------------------------------------------------------------- public API

def record_interaction(*, session_id: str, question: str, answer: str, outcome: str, sources: list[dict],
                       top_score: float | None, latency_ms: float, details: dict | None = None) -> str:
    """Store one question and answer; returns its message id (used for feedback)."""
    message_id = uuid.uuid4().hex
    get_backend().add({
        "id": message_id, "created_at": _now(), "session_id": session_id, "question": question,
        "answer": answer, "outcome": outcome, "sources": sources, "top_score": top_score,
        "latency_ms": latency_ms, "rating": None, "comment": None, **(details or {}),
    })
    return message_id


def set_feedback(message_id: str, rating: str, comment: str | None) -> bool:
    return get_backend().set_feedback(message_id, rating, comment)


def get(message_id: str) -> dict | None:
    """One full interaction record, or None."""
    return get_backend().get(message_id)


def recent(limit: int = INSIGHTS_WINDOW, since: datetime | None = None) -> list[dict]:
    """Most recent interactions, newest first."""
    return get_backend().recent(limit, since)


def _normalise(question: str) -> str:
    return " ".join(question.lower().split())


def _unanswered(records: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for r in records:
        if r["outcome"] not in ("not_found", "unverified"):
            continue
        g = groups.setdefault(_normalise(r["question"]), {
            "question": _normalise(r["question"]), "count": 0, "last_seen": r["created_at"], "outcomes": set()})
        g["count"] += 1
        g["last_seen"] = max(g["last_seen"], r["created_at"])
        g["outcomes"].add(r["outcome"])
    # Most frequently asked first; ties broken by the most recent.
    rows = sorted(groups.values(), key=lambda g: (g["count"], g["last_seen"]), reverse=True)
    return [{**g, "outcomes": ",".join(sorted(g["outcomes"]))} for g in rows]


def insights(limit: int = 20) -> dict:
    """Usage, answer rate, feedback, knowledge gaps and downvoted answers (recent window)."""
    records = recent()
    total = len(records)
    by_outcome: dict[str, int] = {}
    for r in records:
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    downvoted = [{k: r.get(k) for k in ("id", "created_at", "question", "answer", "outcome", "comment")}
                 for r in records if r.get("rating") == "down"][:limit]
    return {
        "total": total,
        "window": INSIGHTS_WINDOW,
        "answer_rate": by_outcome.get("answered", 0) / total if total else None,
        "by_outcome": dict(sorted(by_outcome.items(), key=lambda kv: -kv[1])),
        "feedback": {"up": sum(r.get("rating") == "up" for r in records),
                     "down": sum(r.get("rating") == "down" for r in records)},
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "unanswered": _unanswered(records)[:limit],
        "downvoted": downvoted,
        "storage": get_settings().storage_backend,
    }


def eval_candidates() -> list[dict]:
    """Unanswered and downvoted questions, in the eval file's format, for a human to label."""
    records = recent()
    groups: dict[str, dict] = {}
    for r in records:
        if r["outcome"] in ("not_found", "unverified") or r.get("rating") == "down":
            g = groups.setdefault(_normalise(r["question"]), {
                "question": _normalise(r["question"]), "seen_outcome": r["outcome"], "rating": None, "count": 0})
            g["count"] += 1
            g["rating"] = g["rating"] or r.get("rating")
    rows = sorted(groups.values(), key=lambda g: -g["count"])
    return [
        {"id": f"review-{i:03d}", "type": "needs_review", "question": g["question"],
         "seen_outcome": g["seen_outcome"], "rating": g["rating"], "times_asked": g["count"],
         "note": "Set type to in_scope (add sources and answer_keywords) or out_of_scope, "
                 "then add to eval/questions.jsonl"}
        for i, g in enumerate(rows, start=1)
    ]
