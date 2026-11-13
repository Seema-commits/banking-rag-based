"""Interaction log and feedback (SQLite) for the admin Insights tab.

Stores each question (already PII-masked), the answer, outcome, sources and timings, plus any
thumbs-up/down feedback. Used to find unanswered questions and knowledge gaps, and to export
failures as candidates for the eval set. Keeps the most recent MAX_ROWS interactions.

On Cloud Run the disk is wiped on restart, so this history is temporary there.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.config import get_settings

MAX_ROWS = 5000
_lock = threading.Lock()

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


def _connect() -> sqlite3.Connection:
    path = get_settings().interactions_db
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def record_interaction(*, session_id: str, question: str, answer: str, outcome: str,
                       sources: list[dict], top_score: float | None, latency_ms: float) -> str:
    message_id = uuid.uuid4().hex
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO interactions (id, created_at, session_id, question, answer, outcome, sources,"
            " top_score, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (message_id, datetime.now(timezone.utc).isoformat(), session_id, question, answer, outcome,
             json.dumps(sources), top_score, latency_ms),
        )
        conn.execute(
            "DELETE FROM interactions WHERE id NOT IN "
            "(SELECT id FROM interactions ORDER BY created_at DESC LIMIT ?)", (MAX_ROWS,)
        )
    return message_id


def set_feedback(message_id: str, rating: str, comment: str | None) -> bool:
    with _lock, _connect() as conn:
        cur = conn.execute("UPDATE interactions SET rating = ?, comment = ? WHERE id = ?",
                           (rating, comment, message_id))
        return cur.rowcount > 0


def insights(limit: int = 20) -> dict:
    with _lock, _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        by_outcome = {r["outcome"]: r["n"] for r in conn.execute(
            "SELECT outcome, COUNT(*) AS n FROM interactions GROUP BY outcome ORDER BY n DESC")}
        ratings = {r["rating"]: r["n"] for r in conn.execute(
            "SELECT rating, COUNT(*) AS n FROM interactions WHERE rating IS NOT NULL GROUP BY rating")}
        avg_latency = conn.execute("SELECT AVG(latency_ms) FROM interactions").fetchone()[0]
        unanswered = [dict(r) for r in conn.execute(
            "SELECT LOWER(TRIM(question)) AS question, COUNT(*) AS count, MAX(created_at) AS last_seen,"
            " GROUP_CONCAT(DISTINCT outcome) AS outcomes FROM interactions"
            " WHERE outcome IN ('not_found', 'unverified') GROUP BY LOWER(TRIM(question))"
            " ORDER BY count DESC, last_seen DESC LIMIT ?", (limit,))]
        downvoted = [dict(r) for r in conn.execute(
            "SELECT id, created_at, question, answer, outcome, comment FROM interactions"
            " WHERE rating = 'down' ORDER BY created_at DESC LIMIT ?", (limit,))]
    answered = by_outcome.get("answered", 0)
    return {
        "total": total,
        "answer_rate": answered / total if total else None,
        "by_outcome": by_outcome,
        "feedback": {"up": ratings.get("up", 0), "down": ratings.get("down", 0)},
        "avg_latency_ms": round(avg_latency) if avg_latency else None,
        "unanswered": unanswered,
        "downvoted": downvoted,
    }


def eval_candidates() -> list[dict]:
    """Unanswered and downvoted questions, in the eval file's format, for a human to label."""
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT LOWER(TRIM(question)) AS question, MAX(outcome) AS outcome, MAX(rating) AS rating,"
            " COUNT(*) AS count FROM interactions WHERE outcome IN ('not_found', 'unverified')"
            " OR rating = 'down' GROUP BY LOWER(TRIM(question)) ORDER BY count DESC"
        ).fetchall()
    return [
        {"id": f"review-{i:03d}", "type": "needs_review", "question": r["question"],
         "seen_outcome": r["outcome"], "rating": r["rating"], "times_asked": r["count"],
         "note": "Set type to in_scope (add sources and answer_keywords) or out_of_scope, "
                 "then add to eval/questions.jsonl"}
        for i, r in enumerate(rows, start=1)
    ]
