"""Tests for the owner Monitoring dashboard data. Run from backend/: python -m pytest"""

import csv
import io
from datetime import datetime, timedelta, timezone

import pytest

from app import usage
from app.config import get_settings
from app.insights import monitoring, store


@pytest.fixture
def records(tmp_path, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "storage_backend", "sqlite")
    monkeypatch.setattr(s, "interactions_db", tmp_path / "i.db")
    monkeypatch.setattr(s, "slow_answer_ms", 10_000)
    monkeypatch.setattr(s, "llm_cost_per_million_input_tokens", 1.0)
    monkeypatch.setattr(s, "llm_cost_per_million_output_tokens", 2.0)
    store.get_backend.cache_clear()
    usage.reset()
    backend = store.get_backend()
    now = datetime.now(timezone.utc)

    def add(minutes_ago, outcome, latency, question="What is KYC?", **extra):
        rid = f"r{len(backend.recent(1000))}"
        backend.add({"id": rid, "created_at": now - timedelta(minutes=minutes_ago), "session_id": "s",
                     "question": question, "answer": "a", "outcome": outcome, "sources": [], "latency_ms": latency,
                     **extra})
        return rid

    add(5, "answered", 2000, timings_ms={"retrieval": 20, "grade": 300, "generate": 1500, "grounding": 180},
        tokens={"prompt_tokens": 1000, "completion_tokens": 200})
    add(15, "answered", 3000, timings_ms={"retrieval": 30, "grade": 400, "generate": 2300, "grounding": 270})
    add(20, "answered_web", 14000, question="What is the repo rate?",
        timings_ms={"retrieval": 25, "grade": 350, "web_search": 9000, "generate": 4000, "grounding": 600})
    add(25, "answered", 5, cached=True)
    add(30, "not_found", 900, question="Home loan rate?")
    add(32, "not_found", 950, question="home loan rate?")
    add(40, "llm_auth_error", 300)
    add(45, "error", 500, error_type="RuntimeError")
    down = add(50, "answered", 2500, rating="down", comment="wrong")
    add(60 * 30, "answered", 1000)  # 30 hours ago: outside the 24 h window
    yield down
    store.get_backend.cache_clear()


def test_summary_counts_latency_cost(records):
    m = monitoring.monitoring(24)
    s = m["summary"]
    assert s["total"] == 9 and s["answered"] == 5 and s["not_answered"] == 2 and s["errors"] == 2
    assert s["cache_hit_rate"] == round(1 / 9, 3) and s["web_searches"] == 0  # web_searched flag not set here
    # median of the 8 uncached times (the cached 5 ms answer is excluded): (950 + 2000) / 2
    assert s["max_ms"] == 14000 and s["p50_ms"] == 1475
    assert s["tokens"] == {"prompt": 1000, "completion": 200, "total": 1200}
    assert s["estimated_cost_usd"] == round(1000 / 1e6 + 200 / 1e6 * 2, 4)
    assert s["feedback"] == {"up": 0, "down": 1}


def test_steps_errors_slow_gaps(records):
    m = monitoring.monitoring(24)
    assert m["steps"][0]["step"] == "web_search"  # slowest average step first
    assert {e["outcome"] for e in m["errors"]} == {"llm_auth_error", "error"}
    assert "groq-api-key" in next(e for e in m["errors"] if e["outcome"] == "llm_auth_error")["fix"]
    assert next(e for e in m["errors"] if e["outcome"] == "error")["examples"][0]["error_type"] == "RuntimeError"
    assert [x["slowest_step"] for x in m["slow"]] == ["Web search"]
    assert m["gaps"][0] == {**m["gaps"][0], "question": "home loan rate?", "count": 2}
    assert m["downvoted"][0]["comment"] == "wrong"


def test_timeline_buckets(records):
    m = monitoring.monitoring(24)
    assert len(m["timeline"]) in (24, 25)  # hourly
    assert sum(b["answered"] + b["not_answered"] + b["error"] for b in m["timeline"]) == 9
    assert len(monitoring.monitoring(24 * 7)["timeline"]) in (7, 8)  # daily


def test_recommendations(records):
    recs = monitoring.monitoring(24)["recommendations"]
    assert recs[0]["severity"] == "critical" and "API key" in recs[0]["title"]
    titles = " | ".join(r["title"] for r in recs)
    assert "failed" in titles and "couldn't answer" in titles and "slowest 5%" in titles
    assert any(r.get("action") == "upload" for r in recs)
    assert any("Web search is the slowest step" in r["detail"] for r in recs)
    assert "History is stored in a local file" in titles


def _status(falling_back, error=None):
    return {"configured": "llama-3.1-8b-instant", "active": "openai/gpt-oss-120b" if falling_back else
            "llama-3.1-8b-instant", "falling_back": falling_back, "error": error, "failed_at": None,
            "retry_in_s": 300 if falling_back else 0}


def test_warns_while_checks_run_on_the_big_model(records, monkeypatch):
    error = "NotFoundError: The model `llama-3.1-8b-instant` does not exist or you do not have access to it."
    monkeypatch.setattr(monitoring.graph, "check_model_status", lambda: _status(True, error))
    m = monitoring.monitoring(24)
    rec = next(r for r in m["recommendations"] if "big model" in r["title"])
    assert rec["severity"] == "warning" and "openai/gpt-oss-120b" in rec["title"]
    assert "does not exist" in rec["detail"] and "CHECK_MODEL" in rec["detail"]
    assert m["check_model"]["falling_back"]


def test_notes_past_answers_checked_on_the_big_model(records, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(s, "check_model", "openai/gpt-oss-20b")
    monkeypatch.setattr(monitoring.graph, "check_model_status", lambda: _status(False))
    backend = store.get_backend()
    now = datetime.now(timezone.utc)
    for i, check in enumerate(["openai/gpt-oss-120b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]):
        backend.add({"id": f"m{i}", "created_at": now - timedelta(minutes=3), "session_id": "s", "question": "q",
                     "answer": "a", "outcome": "answered", "sources": [], "latency_ms": 1500,
                     "llm_model": "openai/gpt-oss-120b", "check_model": check})
    titles = [r["title"] for r in monitoring.monitoring(24)["recommendations"]]
    assert "2 answers in this period ran their checks on the big model" in titles
    assert not any(t.startswith("Checks are running") for t in titles)


def test_no_check_model_note_when_all_is_well(records, monkeypatch):
    monkeypatch.setattr(monitoring.graph, "check_model_status", lambda: _status(False))
    assert not any("big model" in r["title"] for r in monitoring.monitoring(24)["recommendations"])


def test_get_and_csv(records):
    assert store.get(records)["comment"] == "wrong" and store.get("missing") is None
    rows = list(csv.DictReader(io.StringIO(monitoring.audit_csv(24))))
    assert len(rows) == 9 and {"question", "answer", "prompt_version", "rating"} <= set(rows[0])


def test_empty_window(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "interactions_db", tmp_path / "empty.db")
    store.get_backend.cache_clear()
    m = monitoring.monitoring(24)
    assert m["summary"]["total"] == 0 and m["summary"]["p95_ms"] is None and m["errors"] == []
    store.get_backend.cache_clear()


def test_percentile():
    assert monitoring.percentile([], 0.5) is None
    assert monitoring.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert monitoring.percentile([10], 0.95) == 10
