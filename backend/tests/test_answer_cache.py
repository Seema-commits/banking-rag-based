"""Tests for the answer cache. Run from backend/: python -m pytest"""

import pytest

from app.config import get_settings
from app.rag import answer_cache


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    answer_cache.clear()
    s = get_settings()
    monkeypatch.setattr(s, "answer_cache_enabled", True)
    monkeypatch.setattr(s, "answer_cache_ttl_seconds", 3600)
    monkeypatch.setattr(s, "answer_cache_max_entries", 3)
    yield
    answer_cache.clear()


def test_same_question_in_different_form_hits():
    answer_cache.put("Can I do Video KYC on a Sunday?", {"answer": "No [1]"})
    assert answer_cache.get("can i do video kyc on a sunday") == {"answer": "No [1]"}
    assert answer_cache.get("  Can I do   Video KYC on a Sunday ?? ") == {"answer": "No [1]"}
    assert answer_cache.get("Can I do Video KYC on a Saturday?") is None


def test_returned_value_is_a_copy():
    answer_cache.put("q", {"sources": [{"page": 1}]})
    answer_cache.get("q")["sources"].append({"page": 2})
    assert answer_cache.get("q") == {"sources": [{"page": 1}]}


def test_expiry(monkeypatch):
    answer_cache.put("q", {"answer": "a"})
    monkeypatch.setattr(answer_cache.time, "time", lambda: 10**12)
    assert answer_cache.get("q") is None


def test_oldest_entries_dropped_beyond_limit():
    for q in ["a", "b", "c"]:
        answer_cache.put(q, {"answer": q})
    answer_cache.get("a")  # recently used, so it survives
    answer_cache.put("d", {"answer": "d"})
    assert answer_cache.get("b") is None and answer_cache.get("a") and answer_cache.size() == 3


def test_clear_and_evict_by_message():
    answer_cache.put("q", {"answer": "a"})
    answer_cache.link_message("m1", "Q?")
    assert answer_cache.evict_message("m1") and answer_cache.get("q") is None
    assert not answer_cache.evict_message("m1")
    answer_cache.put("x", {"answer": "x"})
    answer_cache.clear()
    assert answer_cache.size() == 0


def test_disabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "answer_cache_enabled", False)
    answer_cache.put("q", {"answer": "a"})
    assert answer_cache.get("q") is None
