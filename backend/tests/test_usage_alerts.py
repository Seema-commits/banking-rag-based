"""Tests for owner-alert signals: daily usage thresholds and web-search failure reasons."""

import logging

import httpx
import pytest

from app import usage
from app.config import get_settings
from app.rag import websearch


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    usage.reset()
    s = get_settings()
    monkeypatch.setattr(s, "chat_requests_per_day", 10)
    monkeypatch.setattr(s, "daily_llm_token_warning", 1000)
    monkeypatch.setattr(s, "daily_web_search_warning", 0)  # off
    yield
    usage.reset()


def events(caplog):
    return [(r.fields["metric"], r.fields["percent"]) for r in caplog.records
            if getattr(r, "fields", {}).get("event") == "usage_threshold"]


def test_warns_once_at_80_and_100_percent(caplog):
    caplog.set_level(logging.INFO)
    for _ in range(12):
        usage.add("questions")
    assert events(caplog) == [("questions", 80), ("questions", 100)]


def test_large_jump_crosses_both_marks(caplog):
    caplog.set_level(logging.INFO)
    usage.add("llm_tokens", 400)
    assert events(caplog) == []
    usage.add("llm_tokens", 700)  # 400 -> 1100: crosses 800 and 1000 at once
    assert events(caplog) == [("llm_tokens", 80), ("llm_tokens", 100)]
    assert usage.snapshot()["llm_tokens"] == {"used": 1100, "limit": 1000}


def test_zero_limit_never_warns_and_new_day_resets(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    for _ in range(50):
        usage.add("web_searches")
    assert events(caplog) == []
    usage._state["date"] = "2000-01-01"  # pretend it's a new day
    usage.add("questions")
    assert usage.snapshot()["questions"]["used"] == 1 and usage.snapshot()["web_searches"]["used"] == 0


def _status_error(code):
    request = httpx.Request("POST", "https://api.tavily.com/search")
    return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code, request=request))


@pytest.mark.parametrize("exc, reason", [
    (_status_error(401), "auth"), (_status_error(403), "auth"),
    (_status_error(429), "quota"), (_status_error(432), "quota"), (_status_error(433), "quota"),
    (_status_error(500), "http_500"),
    (httpx.ConnectTimeout("t"), "timeout"), (httpx.ConnectError("c"), "network"),
    (ValueError("bad json"), "bad_response"),
])
def test_failure_reasons(exc, reason):
    assert websearch.failure_reason(exc) == reason


def test_failed_search_logs_reason_and_counts(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    s = get_settings()
    monkeypatch.setattr(s, "tavily_api_key", "tvly-x")
    monkeypatch.setattr(s, "web_search_domains", ["rbi.org.in"])

    def boom(*a, **k):
        raise _status_error(401)

    monkeypatch.setattr(websearch.httpx, "post", boom)
    assert websearch.web_search("q") == []
    failed = [r for r in caplog.records if getattr(r, "fields", {}).get("event") == "web_search_failed"]
    assert failed and failed[0].fields["reason"] == "auth" and failed[0].levelno == logging.ERROR
    assert usage.snapshot()["web_searches"]["used"] == 1
