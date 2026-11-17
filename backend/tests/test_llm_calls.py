"""Tests for the Groq call wrapper (fast checks, fallbacks). Run from backend/: python -m pytest"""

import types

import groq
import httpx
import pytest

from app.config import get_settings
from app.rag import graph


def _reply(text="ok"):
    usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))], usage=usage)


def _reset_check_model(monkeypatch, now=1000.0):
    clock = types.SimpleNamespace(now=now)
    monkeypatch.setattr(graph, "_clock", lambda: clock.now)
    monkeypatch.setattr(graph, "_check_model_down_until", 0.0)
    monkeypatch.setattr(graph, "_check_model_error", "")
    monkeypatch.setattr(graph, "_check_model_failed_at", None)
    monkeypatch.setattr(get_settings(), "check_model_retry_seconds", 600)
    return clock


def _not_found(model):
    return groq.NotFoundError(
        "model_not_found", response=httpx.Response(404, request=httpx.Request("POST", "https://x")),
        body={"error": {"message": f"The model `{model}` does not exist or you do not have access to it.",
                        "type": "invalid_request_error", "code": "model_not_found"}})


class Recorder(list):
    reject_extras = False


@pytest.fixture
def calls(monkeypatch):
    seen = Recorder()

    def create(**kwargs):
        seen.append(kwargs)
        if seen.reject_extras and ("reasoning_effort" in kwargs or "response_format" in kwargs):
            raise groq.BadRequestError(
                "unsupported", response=httpx.Response(400, request=httpx.Request("POST", "https://x")), body=None)
        return _reply('{"decision": "answer"}')

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(graph, "get_client", lambda: client)
    _reset_check_model(monkeypatch)
    settings = get_settings()
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "check_model", "")
    monkeypatch.setattr(settings, "check_reasoning_effort", "low")
    return seen


def test_fast_checks_use_low_reasoning_and_json_mode(calls, monkeypatch):
    monkeypatch.setattr(get_settings(), "check_reasoning_effort", "low")
    assert graph.complete_json([{"role": "user", "content": "x"}]) == {"decision": "answer"}
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_customer_answer_uses_model_default(calls):
    graph.complete([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in calls[0] and "response_format" not in calls[0]


def test_falls_back_when_model_rejects_extras(calls):
    calls.reject_extras = True
    assert graph.complete([{"role": "user", "content": "x"}], json_mode=True, fast=True)
    assert len(calls) == 2 and "reasoning_effort" not in calls[1] and "response_format" not in calls[1]


def test_reasoning_effort_can_be_disabled(calls, monkeypatch):
    monkeypatch.setattr(get_settings(), "check_reasoning_effort", "")
    graph.complete([{"role": "user", "content": "x"}], fast=True)
    assert "reasoning_effort" not in calls[0]


def test_checks_use_fast_model_and_answers_use_main_model(calls, monkeypatch):
    monkeypatch.setattr(get_settings(), "check_model", "llama-3.1-8b-instant")
    graph.complete_json([{"role": "user", "content": "grade"}])
    graph.complete([{"role": "user", "content": "answer"}])
    assert calls[0]["model"] == "llama-3.1-8b-instant"
    assert "reasoning_effort" not in calls[0]  # not a reasoning model
    assert calls[1]["model"] == "openai/gpt-oss-120b"


def _client_failing_for(monkeypatch, bad_model):
    seen = []

    def create(**kwargs):
        seen.append(kwargs["model"])
        if kwargs["model"] == bad_model:
            raise _not_found(bad_model)
        return _reply("ok")

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(graph, "get_client", lambda: client)
    monkeypatch.setattr(get_settings(), "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(get_settings(), "check_model", bad_model)
    return seen


def test_falls_back_to_main_model_when_check_model_is_gone(monkeypatch):
    _reset_check_model(monkeypatch)
    seen = _client_failing_for(monkeypatch, "retired-model")
    assert graph.complete([{"role": "user", "content": "x"}], fast=True) == "ok"
    assert graph.complete([{"role": "user", "content": "y"}], fast=True) == "ok"
    # first call tries the check model, then falls back; later calls go straight to the main model
    assert seen == ["retired-model", "openai/gpt-oss-120b", "openai/gpt-oss-120b"]
    assert graph.active_check_model() == "openai/gpt-oss-120b"


def test_check_model_is_retried_after_the_cooldown(monkeypatch):
    clock = _reset_check_model(monkeypatch)
    seen = _client_failing_for(monkeypatch, "retired-model")
    graph.complete([{"role": "user", "content": "x"}], fast=True)
    clock.now += 599
    graph.complete([{"role": "user", "content": "x"}], fast=True)
    assert seen == ["retired-model", "openai/gpt-oss-120b", "openai/gpt-oss-120b"]  # still cooling down
    clock.now += 2
    graph.complete([{"role": "user", "content": "x"}], fast=True)
    assert seen[-2:] == ["retired-model", "openai/gpt-oss-120b"]  # tried again, still failing


def test_recovers_when_the_check_model_works_again(monkeypatch):
    clock = _reset_check_model(monkeypatch)
    seen = _client_failing_for(monkeypatch, "flaky-model")
    graph.complete([{"role": "user", "content": "x"}], fast=True)
    assert graph.check_model_status()["falling_back"]
    _client_failing_for(monkeypatch, "some-other-model")  # flaky-model works from now on
    monkeypatch.setattr(get_settings(), "check_model", "flaky-model")
    clock.now += 601
    graph.complete([{"role": "user", "content": "x"}], fast=True)
    status = graph.check_model_status()
    assert status["active"] == "flaky-model" and not status["falling_back"] and status["error"] is None


def test_status_reports_groqs_reason_and_retry_time(monkeypatch, caplog):
    clock = _reset_check_model(monkeypatch)
    _client_failing_for(monkeypatch, "llama-3.1-8b-instant")
    with caplog.at_level("WARNING"):
        graph.complete([{"role": "user", "content": "x"}], fast=True)
    clock.now += 100
    status = graph.check_model_status()
    assert status["falling_back"] and status["configured"] == "llama-3.1-8b-instant"
    assert status["active"] == "openai/gpt-oss-120b"
    assert "does not exist or you do not have access" in status["error"]
    assert status["error"].startswith("NotFoundError")
    assert status["retry_in_s"] == 500 and status["failed_at"]
    assert "does not exist" in caplog.text  # the log line carries Groq's message, not just the type


def test_answers_never_use_the_check_model(monkeypatch):
    _reset_check_model(monkeypatch)
    seen = _client_failing_for(monkeypatch, "retired-model")
    graph.complete([{"role": "user", "content": "answer"}])
    assert seen == ["openai/gpt-oss-120b"] and not graph.check_model_status()["falling_back"]


def test_complete_stream_passes_pieces_and_returns_full_text(monkeypatch):
    def chunk(text=None, usage=None):
        delta = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)] if usage is None else [],
                                     x_groq=types.SimpleNamespace(usage=usage) if usage else None)

    seen_kwargs = {}

    def create(**kwargs):
        seen_kwargs.update(kwargs)
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return iter([chunk("Video "), chunk(None), chunk("KYC [1]."), chunk(usage=usage)])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(graph, "get_client", lambda: client)
    pieces = []
    assert graph.complete_stream([{"role": "user", "content": "x"}], pieces.append) == "Video KYC [1]."
    assert pieces == ["Video ", "KYC [1]."]
    assert seen_kwargs["stream"] is True and seen_kwargs["model"] == get_settings().groq_model
