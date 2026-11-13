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
    monkeypatch.setattr(graph, "_check_model_failed", False)
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


def test_falls_back_to_main_model_when_check_model_is_gone(monkeypatch):
    seen = []

    def create(**kwargs):
        seen.append(kwargs["model"])
        if kwargs["model"] == "retired-model":
            raise groq.NotFoundError("model_not_found",
                                     response=httpx.Response(404, request=httpx.Request("POST", "https://x")), body=None)
        return _reply("ok")

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(graph, "get_client", lambda: client)
    monkeypatch.setattr(graph, "_check_model_failed", False)
    monkeypatch.setattr(get_settings(), "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(get_settings(), "check_model", "retired-model")
    assert graph.complete([{"role": "user", "content": "x"}], fast=True) == "ok"
    assert graph.complete([{"role": "user", "content": "y"}], fast=True) == "ok"
    # first call tries the check model, then falls back; later calls go straight to the main model
    assert seen == ["retired-model", "openai/gpt-oss-120b", "openai/gpt-oss-120b"]
