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
