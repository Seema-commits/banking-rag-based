"""Tests for small-talk detection. Run from backend/: python -m pytest"""

import json
from pathlib import Path

import pytest

from app.rag.smalltalk import small_talk_reply


@pytest.mark.parametrize(
    "text, kind",
    [
        ("Hi", "greeting"), ("hello!", "greeting"), ("Hey there", "greeting"), ("hiii", "greeting"),
        ("Good morning", "greeting"), ("Namaste 🙏", "greeting"), ("hi, how are you?", "greeting"),
        ("Thanks", "thanks"), ("thank you so much!", "thanks"), ("ok thanks", "thanks"), ("thx", "thanks"),
        ("Bye", "goodbye"), ("ok bye", "goodbye"), ("see you later", "goodbye"),
        ("What can you do?", "help"), ("help", "help"), ("who are you", "help"), ("How can you help me?", "help"),
    ],
)
def test_detects_small_talk(text, kind):
    result = small_talk_reply(text)
    assert result is not None and result[0] == kind and result[1]


@pytest.mark.parametrize(
    "text",
    [
        "What we need in KYC in delhi",
        "hi, what documents do I need for KYC?",
        "Thanks, and what is the minimum balance?",
        "help me close my account",
        "Hello, can I do video KYC on Sunday?",
        "",
    ],
)
def test_real_questions_are_not_small_talk(text):
    assert small_talk_reply(text) is None


def test_no_eval_question_is_small_talk():
    path = Path(__file__).resolve().parents[2] / "eval" / "questions.jsonl"
    questions = [json.loads(line)["question"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [q for q in questions if small_talk_reply(q)] == []
