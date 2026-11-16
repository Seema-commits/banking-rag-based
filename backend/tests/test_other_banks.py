"""Questions about another bank. Run from backend/: python -m pytest

Replays a real conversation: "process to open a minor account?" was answered well, but "minimum
balance for my kid's account at HDFC Bank?" went to web search, failed the fact-check twice and
ended in "couldn't verify" after ~30 s. These tests check the agent now answers it in one pass.
"""

import types

import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.rag import graph
from app.rag.other_banks import find_other_bank

OWN = "Thistlewood Bank"


@pytest.mark.parametrize("text, bank", [
    ("I want to open my kids account in HDFC Bank. Is there any minimum balance?", "HDFC Bank"),
    ("minimum balance at hdfc for a 14 year old", "HDFC Bank"),
    ("Does SBI allow video KYC?", "State Bank of India"),
    ("Is the State Bank of India charge the same?", "State Bank of India"),
    ("Bank of India vs ICICI fees", "Bank of India"),
    ("compare with Kotak", "Kotak Mahindra Bank"),
    ("What about South Indian Bank?", "South Indian Bank"),
])
def test_finds_other_banks(text, bank):
    assert find_other_bank(text, OWN) == bank


@pytest.mark.parametrize("text", [
    "What is the minimum balance for a savings account?",
    "How do I open an account at Thistlewood Bank?",
    "What does the Reserve Bank of India say about KYC updation?",
    "Is the Reserve Bank's rule on re-KYC the same for minors?",
    "Which documents are officially valid?",
    "Is my sbin0001 IFSC wrong?",  # 'sbi' inside another word isn't a match
    "",
])
def test_ignores_own_bank_rbi_and_plain_questions(text):
    assert find_other_bank(text, OWN) is None


def test_own_bank_is_never_another_bank():
    assert find_other_bank("Is HDFC Bank's process different?", "HDFC Bank") is None


def test_extra_names_from_settings():
    assert find_other_bank("What does Acme Co-op charge?", OWN, ["Acme Co-op"]) == "Acme Co-op"


# ---- the agent, with search and Groq simulated ----

MINOR_CHUNK = ("Minor accounts: under 10 years the account is operated by a parent or guardian; "
               "10 years and above the minor may operate it. Minimum balance for a minor account: Rs 1,000.")


@pytest.fixture
def agent(monkeypatch):
    settings = get_settings()
    for name, value in [("bank_name", OWN), ("other_bank_names", []), ("self_correction", True),
                        ("grounding_check", True), ("max_regenerations", 1), ("max_query_rewrites", 1),
                        ("answer_cache_enabled", False), ("stream_answer", False)]:
        monkeypatch.setattr(settings, name, value)

    log = types.SimpleNamespace(calls=[], answer_prompts=[], web=0, grade="answer", grounded=[True],
                                found=True)

    def fake_search(query):
        log.calls.append("search")
        if not log.found:
            return [], 0.0
        doc = Document(page_content=MINOR_CHUNK, metadata={"doc_id": "d1", "filename": "kyc.pdf", "page": 3})
        return [types.SimpleNamespace(doc=doc, vector_score=0.6)], 0.6

    def fake_json(messages, **_):
        system = messages[0]["content"]
        if system == graph.GRADE_PROMPT:
            log.calls.append("grade")
            return {"decision": log.grade}
        log.calls.append("check")
        ok = log.grounded.pop(0) if log.grounded else False
        return {"grounded": ok, "unsupported": [] if ok else ["HDFC charges Rs 5,000"]}

    def fake_complete(messages, **_):
        if messages[0]["content"] == graph.CONTEXTUALIZE_PROMPT:
            log.calls.append("rewrite")
            return "Minimum balance for a 14 year old's account at HDFC Bank"
        if messages[0]["content"] == graph.SEARCH_REWRITE_PROMPT:
            log.calls.append("query_rewrite")
            return "minor account minimum balance"
        log.calls.append("generate")
        log.answer_prompts.append(messages[0]["content"])
        return "I can only answer for Thistlewood Bank. At Thistlewood Bank the minimum is Rs 1,000 [1]."

    def fake_web(query):
        log.web += 1
        return []

    monkeypatch.setattr(graph, "search", fake_search)
    monkeypatch.setattr(graph, "complete_json", fake_json)
    monkeypatch.setattr(graph, "complete", fake_complete)
    monkeypatch.setattr(graph, "web_search", fake_web)
    monkeypatch.setattr(graph, "web_search_available", lambda: True)
    return log


def ask(question, history=None):
    return graph.get_graph().invoke({"question": question, "history": history or []})


def test_answers_own_bank_part_and_names_the_other_bank(agent):
    state = ask("I want to open my kids account in HDFC Bank. Is there any minimum balance criteria?")
    assert state["outcome"] == "answered"
    assert state["other_bank"] == "HDFC Bank"
    assert agent.calls == ["search", "grade", "generate", "check"]  # one pass, no web, no retries
    prompt = agent.answer_prompts[0]
    assert "asking about HDFC Bank" in prompt and "Never state or guess anything about HDFC Bank" in prompt


def test_not_in_documents_refuses_fast_without_web_or_rewrite(agent):
    agent.grade = "not_in_documents"
    state = ask("What is HDFC Bank's minimum balance?")
    assert state["outcome"] == "other_bank"
    assert agent.calls == ["search", "grade"] and agent.web == 0
    assert "only answer questions about Thistlewood Bank" in state["answer"]
    assert "HDFC Bank's official website" in state["answer"]


def test_nothing_retrieved_names_the_bank(agent):
    agent.found = False
    state = ask("Does ICICI have a kids account?")
    assert state["outcome"] == "other_bank" and "ICICI Bank" in state["answer"]
    assert agent.calls == ["search"]


def test_failed_fact_check_names_the_bank_instead_of_couldnt_verify(agent):
    agent.grounded = [False, False]
    state = ask("Minimum balance at HDFC for a minor?")
    assert state["outcome"] == "other_bank"
    assert "couldn't verify" not in state["answer"]
    assert agent.web == 0


def test_follow_up_keeps_the_bank_from_the_rewrite(agent):
    history = [{"role": "user", "content": "How do I open a minor account?"},
               {"role": "assistant", "content": "Visit a branch with the documents."}]
    state = ask("My kid is 14. And is there a minimum balance for HDFC?", history)
    assert state["other_bank"] == "HDFC Bank"
    assert agent.calls[0] == "rewrite" and agent.web == 0


def test_questions_without_another_bank_still_use_the_web(agent):
    agent.grade = "not_in_documents"
    ask("What does the RBI say about re-KYC for minors?")
    assert agent.web == 1
