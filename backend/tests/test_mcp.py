"""MCP server end to end: the real app on a local port, the official MCP client over HTTP, with the
search and Groq simulated. Run from backend/: python -m pytest"""

import contextlib
import json
import socket
import threading
import time
import types

import anyio
import httpx
import pytest
import uvicorn
from langchain_core.documents import Document
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from app import main, mcp_server
from app.api import ratelimit
from app.config import get_settings
from app.insights import store
from app.rag import graph, registry
from app.rag.registry import DocumentRecord

TOKEN = "test-mcp-token"
CHUNK = ("Video KYC calls are available Monday to Saturday, 9 AM to 9 PM. "
         "NRIs and minors cannot use Video KYC.")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fake_search(query):
    if "home loan" in query.lower():
        return [], 0.0
    doc = Document(page_content=CHUNK, metadata={"doc_id": "d1", "filename": "video-kyc-guide.pdf", "page": 2})
    return [types.SimpleNamespace(doc=doc, vector_score=0.71)], 0.71


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The real FastAPI app (with /mcp) served by uvicorn; startup indexing is skipped."""
    mp = pytest.MonkeyPatch()
    s = get_settings()
    for name, value in [("mcp_token", TOKEN), ("storage_backend", "sqlite"),
                        ("interactions_db", tmp_path_factory.mktemp("db") / "i.db"),
                        ("answer_cache_enabled", False), ("stream_answer", False), ("self_correction", True),
                        ("grounding_check", True), ("mcp_requests_per_minute", 100),
                        ("chat_requests_per_day", 0), ("tavily_api_key", "")]:
        mp.setattr(s, name, value)
    store.get_backend.cache_clear()
    calls = types.SimpleNamespace(llm=0)

    def fake_json(messages, **_):
        calls.llm += 1
        if messages[0]["content"] == graph.GRADE_PROMPT:
            return {"decision": "answer"}
        return {"grounded": True, "unsupported": []}

    def fake_complete(messages, **_):
        calls.llm += 1
        return "No. Video KYC runs Monday to Saturday, 9 AM to 9 PM [1]."

    mp.setattr(graph, "search", _fake_search)
    mp.setattr(mcp_server, "search", _fake_search)
    mp.setattr(graph, "complete_json", fake_json)
    mp.setattr(graph, "complete", fake_complete)
    mp.setattr(registry, "list_documents", lambda: [
        DocumentRecord(id="d1", filename="video-kyc-guide.pdf", pages=3, chunks=6, uploaded_at="2026-11-17T10:00:00+00:00")])

    @contextlib.asynccontextmanager
    async def mcp_only(_app):  # the real lifespan also loads the embedding model; only MCP is needed here
        async with mcp_server.session_manager():
            yield

    mp.setattr(main.app.router, "lifespan_context", mcp_only)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield types.SimpleNamespace(url=f"http://127.0.0.1:{port}{mcp_server.PATH}", calls=calls, port=port)
    srv.should_exit = True
    thread.join(timeout=5)
    mp.undo()
    store.get_backend.cache_clear()


def _call(url, fn, token=TOKEN):
    """Open an MCP session (like Claude Desktop would) and run fn(session)."""

    async def go():
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with create_mcp_http_client(headers=headers) as http, \
                streamable_http_client(url, http_client=http) as (read, write, *_), \
                ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)

    return anyio.run(go)


def _data(result):
    assert not result.is_error, result.content
    return result.structured_content or json.loads(result.content[0].text)


def test_lists_three_read_only_tools(server):
    tools = _call(server.url, lambda s: s.list_tools()).tools
    assert {t.name for t in tools} == {"ask_thistlewood", "search_bank_documents", "list_documents"}
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)
    ask = next(t for t in tools if t.name == "ask_thistlewood")
    assert "question" in ask.input_schema["properties"] and "fact-checked" in ask.description


def test_ask_runs_the_agent_and_is_audited_as_mcp(server):
    before = server.calls.llm
    res = _data(_call(server.url, lambda s: s.call_tool("ask_thistlewood", {"question": "Can I do Video KYC on Sunday?"})))
    assert res["outcome"] == "answered" and "Monday to Saturday" in res["answer"]
    assert res["sources"] == [{"citation": 1, "document": "video-kyc-guide.pdf", "page": 2, "url": None}]
    assert server.calls.llm - before == 3  # grade, answer, fact-check: the same agent as the chat
    record = store.recent(limit=1)[0]
    assert record["channel"] == "mcp" and record["question"] == "Can I do Video KYC on Sunday?"


def test_ask_masks_personal_details(server):
    res = _data(_call(server.url, lambda s: s.call_tool(
        "ask_thistlewood", {"question": "My PAN is ABCDE1234F, can I do Video KYC?"})))
    assert "privacy_notice" in res
    assert "ABCDE1234F" not in store.recent(limit=1)[0]["question"]


def test_search_returns_passages_without_an_llm_call(server):
    before = server.calls.llm
    res = _data(_call(server.url, lambda s: s.call_tool("search_bank_documents", {"query": "video kyc timings"})))
    assert res["passages"][0] == {"document": "video-kyc-guide.pdf", "page": 2, "relevance": 0.71, "text": CHUNK}
    assert server.calls.llm == before
    empty = _data(_call(server.url, lambda s: s.call_tool("search_bank_documents", {"query": "home loan rate"})))
    assert empty["passages"] == []


def test_list_documents(server):
    res = _data(_call(server.url, lambda s: s.call_tool("list_documents", {})))
    assert res["documents"] == [{"document": "video-kyc-guide.pdf", "pages": 3, "added": "2026-11-17T10:00:00+00:00"}]


def test_requires_the_token(server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    assert httpx.post(server.url, json=body, headers=headers).status_code == 401
    wrong = httpx.post(server.url, json=body, headers={**headers, "Authorization": "Bearer nope"})
    assert wrong.status_code == 401 and wrong.headers["www-authenticate"] == "Bearer"


def test_off_without_a_token(server, monkeypatch):
    monkeypatch.setattr(get_settings(), "mcp_token", "")
    r = httpx.post(server.url, json={}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 404


def test_rate_limit_becomes_a_tool_error(server, monkeypatch):
    monkeypatch.setattr(get_settings(), "mcp_requests_per_minute", 1)
    monkeypatch.setattr(ratelimit, "_per_ip", ratelimit.defaultdict(ratelimit.deque))
    _data(_call(server.url, lambda s: s.call_tool("list_documents", {})))  # not limited
    first = _call(server.url, lambda s: s.call_tool("search_bank_documents", {"query": "video kyc"}))
    assert not first.is_error
    second = _call(server.url, lambda s: s.call_tool("search_bank_documents", {"query": "video kyc"}))
    assert second.is_error and "too quickly" in second.content[0].text


def test_chat_page_still_works_and_is_audited_as_web(server):
    r = httpx.post(f"http://127.0.0.1:{server.port}/api/chat", json={"question": "Video KYC on Sunday?"})
    assert r.status_code == 200 and r.json()["outcome"] == "answered"
    assert store.recent(limit=1)[0]["channel"] == "web"
