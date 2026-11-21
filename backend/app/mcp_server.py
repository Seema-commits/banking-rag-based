"""MCP server: the bank's knowledge base as tools any AI agent can use (docs/mcp.md).

MCP (Model Context Protocol) is an open standard for connecting AI apps to tools. With this,
Claude Desktop, Claude Code or another agent can ask Thistlewood Bank questions and get the same
checked, cited answers as the chat page, because the tools call the same pipeline:

    ask_thistlewood          the full agent: PII masking, cache, hybrid search, relevance check,
                             answer, fact-check; logged in the audit log with channel "mcp"
    search_bank_documents    hybrid search only (no LLM): the matching passages with file and page
    list_documents           what the knowledge base contains

Served over Streamable HTTP at /mcp on the same Cloud Run service, in stateless JSON mode (no
server-side session, so it works across restarts and instances). It is off unless MCP_TOKEN is
set; every request must send "Authorization: Bearer <MCP_TOKEN>". Calls share the chat's daily
question cap and have their own per-minute limit (MCP_REQUESTS_PER_MINUTE). All tools are
read-only: nothing can change documents or see account data.
"""

import hmac
import json
import logging

import anyio
from fastapi import HTTPException
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from app.api import ratelimit
from app.api.chat import ChatRequest, run_chat
from app.config import get_settings
from app.guardrails.pii import mask_pii
from app.observability import log_event
from app.rag import registry
from app.rag.retrieval import search

logger = logging.getLogger(__name__)

PATH = "/mcp"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

server = MCPServer(
    name="thistlewood-bank",
    title="Thistlewood Bank knowledge base",
    version="1.0.0",
    instructions=(
        "Answers questions about Thistlewood Bank (a fictional bank in a portfolio demo): account opening, "
        "KYC, Video KYC, re-KYC, fees and support. Use ask_thistlewood for a checked answer with citations, "
        "search_bank_documents to read the source passages yourself, and list_documents to see what is "
        "covered. It cannot see anyone's account and gives no financial advice. Never send personal "
        "details such as Aadhaar, PAN, card or account numbers."
    ),
)


def _limit(count_daily: bool) -> None:
    try:
        ratelimit.check_limits("mcp", get_settings().mcp_requests_per_minute, count_daily=count_daily)
    except HTTPException as exc:
        raise ToolError(exc.detail) from exc


@server.tool(title="Ask Thistlewood Bank", annotations=READ_ONLY)
async def ask_thistlewood(question: str) -> dict:
    """Ask a question about Thistlewood Bank's accounts, KYC, Video KYC, re-KYC, fees or support.

    Runs the bank's full answering agent: personal details are masked, the bank's documents are
    searched, the answer is written only from them (or from official Indian government websites when
    the documents don't cover it) and fact-checked against the sources. Returns the answer with
    numbered citations, its sources, and an outcome such as "answered", "not_found", "other_bank" or
    "needs_account_access". Use the answer as written; don't add facts that aren't in it.
    """
    if not question.strip():
        raise ToolError("Ask a question, for example: What documents do I need to open an account?")
    _limit(count_daily=True)
    try:
        # The agent blocks while Groq works; run it in a worker thread so the server stays responsive.
        res = await anyio.to_thread.run_sync(
            lambda: run_chat(ChatRequest(question=question[:2000]), channel="mcp"))
    except HTTPException as exc:
        raise ToolError(exc.detail) from exc
    return {
        "answer": res.answer,
        "outcome": res.outcome,
        "sources": [
            {"citation": i, "document": s.filename, "page": s.page or None, "url": s.url}
            for i, s in enumerate(res.sources, start=1)
        ],
        "cached": res.cached,
        "latency_ms": res.latency_ms,
        **({"privacy_notice": res.notice} if res.notice else {}),
    }


@server.tool(title="Search the bank's documents", annotations=READ_ONLY)
async def search_bank_documents(query: str) -> dict:
    """Search Thistlewood Bank's documents and return the most relevant passages, without writing an answer.

    Uses the same hybrid search as the chat (meaning-based embeddings plus keyword matching) and the
    same relevance threshold, so an empty result means the documents don't cover the query. Each
    passage has its document name and page, to cite. Personal details in the query are masked first.
    """
    if not query.strip():
        raise ToolError("Give a search query, for example: Video KYC timings")
    _limit(count_daily=False)  # no LLM call, so it doesn't use the daily question cap
    masked = mask_pii(query.strip()[:2000])
    hits, best = await anyio.to_thread.run_sync(lambda: search(masked.text))
    log_event(logger, "mcp_search", results=len(hits), top_score=round(best, 3), pii_masked=sorted(set(masked.found)))
    return {
        "query": masked.text,
        "passages": [
            {
                "document": h.doc.metadata.get("filename", "unknown"),
                "page": h.doc.metadata.get("page"),
                "relevance": None if h.vector_score is None else round(h.vector_score, 3),
                "text": h.doc.page_content,
            }
            for h in hits
        ],
    }


@server.tool(title="List the bank's documents", annotations=READ_ONLY)
def list_documents() -> dict:
    """List the documents in Thistlewood Bank's knowledge base: file name, pages and when it was added."""
    return {"documents": [{"document": d.filename, "pages": d.pages, "added": d.uploaded_at}
                          for d in registry.list_documents()]}


# ---------------------------------------------------------------------------- HTTP endpoint

def build_routes() -> list:
    """The /mcp route, for the FastAPI app. Call once at import time; run session_manager() in the
    app's lifespan."""
    starlette_app = server.streamable_http_app(
        streamable_http_path=PATH,
        stateless_http=True,
        json_response=True,
        # The library's DNS-rebinding guard protects unauthenticated servers on a laptop. This
        # server requires a bearer token and must accept its public Cloud Run host name.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    route = next(r for r in starlette_app.routes if getattr(r, "path", None) == PATH)
    route.app = _require_token(route.app)
    return [route]


def session_manager():
    return server.session_manager.run()


def _require_token(app):
    """ASGI wrapper: 404 when MCP is off (no MCP_TOKEN), 401 without the right bearer token."""

    async def guarded(scope, receive, send):
        token = get_settings().mcp_token
        if not token:
            return await _json(send, 404, {"detail": "MCP is not enabled on this server"})
        auth = dict(scope.get("headers") or []).get(b"authorization", b"").decode("latin-1")
        given = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
        if not hmac.compare_digest(given.encode(), token.encode()):
            log_event(logger, "mcp_unauthorized", level=logging.WARNING)
            return await _json(send, 401, {"detail": "Missing or wrong MCP token"},
                               {"www-authenticate": "Bearer"})
        return await app(scope, receive, send)

    return guarded


async def _json(send, status: int, body: dict, extra_headers: dict | None = None) -> None:
    data = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]
    headers += [(k.encode(), v.encode()) for k, v in (extra_headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": data})
