import json
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from typing import Literal

import groq
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import observability as obs
from app import usage
from app.api import sessions
from app.api.ratelimit import limit_chat
from app.config import get_settings
from app.guardrails.pii import mask_pii, privacy_notice
from app.insights import store
from app.rag import answer_cache
from app.rag.graph import (PROMPT_VERSION, WEB_LABEL, LLMNotConfiguredError, active_check_model,
                           get_graph, set_token_sink)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Chat history lives on the server; the client only sends the id it got back last time.
    # (A "history" field from older clients is ignored.)
    session_id: str | None = Field(default=None, max_length=64)


class Source(BaseModel):
    doc_id: str  # "web" for web search results
    filename: str  # page title for web results
    page: int  # 0 for web results
    url: str | None = None  # link for web results


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    # answered, answered_web, small_talk, not_found, needs_account_access, financial_advice,
    # off_topic, unsafe, unverified
    outcome: str
    session_id: str
    # Id for thumbs-up/down feedback on this answer (None if interactions aren't stored).
    message_id: str | None = None
    # Set when personal details were removed from the question before processing.
    notice: str | None = None
    # Example questions to show as clickable buttons (after a greeting).
    suggestions: list[str] = []
    # True when the answer came from the answer cache (a repeated question).
    cached: bool = False
    # Server-side processing time and the time per agent step, in milliseconds.
    latency_ms: float = 0
    timings: dict[str, float] = {}


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)


def _run_graph(inputs: dict, on_step: Callable[[str], None] | None) -> dict:
    """Run the agent and return its final state, reporting each step as it starts."""
    state: dict = {}
    for mode, chunk in get_graph().stream(inputs, stream_mode=["tasks", "values"]):
        if mode == "values":
            state = chunk
        elif on_step and "input" in chunk and "result" not in chunk:
            on_step(chunk["name"])
    return state


# Outcomes worth caching: real answers and deliberate refusals, not errors or small talk.
_CACHEABLE = {"answered", "answered_web", "not_found", "needs_account_access", "financial_advice",
              "off_topic", "unsafe"}


def _audit_details(metrics: dict, history: list, pii_found: list[str]) -> dict:
    """What produced this answer, for the audit log: no personal data, only settings and numbers."""
    settings = get_settings()
    return {
        "llm_model": settings.groq_model,
        "check_model": active_check_model(),
        "prompt_version": PROMPT_VERSION,
        "retrieval_mode": settings.retrieval_mode,
        "relevance_threshold": settings.relevance_threshold,
        "history_messages": len(history),
        "pii_masked": sorted(set(pii_found)),
        "tokens": {k: metrics.get(k, 0) for k in ("prompt_tokens", "completion_tokens")},
        "timings_ms": {k[:-3]: v for k, v in metrics.items() if k.endswith("_ms") and k != "llm_ms"},
        **{k: metrics[k] for k in ("decision", "grounded", "regenerations", "query_rewrites", "web_searched",
                                   "web_results", "cached", "small_talk", "rewritten") if k in metrics},
    }


def _store_failure(settings, session_id, question, outcome, metrics, history, started) -> None:
    """Failed requests go into the audit log too, so the owner's Monitoring tab can show them."""
    if not settings.store_interactions:
        return
    try:
        details = _audit_details(metrics, history, question.found)
        if "error_type" in metrics:
            details["error_type"] = metrics["error_type"]
        store.record_interaction(
            session_id=session_id, question=question.text, answer="", outcome=outcome, sources=[],
            top_score=metrics.get("top_score"), latency_ms=round((time.perf_counter() - started) * 1000, 1),
            details=details,
        )
    except Exception:
        logger.exception("Could not store the failed interaction")


def run_chat(req: ChatRequest, on_step: Callable[[str], None] | None = None,
             on_token: Callable[[str], None] | None = None) -> ChatResponse:
    """Answer one question. Raises HTTPException with a user-facing message on failure.

    on_step(step) is called as each agent step starts; on_token(text) receives the answer text
    while it is being written (before the grounding check; the returned answer is final)."""
    settings = get_settings()
    metrics = obs.start_request()
    started = time.perf_counter()
    outcome = "error"

    # Guardrail: mask Aadhaar, PAN, card/account numbers, OTPs etc. before anything is sent
    # to the LLM provider, the vector search, the logs or the interaction store.
    question = mask_pii(req.question.strip())
    session_id, history = sessions.get_history(req.session_id, settings.max_history_messages)
    try:
        try:
            token = set_token_sink(on_token)
            try:
                state = _run_graph({"question": question.text, "history": history}, on_step)
            finally:
                token.var.reset(token)
        except LLMNotConfiguredError as exc:
            outcome = "llm_not_configured"
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
        except groq.AuthenticationError:
            outcome = "llm_auth_error"
            logger.error("Groq rejected the API key (401). Check GROQ_API_KEY.")
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The AI service rejected the API key. Check GROQ_API_KEY on the server.",
            )
        except groq.RateLimitError:
            outcome = "llm_rate_limited"
            logger.warning("Groq rate limit reached")
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The AI service is busy right now. Please try again in a minute.",
            )
        except Exception as exc:
            metrics["error_type"] = type(exc).__name__
            logger.exception("Chat pipeline failed")
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "The assistant failed to answer. Please try again."
            )

        # Same order and numbering the LLM saw, so citation [n] matches source n in the UI.
        sources = [
            Source(doc_id=s["doc_id"], filename=s["filename"], page=s["page"], url=s.get("url"))
            for s in state.get("sources") or []
        ]
        outcome = state.get("outcome") or ("answered" if sources else "not_found")
        # Cache by the standalone question (follow-ups already resolved by the agent).
        cached = bool(state.get("cached"))
        cache_question = state.get("standalone_question") or question.text
        cacheable = outcome in _CACHEABLE
        if cacheable and not cached:
            answer_cache.put(cache_question, {k: state.get(k) for k in ("answer", "sources", "outcome", "suggestions")})
        # Some models emit their own citation style, e.g. "【1†L1-L5】"; normalise it to "[1]".
        answer = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", state["answer"])
        if outcome == "answered_web":
            answer = WEB_LABEL + answer
        sessions.append_turns(session_id, question.text, answer, settings.max_history_messages)

        message_id = None
        if settings.store_interactions:
            try:
                message_id = store.record_interaction(
                    session_id=session_id, question=question.text, answer=answer, outcome=outcome,
                    sources=[s.model_dump() for s in sources], top_score=metrics.get("top_score"),
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                    details=_audit_details(metrics, history, question.found),
                )
            except Exception:
                logger.exception("Could not store the interaction")  # never fail the answer for it
        if message_id and cacheable:
            answer_cache.link_message(message_id, cache_question)  # so 👎 can evict it

        return ChatResponse(
            answer=answer, sources=sources, outcome=outcome, session_id=session_id,
            message_id=message_id, notice=privacy_notice(question.found),
            suggestions=state.get("suggestions") or [],
            cached=cached,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            timings={k[:-3]: v for k, v in metrics.items() if k.endswith("_ms") and k != "llm_ms"},
        )
    finally:
        if outcome == "error" or outcome.startswith("llm_"):
            _store_failure(settings, session_id, question, outcome, metrics, history, started)
        # One structured line per request. No question or answer text, only numbers and flags.
        tokens = metrics.get("prompt_tokens", 0) + metrics.get("completion_tokens", 0)
        usage.add("llm_tokens", tokens)
        obs.log_event(
            logger,
            "chat",
            level=logging.ERROR if outcome.startswith("llm_") or outcome == "error" else logging.INFO,
            outcome=outcome,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            question_chars=len(req.question),
            history_messages=len(history),
            pii_masked=sorted(set(question.found)),
            llm_model=settings.groq_model,
            total_tokens=tokens,
            **metrics,
        )


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(limit_chat)])
def chat(req: ChatRequest) -> ChatResponse:
    return run_chat(req)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", dependencies=[Depends(limit_chat)])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Same as /api/chat, as server-sent events:
    {"type": "step", "step": "<node>"} as each agent step starts (so the UI can show progress),
    {"type": "token", "text": "..."} while the answer is written (a draft: it is checked
    afterwards, and a new "generate" step means the draft is being rewritten),
    then {"type": "result", ...ChatResponse} or {"type": "error", "status": <code>, "detail": "..."}.
    The result's answer is the final one and replaces the draft."""
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            result = run_chat(
                req,
                on_step=lambda step: events.put({"type": "step", "step": step}),
                on_token=lambda text: events.put({"type": "token", "text": text}),
            )
            events.put({"type": "result", **result.model_dump()})
        except HTTPException as exc:
            events.put({"type": "error", "status": exc.status_code, "detail": exc.detail})
        except Exception:  # run_chat already converts pipeline errors; this is a last resort
            logger.exception("Streaming chat failed")
            events.put({"type": "error", "status": 500, "detail": "The assistant failed to answer."})
        finally:
            events.put(None)

    def stream() -> Iterator[str]:
        threading.Thread(target=worker, daemon=True).start()
        while (event := events.get()) is not None:
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/suggestions")
def suggestions() -> list[str]:
    """Example questions for the empty chat screen."""
    return get_settings().suggested_questions


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
def feedback(req: FeedbackRequest) -> None:
    """Thumbs up/down (and an optional comment) on one answer."""
    comment = mask_pii(req.comment).text if req.comment else None
    if not store.set_feedback(req.message_id, req.rating, comment):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown message")
    if req.rating == "down" and answer_cache.evict_message(req.message_id):
        logger.info("Removed a downvoted answer from the cache")
    obs.log_event(logger, "feedback", rating=req.rating, has_comment=bool(comment))
