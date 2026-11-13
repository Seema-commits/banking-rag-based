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
from app.api import sessions
from app.api.ratelimit import limit_chat
from app.config import get_settings
from app.guardrails.pii import mask_pii, privacy_notice
from app.insights import store
from app.rag.graph import WEB_LABEL, LLMNotConfiguredError, get_graph

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


def run_chat(req: ChatRequest, on_step: Callable[[str], None] | None = None) -> ChatResponse:
    """Answer one question. Raises HTTPException with a user-facing message on failure."""
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
            state = _run_graph({"question": question.text, "history": history}, on_step)
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
                )
            except Exception:
                logger.exception("Could not store the interaction")  # never fail the answer for it

        return ChatResponse(
            answer=answer, sources=sources, outcome=outcome, session_id=session_id,
            message_id=message_id, notice=privacy_notice(question.found),
            suggestions=state.get("suggestions") or [],
        )
    finally:
        # One structured line per request. No question or answer text, only numbers and flags.
        tokens = metrics.get("prompt_tokens", 0) + metrics.get("completion_tokens", 0)
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
    """Same as /api/chat, as server-sent events: {"type": "step", "step": "<node>"} as each
    agent step starts (so the UI can show progress), then {"type": "result", ...ChatResponse}
    or {"type": "error", "status": <code>, "detail": "<message>"}."""
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            result = run_chat(req, on_step=lambda step: events.put({"type": "step", "step": step}))
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
    obs.log_event(logger, "feedback", rating=req.rating, has_comment=bool(comment))
