import logging
import time
from typing import Literal

import groq

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app import observability as obs
from app.api.ratelimit import limit_chat
from app.config import get_settings
from app.guardrails.pii import mask_pii, privacy_notice
from app.rag.graph import LLMNotConfiguredError, get_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = []


class Source(BaseModel):
    doc_id: str
    filename: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    # Set when personal details were removed from the question before processing.
    notice: str | None = None


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(limit_chat)])
def chat(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    metrics = obs.start_request()
    started = time.perf_counter()
    outcome = "error"

    # Guardrail: mask Aadhaar, PAN, card/account numbers, OTPs etc. before anything is sent
    # to the LLM provider or the vector search. History is masked too, since it is resent.
    question = mask_pii(req.question.strip())
    history = [
        {"role": m.role, "content": mask_pii(m.content).text}
        for m in req.history[-settings.max_history_messages :]
    ]
    try:
        try:
            state = get_graph().invoke({"question": question.text, "history": history})
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
            Source(doc_id=s["doc_id"], filename=s["filename"], page=s["page"])
            for s in state.get("sources") or []
        ]
        outcome = "answered" if sources else "not_found"
        return ChatResponse(
            answer=state["answer"], sources=sources, notice=privacy_notice(question.found)
        )
    finally:
        # One structured line per request. No question or answer text, only numbers and flags.
        tokens = metrics.get("prompt_tokens", 0) + metrics.get("completion_tokens", 0)
        obs.log_event(
            logger,
            "chat",
            level=logging.INFO if outcome in ("answered", "not_found") else logging.ERROR,
            outcome=outcome,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            question_chars=len(req.question),
            history_messages=len(history),
            pii_masked=sorted(set(question.found)),
            llm_model=settings.groq_model,
            total_tokens=tokens,
            **metrics,
        )
