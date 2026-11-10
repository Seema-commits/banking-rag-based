import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
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


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    history = [m.model_dump() for m in req.history[-get_settings().max_history_messages :]]
    try:
        state = get_graph().invoke({"question": req.question.strip(), "history": history})
    except LLMNotConfiguredError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except Exception:
        logger.exception("Chat pipeline failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The assistant failed to answer. Please try again."
        )

    # One source entry per (document, page), in retrieval order.
    sources: dict[tuple[str, int], Source] = {}
    for doc in state.get("documents") or []:
        m = doc.metadata
        key = (m.get("doc_id", ""), int(m.get("page", 0)))
        sources.setdefault(
            key, Source(doc_id=key[0], filename=m.get("filename", "unknown"), page=key[1])
        )
    return ChatResponse(answer=state["answer"], sources=list(sources.values()))
