"""LangGraph pipeline: contextualize -> retrieve -> (generate | no_context)."""

from functools import lru_cache
from typing import Literal, TypedDict

from groq import Groq
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.rag.vectorstore import get_vectorstore


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class SourceGroup(TypedDict):
    """All retrieved chunks from one (document, page). Its position in the list is its citation number."""

    doc_id: str
    filename: str
    page: int
    chunks: list[str]


class RAGState(TypedDict, total=False):
    question: str
    history: list[ChatTurn]
    search_query: str
    documents: list[Document]
    sources: list[SourceGroup]
    answer: str


class LLMNotConfiguredError(RuntimeError):
    pass


@lru_cache
def get_client() -> Groq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMNotConfiguredError("GROQ_API_KEY is not set in backend/.env")
    return Groq(api_key=settings.groq_api_key)


def complete(messages: list[dict]) -> str:
    """One Groq chat completion; returns the assistant's text."""
    settings = get_settings()
    chat_completion = get_client().chat.completions.create(
        messages=messages,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
    )
    return (chat_completion.choices[0].message.content or "").strip()


def _history_messages(history: list[ChatTurn]) -> list[dict]:
    return [{"role": t["role"], "content": t["content"]} for t in history]


CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question, rewrite the question so it "
    "can be understood without the chat history (resolve pronouns and references). "
    "Do NOT answer it. Return only the rewritten question."
)

ANSWER_PROMPT = """You are a helpful assistant that answers questions using ONLY the context below, \
which comes from documents uploaded by the administrator.

Rules:
- If the answer is not in the context, say you couldn't find it in the uploaded documents. Do not make things up.
- Cite sources inline using their numbers, like [1] or [2].
- Be concise and clear. Answer only what was asked.
- Format with plain Markdown: short paragraphs or bullet lists. Use a table only when comparing \
several items side by side. Never use HTML tags such as <br>.

Context:
{context}"""

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the uploaded documents. "
    "Try rephrasing your question, or ask an administrator to upload the relevant documents."
)


def contextualize(state: RAGState) -> RAGState:
    history = state.get("history") or []
    if not history:
        return {"search_query": state["question"]}
    messages = [
        {"role": "system", "content": CONTEXTUALIZE_PROMPT},
        *_history_messages(history),
        {"role": "user", "content": state["question"]},
    ]
    rewritten = complete(messages)
    return {"search_query": rewritten or state["question"]}


def group_sources(docs: list[Document]) -> list[SourceGroup]:
    """Merge chunks from the same (document, page), keeping retrieval order.

    The same numbering is shown to the LLM and returned to the UI, so a citation
    like [2] in the answer always points at the second source in the list.
    """
    groups: dict[tuple[str, int], SourceGroup] = {}
    for d in docs:
        m = d.metadata
        key = (m.get("doc_id", ""), int(m.get("page", 0)))
        group = groups.setdefault(
            key,
            SourceGroup(doc_id=key[0], filename=m.get("filename", "unknown"), page=key[1], chunks=[]),
        )
        group["chunks"].append(d.page_content)
    return list(groups.values())


def retrieve(state: RAGState) -> RAGState:
    settings = get_settings()
    results = get_vectorstore().similarity_search_with_relevance_scores(
        state["search_query"], k=settings.top_k
    )
    # The nearest chunks are returned even for unrelated questions, so drop weak matches.
    docs = [doc for doc, score in results if score >= settings.relevance_threshold]
    return {"documents": docs, "sources": group_sources(docs)}


def route_after_retrieve(state: RAGState) -> Literal["generate", "no_context"]:
    return "generate" if state.get("sources") else "no_context"


def _format_context(sources: list[SourceGroup]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {s['filename']}, page {s['page']})\n" + "\n...\n".join(s["chunks"])
        for i, s in enumerate(sources, start=1)
    )


def generate(state: RAGState) -> RAGState:
    messages = [
        {"role": "system", "content": ANSWER_PROMPT.format(context=_format_context(state["sources"]))},
        *_history_messages(state.get("history") or []),
        {"role": "user", "content": state["question"]},
    ]
    return {"answer": complete(messages)}


def no_context(state: RAGState) -> RAGState:
    return {"answer": NO_CONTEXT_ANSWER}


@lru_cache
def get_graph():
    graph = StateGraph(RAGState)
    graph.add_node("contextualize", contextualize)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("no_context", no_context)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)
    return graph.compile()
