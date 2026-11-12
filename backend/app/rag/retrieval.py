"""Retrieval: vector search, optionally combined with BM25 keyword search.

Vector search (embeddings) matches meaning: "papers to open an account" finds "documents
required for account opening". It is weaker on exact terms and codes ("closure", "Form 60",
"CKYC"). BM25 is the classic keyword ranking used by search engines and is strong exactly
there. Hybrid mode runs both and merges the two rankings with reciprocal rank fusion (RRF).

The relevance threshold is always applied to the *vector* score of the best match: if no
chunk is semantically close enough, the question is treated as out of scope, whatever the
keyword search finds. That keeps "not found" behaviour the same in both modes.
"""

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import get_settings
from app.rag.vectorstore import get_vectorstore

_STOPWORDS = set(
    "a an the is are was were be been to of for and or in on at by with from as it its this that "
    "these those i me my we our you your do does did can could should would will shall may might "
    "what which who whom how when where why if then than so not no yes there here have has had "
    "about into over under up down out any all some each".split()
)
_RRF_K = 60  # standard reciprocal-rank-fusion constant
_CANDIDATES_PER_RETRIEVER = 3  # each retriever proposes top_k * this candidates


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 over every chunk in the vector store. Small corpus, so held in memory."""

    def __init__(self, ids: list[str], docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.ids, self.docs, self.k1, self.b = ids, docs, k1, b
        self.tfs = [Counter(tokenize(d.page_content)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = [t for t in tokenize(query) if t in self.idf]
        if not terms or not self.docs:
            return []
        scores = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


_lock = threading.Lock()
_index: BM25Index | None = None


def invalidate_keyword_index() -> None:
    """Call after adding or deleting documents; the index is rebuilt on the next search."""
    global _index
    with _lock:
        _index = None


def _keyword_index() -> BM25Index:
    global _index
    with _lock:
        if _index is None:
            data = get_vectorstore().get(include=["documents", "metadatas"])
            docs = [Document(page_content=t or "", metadata=m or {})
                    for t, m in zip(data["documents"], data["metadatas"])]
            _index = BM25Index(data["ids"], docs)
        return _index


@dataclass
class Hit:
    doc: Document
    vector_score: float | None  # relevance 0-1 from embeddings; None if only BM25 found it


def _key(doc: Document) -> tuple:
    m = doc.metadata
    return (m.get("doc_id"), m.get("page"), m.get("start_index"))


def search(query: str) -> tuple[list[Hit], float]:
    """Return (ranked hits, best vector score). Hits are empty if the best score is below the
    relevance threshold, meaning nothing in the documents is close enough to the question."""
    settings = get_settings()
    k = settings.top_k
    vector = get_vectorstore().similarity_search_with_relevance_scores(
        query, k=k * _CANDIDATES_PER_RETRIEVER if settings.retrieval_mode == "hybrid" else k
    )
    best = max((s for _, s in vector), default=0.0)
    if best < settings.relevance_threshold:
        return [], best

    if settings.retrieval_mode != "hybrid":
        hits = [Hit(d, s) for d, s in vector if s >= settings.relevance_threshold]
        return hits[:k], best

    index = _keyword_index()
    keyword = [index.docs[i] for i, _ in index.search(query, k * _CANDIDATES_PER_RETRIEVER)]

    fused: dict[tuple, float] = {}
    by_key: dict[tuple, Hit] = {}
    for rank, (doc, score) in enumerate(vector):
        key = _key(doc)
        fused[key] = fused.get(key, 0.0) + 1 / (_RRF_K + rank + 1)
        by_key[key] = Hit(doc, score)
    for rank, doc in enumerate(keyword):
        key = _key(doc)
        fused[key] = fused.get(key, 0.0) + 1 / (_RRF_K + rank + 1)
        by_key.setdefault(key, Hit(doc, None))
    ranked = sorted(fused, key=fused.get, reverse=True)
    return [by_key[key] for key in ranked[:k]], best
