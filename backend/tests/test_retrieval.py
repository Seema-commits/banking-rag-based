"""Tests for BM25 keyword search and repeated-line stripping. Run from backend/: python -m pytest"""

from langchain_core.documents import Document

from app.rag.ingestion import strip_repeated_lines
from app.rag.retrieval import BM25Index, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the charge for Account Closure?") == ["charge", "account", "closure"]


def test_bm25_prefers_exact_rare_terms():
    docs = [
        Document(page_content="Account opening needs a PAN card and an officially valid document."),
        Document(page_content="Account closure within 14 days is free; after that Rs 500 applies."),
        Document(page_content="Video KYC is available Monday to Saturday."),
    ]
    index = BM25Index([str(i) for i in range(len(docs))], docs)
    ranked = [i for i, _ in index.search("charge for closing account closure", k=3)]
    assert ranked[0] == 1


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index(["0"], [Document(page_content="Video KYC hours")])
    assert index.search("weather forecast", k=3) == []


def test_strip_repeated_lines_removes_footers_keeps_content():
    footer = "Thistlewood Bank is fictional. Portfolio demo."
    pages = [
        Document(page_content=f"Video KYC hours\nCalls run 9 AM to 9 PM\n{footer}\nPage 1"),
        Document(page_content=f"Rejection reasons\nPhotocopies are not accepted\n{footer}\nPage 2"),
        Document(page_content=f"Dropped calls\nRestart within 72 hours\n{footer}\nPage 3"),
    ]
    removed = strip_repeated_lines(pages)
    assert removed == 6  # footer + "Page N" on each of 3 pages
    text = "\n".join(p.page_content for p in pages)
    assert footer not in text and "Page 2" not in text
    assert "Calls run 9 AM to 9 PM" in text and "Restart within 72 hours" in text


def test_strip_repeated_lines_needs_two_pages():
    pages = [Document(page_content="Only page\nFooter")]
    assert strip_repeated_lines(pages) == 0
