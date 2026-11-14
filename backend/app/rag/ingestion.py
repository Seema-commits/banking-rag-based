"""PDF extraction, chunking and indexing into Chroma."""

import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag import registry
from app.rag.registry import DocumentRecord
from app.rag import answer_cache
from app.rag.retrieval import invalidate_keyword_index
from app.rag.vectorstore import get_vectorstore

# Chroma rejects very large single inserts; stay well under its limit.
_BATCH_SIZE = 256


class IngestionError(Exception):
    pass


def _line_key(line: str) -> str:
    # "Page 2" and "Page 3" count as the same line.
    return re.sub(r"\d+", "#", " ".join(line.split()).lower())


def strip_repeated_lines(pages: list[Document]) -> int:
    """Remove lines that appear on at least 60% of the pages (headers, footers, page numbers).

    Edits the pages in place and returns how many lines were removed. Needs 2+ pages.
    """
    if len(pages) < 2:
        return 0
    counts = Counter()
    for page in pages:
        counts.update({_line_key(l) for l in page.page_content.splitlines() if 0 < len(l.strip()) <= 200})
    min_pages = max(2, -(-len(pages) * 6 // 10))  # ceil(60% of pages)
    repeated = {key for key, n in counts.items() if n >= min_pages}
    if not repeated:
        return 0
    removed = 0
    for page in pages:
        kept = []
        for line in page.page_content.splitlines():
            if line.strip() and _line_key(line) in repeated:
                removed += 1
            else:
                kept.append(line)
        page.page_content = "\n".join(kept)
    return removed


def _chunk_ids(doc_id: str, count: int) -> list[str]:
    return [f"{doc_id}-{i}" for i in range(count)]


def ingest_pdf(path: Path, doc_id: str, filename: str, uploaded_at: str | None = None) -> DocumentRecord:
    settings = get_settings()

    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:  # pypdf raises many different error types
        raise IngestionError(f"Could not read PDF: {exc}") from exc

    if not any(p.page_content.strip() for p in pages):
        raise IngestionError(
            "No extractable text found. The PDF may be scanned images (OCR is not enabled)."
        )

    if settings.strip_repeated_lines:
        strip_repeated_lines(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    raw_chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]

    # Keep only clean, Chroma-compatible metadata (pypdf adds many fields, some None).
    chunks = [
        Document(
            page_content=c.page_content,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "page": int(c.metadata.get("page", 0)) + 1,
                "start_index": int(c.metadata.get("start_index", 0)),
            },
        )
        for c in raw_chunks
    ]
    ids = _chunk_ids(doc_id, len(chunks))

    record = DocumentRecord(
        id=doc_id,
        filename=filename,
        pages=len(pages),
        chunks=len(chunks),
        uploaded_at=uploaded_at or DocumentRecord.now(),
    )

    store = get_vectorstore()
    try:
        for start in range(0, len(chunks), _BATCH_SIZE):
            store.add_documents(
                chunks[start : start + _BATCH_SIZE], ids=ids[start : start + _BATCH_SIZE]
            )
        registry.add_document(record)
    except Exception:
        invalidate_keyword_index()
        # Roll back so no chunks are left in the index without a registry entry
        # (they would be searchable but impossible to delete from the Admin page).
        store.delete(ids=ids)
        raise
    invalidate_keyword_index()
    answer_cache.clear()  # answers may change now that the documents have
    return record


def delete_document(record: DocumentRecord) -> None:
    if record.chunks:
        get_vectorstore().delete(ids=_chunk_ids(record.id, record.chunks))
    (get_settings().upload_dir / f"{record.id}.pdf").unlink(missing_ok=True)
    registry.remove_document(record.id)
    invalidate_keyword_index()
    answer_cache.clear()
