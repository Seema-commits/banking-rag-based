import logging
from contextlib import asynccontextmanager

import shutil
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api import admin, chat
from app.config import get_settings
from app.observability import configure_logging, log_event
from app.rag import registry
from app.rag.ingestion import ingest_pdf
from app.rag.vectorstore import get_vectorstore

configure_logging(get_settings().log_format)
logger = logging.getLogger(__name__)


def seed_documents() -> None:
    """Index the PDFs in SEED_DOCS_DIR when the knowledge base is empty.

    Cloud Run's disk is wiped on every restart, so the demo documents are re-indexed
    each time a new instance starts. Documents uploaded later from the Admin page
    last until the next restart.
    """
    settings = get_settings()
    folder = settings.seed_docs_dir
    if not folder or registry.list_documents():
        return
    pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
    if not pdfs:
        logger.warning("SEED_DOCS_DIR %s has no PDFs", folder)
        return
    start = time.perf_counter()
    seeded = chunks = 0
    for pdf in pdfs:
        doc_id = uuid.uuid4().hex
        dest = settings.upload_dir / f"{doc_id}.pdf"
        shutil.copy(pdf, dest)
        try:
            record = ingest_pdf(dest, doc_id, pdf.name)
            seeded, chunks = seeded + 1, chunks + record.chunks
            logger.info("Seeded %s (%d chunks)", pdf.name, record.chunks)
        except Exception:
            dest.unlink(missing_ok=True)
            logger.exception("Failed to seed %s", pdf.name)
    log_event(logger, "seed", documents=seeded, chunks=chunks,
              duration_ms=round((time.perf_counter() - start) * 1000))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set; /api/chat will return 503")
    if not settings.admin_api_key:
        logger.warning("ADMIN_API_KEY is not set; admin endpoints are disabled")
    # Load the embedding model and open Chroma up front, so the first request isn't slow.
    # The first run downloads the model from Hugging Face.
    logger.info("Loading embedding model %s ...", settings.embedding_model)
    start = time.perf_counter()
    await run_in_threadpool(get_vectorstore)
    await run_in_threadpool(seed_documents)
    log_event(logger, "startup", duration_ms=round((time.perf_counter() - start) * 1000),
              embedding_model=settings.embedding_model, llm_model=settings.groq_model,
              top_k=settings.top_k, relevance_threshold=settings.relevance_threshold,
              documents=len(registry.list_documents()))
    logger.info("Ready")
    yield


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_origin_regex=get_settings().cors_origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Serve the built React app (frontend/dist) at "/" when it exists. Registered last so the
# /api routes above take priority. In development, Vite serves the frontend instead.
if get_settings().static_dir.is_dir():
    app.mount("/", StaticFiles(directory=get_settings().static_dir, html=True), name="frontend")
