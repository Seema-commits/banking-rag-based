import logging
from contextlib import asynccontextmanager

import shutil
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api import admin, chat
from app.config import get_settings
from app.rag import registry
from app.rag.ingestion import ingest_pdf
from app.rag.vectorstore import get_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
# Silence per-request HTTP logs from the Hugging Face / Groq clients.
for noisy in ("httpx", "httpx2", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


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
    for pdf in pdfs:
        doc_id = uuid.uuid4().hex
        dest = settings.upload_dir / f"{doc_id}.pdf"
        shutil.copy(pdf, dest)
        try:
            record = ingest_pdf(dest, doc_id, pdf.name)
            logger.info("Seeded %s (%d chunks)", pdf.name, record.chunks)
        except Exception:
            dest.unlink(missing_ok=True)
            logger.exception("Failed to seed %s", pdf.name)


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
    await run_in_threadpool(get_vectorstore)
    await run_in_threadpool(seed_documents)
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
