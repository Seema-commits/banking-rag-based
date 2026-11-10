import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api import admin, chat
from app.config import get_settings
from app.rag.vectorstore import get_vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
# Silence per-request HTTP logs from the Hugging Face / Groq clients.
for noisy in ("httpx", "httpx2", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


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
    logger.info("Ready")
    yield


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
