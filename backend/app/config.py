from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (Groq)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0

    # Embeddings (Hugging Face, runs locally)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Vector store (Chroma)
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"
    collection_name: str = "documents"

    # Ingestion
    upload_dir: Path = BASE_DIR / "storage" / "uploads"
    registry_path: Path = BASE_DIR / "storage" / "documents.json"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_mb: int = 25

    # Retrieval
    top_k: int = 4
    # Chunks scoring below this (0 = unrelated, 1 = identical) are dropped, so off-topic
    # questions get the "not found" answer. 0 disables the filter. Tune with the eval set.
    relevance_threshold: float = 0.3
    max_history_messages: int = 10

    # Server (used by run.py; Cloud Run and similar hosts set PORT for you)
    host: str = "127.0.0.1"
    port: int = 8731
    reload: bool = True  # auto-restart on code changes; the Docker image sets RELOAD=false

    # Built React app served by FastAPI at "/" when present (production/Docker).
    # In development the Vite dev server serves the frontend instead.
    static_dir: Path = BASE_DIR.parent / "frontend" / "dist"
    # PDFs indexed automatically at startup when the knowledge base is empty
    # (the Docker image points this at sample-docs/thistlewood). Empty = off.
    seed_docs_dir: Path | None = None

    # Abuse protection for the public chat endpoint (0 disables a limit).
    chat_requests_per_minute_per_ip: int = 10
    chat_requests_per_day: int = 500  # across all users, protects the Groq quota

    # Security / API
    admin_api_key: str = ""
    # Extra allowed frontend origins, e.g. a deployed site. Any localhost port is always allowed.
    cors_origins: list[str] = []
    cors_origin_regex: str = r"http://(localhost|127\.0\.0\.1)(:\d+)?"


@lru_cache
def get_settings() -> Settings:
    return Settings()
