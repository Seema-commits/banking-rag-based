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

    # Security / API
    admin_api_key: str = ""
    cors_origins: list[str] = ["http://localhost:5175"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
