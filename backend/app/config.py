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
    # Question/answer log and feedback for the admin Insights tab (questions are PII-masked).
    store_interactions: bool = True
    interactions_db: Path = BASE_DIR / "storage" / "interactions.db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Drop lines that repeat on most pages of a PDF (headers, footers, "Page N") before chunking,
    # so they don't add noise to every chunk.
    strip_repeated_lines: bool = True
    max_upload_mb: int = 25

    # Retrieval
    top_k: int = 4
    # "vector" (embeddings only) or "hybrid" (embeddings + BM25 keyword search, merged with
    # reciprocal rank fusion). Hybrid catches exact terms such as "closure", "Form 60", "CKYC".
    retrieval_mode: str = "hybrid"
    # If no chunk scores at least this (0 = unrelated, 1 = identical), the question gets the
    # "not found" answer without calling the LLM. Tuned for recall (eval: 0.15 keeps 100% of
    # real questions); the LLM grader handles the harder out-of-scope cases.
    relevance_threshold: float = 0.15
    max_history_messages: int = 10

    # Agent / self-correction (each LLM check adds one Groq call)
    self_correction: bool = True  # LLM grades retrieved chunks and decides: answer, retry or refuse
    grounding_check: bool = True  # LLM verifies the answer's claims against the sources
    max_query_rewrites: int = 1  # retries with a reworded search when the chunks don't answer
    max_regenerations: int = 1  # retries when the answer has unsupported claims
    # Model for the agent's internal checks (grading, rewriting, grounding check). A small fast
    # model makes these much quicker; the customer-facing answer always uses GROQ_MODEL.
    # Empty = use GROQ_MODEL for everything. Falls back to GROQ_MODEL if this model fails.
    check_model: str = "llama-3.1-8b-instant"
    # Reasoning effort for checks on reasoning models (gpt-oss, qwen3): "low" is faster.
    check_reasoning_effort: str = "low"

    # Web search tool (Tavily): used only when the documents don't answer a banking question,
    # and only on these official sites. Active when enabled AND a TAVILY_API_KEY is set.
    web_search_enabled: bool = True
    tavily_api_key: str = ""
    web_search_domains: list[str] = ["rbi.org.in", "uidai.gov.in", "incometax.gov.in", "npci.org.in"]
    web_search_max_results: int = 4
    # Speed: with the web tool on, skip the query-rewrite retry and go straight to the web, and
    # don't re-grade web results (they come only from allowlisted sites and the answer is still
    # grounding-checked). Set these to true for the slower, more cautious path.
    rewrite_before_web: bool = False
    grade_web_results: bool = False

    # Server (used by run.py; Cloud Run and similar hosts set PORT for you)
    host: str = "127.0.0.1"
    port: int = 8731
    reload: bool = True  # auto-restart on code changes; the Docker image sets RELOAD=false
    # "text" for readable local logs, "json" for structured logs (Cloud Logging); Docker uses json.
    log_format: str = "text"

    # Built React app served by FastAPI at "/" when present (production/Docker).
    # In development the Vite dev server serves the frontend instead.
    static_dir: Path = BASE_DIR.parent / "frontend" / "dist"
    # PDFs indexed automatically at startup when the knowledge base is empty
    # (the Docker image points this at sample-docs/thistlewood). Empty = off.
    seed_docs_dir: Path | None = None

    # Example questions shown as clickable buttons on the empty chat screen and after a greeting.
    suggested_questions: list[str] = [
        "What documents do I need to open an account?",
        "Can I do Video KYC on a Sunday?",
        "How often do I need to update my KYC?",
        "What is the minimum balance for a savings account?",
    ]

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
