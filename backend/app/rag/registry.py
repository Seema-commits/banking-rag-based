"""A small JSON registry of uploaded documents, so admins can list and delete them."""

import json
import threading
from datetime import datetime, timezone

from pydantic import BaseModel

from app.config import get_settings

_lock = threading.Lock()


class DocumentRecord(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: str

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()


def _load() -> dict[str, dict]:
    path = get_settings().registry_path
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(data: dict[str, dict]) -> None:
    path = get_settings().registry_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_documents() -> list[DocumentRecord]:
    with _lock:
        records = [DocumentRecord(**r) for r in _load().values()]
    return sorted(records, key=lambda r: r.uploaded_at, reverse=True)


def get_document(doc_id: str) -> DocumentRecord | None:
    with _lock:
        raw = _load().get(doc_id)
    return DocumentRecord(**raw) if raw else None


def add_document(record: DocumentRecord) -> None:
    with _lock:
        data = _load()
        data[record.id] = record.model_dump()
        _save(data)


def remove_document(doc_id: str) -> None:
    with _lock:
        data = _load()
        data.pop(doc_id, None)
        _save(data)
