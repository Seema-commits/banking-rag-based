"""Keep uploaded PDFs in Cloud Storage so they survive Cloud Run restarts.

When DOCUMENTS_BUCKET is set:
  - an Admin upload is saved as gs://<bucket>/documents/<doc_id>.pdf (original filename stored as
    metadata) after it has been indexed;
  - deleting a document in Admin deletes that object too;
  - at startup, every PDF in the bucket that isn't indexed yet is downloaded and re-indexed.

The vector index itself stays on the container's disk and is rebuilt from these files, which keeps
the setup simple and cheap for a small knowledge base.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "documents/"


@dataclass
class SavedDocument:
    doc_id: str
    filename: str
    uploaded_at: str | None


def enabled() -> bool:
    return bool(get_settings().documents_bucket)


@lru_cache
def _bucket():
    from google.cloud import storage  # imported here: only needed when a bucket is configured

    return storage.Client().bucket(get_settings().documents_bucket)


def save(doc_id: str, path: Path, filename: str) -> None:
    blob = _bucket().blob(f"{_PREFIX}{doc_id}.pdf")
    blob.metadata = {"filename": filename}
    blob.upload_from_filename(str(path), content_type="application/pdf")


def delete(doc_id: str) -> None:
    from google.api_core.exceptions import NotFound

    try:
        _bucket().blob(f"{_PREFIX}{doc_id}.pdf").delete()
    except NotFound:
        pass


def list_saved() -> list[SavedDocument]:
    saved = []
    for blob in _bucket().list_blobs(prefix=_PREFIX):
        name = blob.name[len(_PREFIX):]
        if not name.endswith(".pdf"):
            continue
        created = getattr(blob, "time_created", None)
        saved.append(SavedDocument(
            doc_id=name[:-4],
            filename=(blob.metadata or {}).get("filename") or name,
            uploaded_at=created.isoformat() if created else None,
        ))
    return saved


def download(doc_id: str, dest: Path) -> None:
    _bucket().blob(f"{_PREFIX}{doc_id}.pdf").download_to_filename(str(dest))
