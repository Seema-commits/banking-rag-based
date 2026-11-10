import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.deps import require_admin
from app.config import get_settings
from app.rag import registry
from app.rag.ingestion import IngestionError, delete_document, ingest_pdf
from app.rag.registry import DocumentRecord

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)]
)


class FailedUpload(BaseModel):
    filename: str
    error: str


class UploadResult(BaseModel):
    uploaded: list[DocumentRecord]
    failed: list[FailedUpload]


@router.get("/verify")
def verify() -> dict:
    """Lets the frontend check an admin key."""
    return {"ok": True}


@router.get("/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    return registry.list_documents()


def _save_upload(file: UploadFile, dest: Path, max_bytes: int) -> None:
    size = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise IngestionError(f"File exceeds {max_bytes // (1024 * 1024)} MB limit")
            out.write(chunk)
    if size == 0:
        raise IngestionError("File is empty")


# Sync endpoint: FastAPI runs it in a worker thread, so CPU-heavy
# embedding doesn't block the event loop.
@router.post("/documents", response_model=UploadResult)
def upload_documents(files: list[UploadFile] = File(...)) -> UploadResult:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    result = UploadResult(uploaded=[], failed=[])
    for file in files:
        filename = Path(file.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            result.failed.append(FailedUpload(filename=filename, error="Only PDF files are supported"))
            continue

        doc_id = uuid.uuid4().hex
        dest = settings.upload_dir / f"{doc_id}.pdf"
        try:
            _save_upload(file, dest, max_bytes)
            result.uploaded.append(ingest_pdf(dest, doc_id, filename))
        except IngestionError as exc:
            dest.unlink(missing_ok=True)
            result.failed.append(FailedUpload(filename=filename, error=str(exc)))
        except Exception:
            dest.unlink(missing_ok=True)
            logger.exception("Failed to ingest %s", filename)
            result.failed.append(FailedUpload(filename=filename, error="Unexpected error while indexing"))
    return result


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(doc_id: str) -> None:
    record = registry.get_document(doc_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    delete_document(record)
