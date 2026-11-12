"""Structured logging and per-request metrics.

- configure_logging("json") prints one JSON object per log line. Google Cloud Logging reads
  these as structured entries (jsonPayload), so fields like latency_ms can be filtered,
  charted as log-based metrics and used in alerts. "text" is easier to read locally.
- start_request() / record() / add() / timed() collect numbers for the current request
  (step timings, retrieval scores, token usage) from anywhere in the pipeline, and
  log_event() writes them out as one line.

Never log question text, answers or document contents here: only counts, timings and flags.
"""

import contextvars
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone

_metrics: contextvars.ContextVar[dict | None] = contextvars.ContextVar("request_metrics", default=None)


def start_request() -> dict:
    """Start collecting metrics for the current request and return the (mutable) dict."""
    metrics: dict = {}
    _metrics.set(metrics)
    return metrics


def record(**fields) -> None:
    metrics = _metrics.get()
    if metrics is not None:
        metrics.update(fields)


def add(key: str, amount: float) -> None:
    metrics = _metrics.get()
    if metrics is not None:
        metrics[key] = metrics.get(key, 0) + amount


@contextmanager
def timed(step: str):
    """Add the elapsed milliseconds of the block to `<step>_ms` (summed if repeated)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        add(f"{step}_ms", round((time.perf_counter() - start) * 1000, 1))


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    """Write one structured log line: {"event": event, **fields}."""
    logger.log(level, event, extra={"fields": {"event": event, **fields}})


class JsonFormatter(logging.Formatter):
    """One JSON object per line, using the field names Cloud Logging understands."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            # Cloud Error Reporting groups errors whose message contains the stack trace.
            message = f"{message}\n{self.formatException(record.exc_info)}"
        payload = {
            "severity": record.levelname,
            "message": message,
            "logger": record.name,
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
        }
        payload.update(getattr(record, "fields", None) or {})
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Readable local format; structured fields are appended as key=value."""

    def __init__(self):
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        fields = {k: v for k, v in (getattr(record, "fields", None) or {}).items() if k != "event"}
        if fields:
            text += " | " + " ".join(f"{k}={v}" for k, v in fields.items())
        return text


def configure_logging(log_format: str = "text") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Silence per-request HTTP logs from the Hugging Face / Groq clients.
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
