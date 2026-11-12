"""Small in-memory rate limiter for the public chat endpoint.

Good enough for a single-instance demo (the Cloud Run guide sets max instances to 1).
With several instances each one keeps its own counters.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from fastapi import HTTPException, Request, status

from app.config import get_settings
from app.observability import log_event

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_per_ip: dict[str, deque[float]] = defaultdict(deque)
_day = {"date": "", "count": 0}


def _client_ip(request: Request) -> str:
    # Behind Cloud Run's proxy the client address is in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def limit_chat(request: Request) -> None:
    settings = get_settings()
    now = time.monotonic()
    today = datetime.now(timezone.utc).date().isoformat()
    ip = _client_ip(request)

    with _lock:
        per_minute = settings.chat_requests_per_minute_per_ip
        if per_minute > 0:
            hits = _per_ip[ip]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= per_minute:
                log_event(logger, "rate_limited", level=logging.WARNING, scope="per_ip_minute")
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "You're sending messages too quickly. Please wait a minute and try again.",
                )

        per_day = settings.chat_requests_per_day
        if _day["date"] != today:
            _day["date"], _day["count"] = today, 0
        if per_day > 0 and _day["count"] >= per_day:
            log_event(logger, "rate_limited", level=logging.WARNING, scope="daily")
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "This demo has reached its daily question limit. Please try again tomorrow.",
            )

        if per_minute > 0:
            _per_ip[ip].append(now)
        _day["count"] += 1
