"""Daily usage counters with threshold warnings, for the owner alerts.

Google Cloud's budget alert only sees Google Cloud costs. Groq tokens and Tavily searches are
billed (or rate-limited) by those providers, so the app counts its own daily usage and writes a
`usage_threshold` log event the first time each counter reaches 80% and 100% of its limit. A
Cloud Monitoring alert turns that event into an email (see ops/alerts/).

Counters reset at midnight UTC and live in memory (one Cloud Run instance), so they are an
early warning, not an exact bill.
"""

import logging
import threading
from datetime import datetime, timezone

from app.config import get_settings
from app.observability import log_event

logger = logging.getLogger(__name__)

WARN_AT = (0.8, 1.0)

_lock = threading.Lock()
_state: dict = {"date": "", "counts": {}, "warned": set()}


def _limits() -> dict[str, int]:
    s = get_settings()
    return {
        "questions": s.chat_requests_per_day,
        "llm_tokens": s.daily_llm_token_warning,
        "web_searches": s.daily_web_search_warning,
    }


def add(metric: str, amount: int = 1) -> None:
    """Count usage; log a usage_threshold event when a limit's 80% or 100% mark is first crossed."""
    if amount <= 0:
        return
    today = datetime.now(timezone.utc).date().isoformat()
    events = []
    with _lock:
        if _state["date"] != today:
            _state.update(date=today, counts={}, warned=set())
        before = _state["counts"].get(metric, 0)
        after = before + amount
        _state["counts"][metric] = after
        limit = _limits().get(metric, 0)
        if limit > 0:
            for level in WARN_AT:
                mark = level * limit
                if before < mark <= after and (metric, level) not in _state["warned"]:
                    _state["warned"].add((metric, level))
                    events.append(level)
    for level in events:
        log_event(logger, "usage_threshold", level=logging.WARNING if level < 1 else logging.ERROR,
                  metric=metric, used=after, limit=limit, percent=int(level * 100))


def snapshot() -> dict:
    """Today's counts and limits (for the admin screen)."""
    with _lock:
        counts = dict(_state["counts"]) if _state["date"] == datetime.now(timezone.utc).date().isoformat() else {}
    return {metric: {"used": counts.get(metric, 0), "limit": limit} for metric, limit in _limits().items()}


def reset() -> None:
    with _lock:
        _state.update(date="", counts={}, warned=set())
