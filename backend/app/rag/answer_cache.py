"""In-memory cache of finished answers, so a repeated question is answered instantly.

The key is the standalone question: the PII-masked question after the agent has rewritten any
follow-up ("what about NRIs?") using the conversation, so the same words in different contexts
don't share an answer. It is normalised for case, spacing and trailing punctuation.

Entries expire after ANSWER_CACHE_TTL_SECONDS, the oldest are dropped beyond
ANSWER_CACHE_MAX_ENTRIES, everything is cleared when documents are uploaded or deleted, and an
answer that gets a thumbs-down is removed.

Per process and lost on restart, which suits a single-instance Cloud Run demo.
"""

import copy
import re
import threading
import time
from collections import OrderedDict

from app.config import get_settings

_lock = threading.Lock()
_entries: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_key_by_message: "OrderedDict[str, str]" = OrderedDict()  # message_id -> cache key, for 👎 eviction
_MAX_MESSAGE_LINKS = 5000


def cache_key(question: str) -> str:
    text = " ".join(question.lower().split())
    return re.sub(r"[\s?.!]+$", "", text)


def get(question: str) -> dict | None:
    settings = get_settings()
    if not settings.answer_cache_enabled:
        return None
    key = cache_key(question)
    with _lock:
        entry = _entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.time() - stored_at > settings.answer_cache_ttl_seconds:
            del _entries[key]
            return None
        _entries.move_to_end(key)
        return copy.deepcopy(value)


def put(question: str, value: dict) -> None:
    settings = get_settings()
    if not settings.answer_cache_enabled:
        return
    with _lock:
        _entries[cache_key(question)] = (time.time(), copy.deepcopy(value))
        _entries.move_to_end(cache_key(question))
        while len(_entries) > settings.answer_cache_max_entries:
            _entries.popitem(last=False)


def link_message(message_id: str, question: str) -> None:
    """Remember which cache entry a shown answer came from, so 👎 feedback can evict it."""
    with _lock:
        _key_by_message[message_id] = cache_key(question)
        while len(_key_by_message) > _MAX_MESSAGE_LINKS:
            _key_by_message.popitem(last=False)


def evict_message(message_id: str) -> bool:
    with _lock:
        key = _key_by_message.pop(message_id, None)
        return key is not None and _entries.pop(key, None) is not None


def clear() -> None:
    """Called when the knowledge base changes."""
    with _lock:
        _entries.clear()


def size() -> int:
    with _lock:
        return len(_entries)
