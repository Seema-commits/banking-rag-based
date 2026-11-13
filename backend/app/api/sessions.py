"""Chat history kept on the server.

The browser only holds a random session id. History is never accepted from the client, so a
caller can't plant fake "assistant" messages to steer the model (a prompt-injection route).
Stored turns are already PII-masked. In memory, bounded, and lost on restart: fine for a
single-instance demo.
"""

import threading
import time
import uuid
from collections import OrderedDict, deque

MAX_SESSIONS = 2000
SESSION_TTL_SECONDS = 2 * 60 * 60

_lock = threading.Lock()
_sessions: "OrderedDict[str, tuple[float, deque]]" = OrderedDict()


def _valid(session_id: str | None) -> bool:
    try:
        return session_id is not None and uuid.UUID(session_id).hex == session_id.replace("-", "")
    except ValueError:
        return False


def get_history(session_id: str | None, max_messages: int) -> tuple[str, list[dict]]:
    """Return (session_id, recent turns). Unknown or invalid ids start a new session."""
    now = time.time()
    with _lock:
        for sid in [s for s, (seen, _) in _sessions.items() if now - seen > SESSION_TTL_SECONDS]:
            del _sessions[sid]
        if _valid(session_id) and session_id in _sessions:
            _, turns = _sessions[session_id]
            _sessions.move_to_end(session_id)
            return session_id, list(turns)[-max_messages:]
        return str(uuid.uuid4()), []


def append_turns(session_id: str, question: str, answer: str, max_messages: int) -> None:
    with _lock:
        _, turns = _sessions.get(session_id, (0.0, deque(maxlen=max_messages)))
        turns.append({"role": "user", "content": question})
        turns.append({"role": "assistant", "content": answer})
        _sessions[session_id] = (time.time(), turns)
        _sessions.move_to_end(session_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def reset() -> None:
    with _lock:
        _sessions.clear()
