"""Friendly replies to greetings, thanks and "what can you do?" without searching or an LLM call.

Only short messages that are *entirely* small talk match, so "hi, what documents do I need for
KYC?" still goes through the normal pipeline.
"""

import re

_INTRO = (
    "I'm the bank's help assistant. I answer questions about accounts, KYC, Video KYC, fees and "
    "support from the bank's documents, with sources."
)

# Greeting and help replies are followed by clickable example questions in the UI
# (SUGGESTED_QUESTIONS), so the text doesn't list them.
WITH_SUGGESTIONS = {"greeting", "help"}

_REPLIES = {
    "greeting": "Hello! 👋 " + _INTRO + " Try one of these questions, or type your own:",
    "help": _INTRO + " Try one of these questions, or type your own:",
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "goodbye": "Goodbye! Come back any time you have a question about the bank's services.",
}

_FILLER = r"(?:\s+(?:there|again|bot|assistant|team|friend|sir|madam|ma'?am))*"
_PATTERNS = [
    ("greeting", rf"(?:hi+|hello+|hey+|hiya|hola|namaste|namaskar|greetings|"
                 rf"good\s+(?:morning|afternoon|evening|day)){_FILLER}"
                 r"(?:\s+how\s+are\s+you(?:\s+doing)?)?"),
    ("greeting", r"how\s+are\s+you(?:\s+doing)?"),
    ("thanks", r"(?:ok(?:ay)?\s+)?(?:thanks?|thank\s*(?:you|u)|thx|ty|many\s+thanks)"
               r"(?:\s+(?:so|very)\s+much|\s+a\s+lot)?" + _FILLER),
    ("goodbye", r"(?:ok(?:ay)?\s+)?(?:bye+|good\s*bye|see\s+you(?:\s+later)?|take\s+care)" + _FILLER),
    ("help", r"(?:help|what\s+can\s+you\s+do|what\s+do\s+you\s+do|who\s+are\s+you|"
             r"how\s+can\s+you\s+help(?:\s+me)?|what\s+can\s+i\s+ask(?:\s+you)?)"),
]
_COMPILED = [(kind, re.compile(rf"^{p}$")) for kind, p in _PATTERNS]


def _normalize(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text.lower())  # drop punctuation and emoji
    return " ".join(text.split())


def small_talk_reply(text: str) -> tuple[str, str] | None:
    """Return (kind, reply) if the whole message is small talk, else None."""
    normalized = _normalize(text)
    if not normalized or len(normalized) > 60:
        return None
    for kind, pattern in _COMPILED:
        if pattern.match(normalized):
            return kind, _REPLIES[kind]
    return None
