"""Mask personal and financial identifiers in user messages.

Runs before anything is sent to the LLM provider, the vector search or the logs, so a user who
types their Aadhaar, PAN, card or account number never has it leave the server.

Patterns are tuned for India (Aadhaar, PAN, +91 mobiles, UPI IDs) plus common international
formats (emails, payment cards checked with the Luhn algorithm). Each match is replaced with a
placeholder such as [AADHAAR], so the LLM still understands what kind of detail was shared.
"""

import re
from dataclasses import dataclass, field

# Order matters: more specific patterns run first, so a 16-digit card number isn't also
# reported as an account number, and an email isn't also read as a UPI ID.

_SECRET_KEYWORDS = r"otp|one[\s-]?time[\s-]?password|m?pin|cvv|cvc|password|passcode|passwd"
# The value must contain a digit, so questions like "will you ask for my OTP during the call?"
# are left alone while "my OTP is 482913" is masked.
_SECRET = re.compile(
    rf"\b(?P<label>{_SECRET_KEYWORDS})\b(?P<sep>\s*(?:is|:|=|-)?\s*)"
    r"(?P<value>(?=[^\s,.;]*\d)[^\s,.;]{3,})",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_UPI = re.compile(r"\b[\w.-]{2,}@[a-zA-Z]{2,}\b")
# A number must not continue into more digits (even after a space or dash), so one long
# number is never split into a shorter "Aadhaar" or "phone" match.
_START = r"(?<![\d-])(?<!\d[ -])"
_END = r"(?![\d-]|[ -]\d)"
_CARD = re.compile(_START + r"(?:\d[ -]?){12,18}\d" + _END)
_AADHAAR = re.compile(_START + r"[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}" + _END)
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<![\d-])(?<!\d[ -])(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}" + _END)
_ACCOUNT = re.compile(_START + r"\d{9,18}" + _END)


@dataclass
class MaskResult:
    text: str
    found: list[str] = field(default_factory=list)  # placeholder labels, in order found

    @property
    def masked(self) -> bool:
        return bool(self.found)


def _luhn_ok(digits: str) -> bool:
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        double = not double
    return total % 10 == 0


def mask_pii(text: str) -> MaskResult:
    found: list[str] = []

    def sub(pattern: re.Pattern, label: str, s: str, check=None) -> str:
        def repl(m: re.Match) -> str:
            if check and not check(m):
                return m.group(0)
            found.append(label)
            return f"[{label}]"

        return pattern.sub(repl, s)

    def secret_repl(m: re.Match) -> str:
        found.append("SECRET")
        return f"{m.group('label')}{m.group('sep')}[SECRET]"

    text = _SECRET.sub(secret_repl, text)
    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_UPI, "UPI_ID", text)
    text = sub(_CARD, "CARD_NUMBER", text,
               check=lambda m: _luhn_ok(re.sub(r"\D", "", m.group(0))))
    text = sub(_AADHAAR, "AADHAAR", text)
    text = sub(_PAN, "PAN", text)
    text = sub(_PHONE, "PHONE", text)
    text = sub(_ACCOUNT, "ACCOUNT_NUMBER", text)
    return MaskResult(text=text, found=found)


_FRIENDLY = {
    "SECRET": "OTP/PIN/password",
    "EMAIL": "email address",
    "UPI_ID": "UPI ID",
    "CARD_NUMBER": "card number",
    "AADHAAR": "Aadhaar number",
    "PAN": "PAN",
    "PHONE": "phone number",
    "ACCOUNT_NUMBER": "account number",
}


def privacy_notice(labels: list[str]) -> str | None:
    """A short message telling the user what was removed, or None if nothing was."""
    if not labels:
        return None
    kinds = list(dict.fromkeys(_FRIENDLY[label] for label in labels))
    listed = kinds[0] if len(kinds) == 1 else ", ".join(kinds[:-1]) + " and " + kinds[-1]
    return (
        f"For your safety, your {listed} {'was' if len(kinds) == 1 else 'were'} removed "
        "before your message was processed. "
        "Never share OTPs, PINs, passwords or full account details in a chat."
    )
