"""Spot questions about a different bank than the one whose documents the assistant has.

The documents only describe one bank (BANK_NAME). A question such as "minimum balance for a
kids account at HDFC" can't be answered from them, and the government websites the web tool
searches don't list one bank's fees either. Without this check the agent searches, writes a
draft, fails the fact-check, retries and only then refuses: about 30 seconds for an unhelpful
"couldn't verify". Recognising the other bank up front lets the agent say so clearly and fast,
and still answer the part of the question its own documents cover.

A fixed list is used rather than an LLM call: it's instant, free and predictable, and new names
can be added in OTHER_BANK_NAMES without touching the code.
"""

import re
from functools import lru_cache

# Display name -> other ways people write it. Matched as whole words, ignoring case.
KNOWN_BANKS: dict[str, list[str]] = {
    "HDFC Bank": ["hdfc"],
    "ICICI Bank": ["icici"],
    "State Bank of India": ["sbi", "state bank of india", "state bank"],
    "Axis Bank": ["axis bank"],
    "Kotak Mahindra Bank": ["kotak"],
    "Punjab National Bank": ["pnb", "punjab national bank"],
    "Bank of Baroda": ["bank of baroda"],
    "Canara Bank": ["canara"],
    "Union Bank of India": ["union bank"],
    "Bank of India": ["bank of india"],
    "Indian Bank": ["indian bank"],
    "Indian Overseas Bank": ["indian overseas bank", "iob"],
    "Central Bank of India": ["central bank of india"],
    "UCO Bank": ["uco bank"],
    "Bank of Maharashtra": ["bank of maharashtra"],
    "IDFC FIRST Bank": ["idfc"],
    "IDBI Bank": ["idbi"],
    "Yes Bank": ["yes bank"],
    "IndusInd Bank": ["indusind"],
    "Federal Bank": ["federal bank"],
    "RBL Bank": ["rbl"],
    "South Indian Bank": ["south indian bank"],
    "Karur Vysya Bank": ["karur vysya", "kvb"],
    "Bandhan Bank": ["bandhan"],
    "AU Small Finance Bank": ["au small finance", "au bank"],
    "Equitas Small Finance Bank": ["equitas"],
    "Ujjivan Small Finance Bank": ["ujjivan"],
    "Airtel Payments Bank": ["airtel payments bank"],
    "India Post Payments Bank": ["india post payments bank", "ippb"],
    "Paytm Payments Bank": ["paytm payments bank"],
    "Fino Payments Bank": ["fino"],
    "Jio Payments Bank": ["jio payments bank"],
    "Citibank": ["citibank", "citi bank"],
    "HSBC": ["hsbc"],
    "Standard Chartered": ["standard chartered"],
    "DBS Bank": ["dbs"],
    "Deutsche Bank": ["deutsche bank"],
    "Barclays": ["barclays"],
}


# Regulators and institutions whose names contain a bank's name ("Reserve Bank of India" contains
# "Bank of India"). They're about rules every bank follows, so they're blanked out before matching.
NOT_BANKS = re.compile(r"reserve\s+bank(?:\s+of\s+india)?|world\s+bank|national\s+housing\s+bank|nabard",
                       re.IGNORECASE)


@lru_cache(maxsize=8)
def _patterns(own_bank: str, extra: tuple[str, ...]) -> list[tuple[str, re.Pattern]]:
    own = own_bank.lower()
    banks = dict(KNOWN_BANKS)
    for name in extra:
        name = name.strip()
        if name:
            banks.setdefault(name, [name.lower()])
    result = []
    for display, aliases in banks.items():
        names = {display.lower(), *aliases}
        if own and (own in names or display.lower() == own):
            continue  # never treat the assistant's own bank as "another bank"
        # Longest first, so "bank of india" isn't cut short by a shorter alias.
        alternatives = "|".join(re.escape(a) for a in sorted(names, key=len, reverse=True))
        result.append((display, re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w-])", re.IGNORECASE)))
    return result


def find_other_bank(text: str, own_bank: str, extra_names: list[str] | tuple[str, ...] = ()) -> str | None:
    """Name of the first other bank mentioned in the text, or None."""
    if not text:
        return None
    text = NOT_BANKS.sub(lambda m: " " * len(m.group()), text)
    best: tuple[int, str] | None = None
    for display, pattern in _patterns(own_bank, tuple(extra_names)):
        match = pattern.search(text)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), display)
    return best[1] if best else None
