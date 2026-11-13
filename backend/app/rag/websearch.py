"""Web search tool (Tavily), limited to an allowlist of official websites.

Used only as a fallback, when the bank's own documents don't answer a banking question. Results
are treated as untrusted data: restricted to allowlisted domains (by the API and checked again
here), trimmed, and passed to the LLM inside the same <documents> block as uploaded documents,
whose instructions the model is told to ignore.

Enabled when WEB_SEARCH_ENABLED is true and TAVILY_API_KEY is set.
"""

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
_MAX_CHARS_PER_RESULT = 1500


@dataclass
class WebResult:
    url: str
    title: str
    content: str


def web_search_available() -> bool:
    settings = get_settings()
    return settings.web_search_enabled and bool(settings.tavily_api_key) and bool(settings.web_search_domains)


def _allowed(url: str, domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def web_search(query: str) -> list[WebResult]:
    """Search the allowlisted sites. Returns [] on any error, so the bot falls back to "not found"."""
    settings = get_settings()
    domains = [d.lower().strip() for d in settings.web_search_domains if d.strip()]
    try:
        response = httpx.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={
                "query": query,
                "search_depth": "basic",  # 1 credit per search
                "include_domains": domains,
                "max_results": settings.web_search_max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=15,
        )
        response.raise_for_status()
        items = response.json().get("results") or []
    except Exception as exc:  # network error, bad key, quota, unexpected JSON
        logger.warning("Web search failed: %s", type(exc).__name__)
        return []

    results = []
    for item in items:
        url, content = str(item.get("url") or ""), str(item.get("content") or "").strip()
        if url and content and _allowed(url, domains):
            title = str(item.get("title") or urlparse(url).hostname or url)[:200]
            results.append(WebResult(url=url, title=title, content=content[:_MAX_CHARS_PER_RESULT]))
    return results
