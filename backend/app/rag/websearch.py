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

from app import usage
from app.config import get_settings
from app.observability import log_event

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


def failure_reason(exc: Exception) -> str:
    """auth (bad or missing key), quota (credits or rate limit), timeout, network, http_<code>,
    or bad_response, so the owner alert can say what to fix."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in (401, 403):
            return "auth"
        if code in (402, 429, 432, 433):  # Tavily uses 432/433 for plan and usage limits
            return "quota"
        return f"http_{code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "network"
    return "bad_response"


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
        reason = failure_reason(exc)
        log_event(logger, "web_search_failed",
                  level=logging.ERROR if reason in ("auth", "quota") else logging.WARNING,
                  reason=reason, error_type=type(exc).__name__,
                  status=getattr(getattr(exc, "response", None), "status_code", None))
        return []
    finally:
        usage.add("web_searches")

    results = []
    for item in items:
        url, content = str(item.get("url") or ""), str(item.get("content") or "").strip()
        if url and content and _allowed(url, domains):
            title = str(item.get("title") or urlparse(url).hostname or url)[:200]
            results.append(WebResult(url=url, title=title, content=content[:_MAX_CHARS_PER_RESULT]))
    return results
