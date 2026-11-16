"""Owner dashboard data: health, trends, where the time goes, problems and suggested fixes.

Computed from the audit log (app.insights.store) for a time window, so it works the same with
SQLite locally and Firestore on Cloud Run. Everything here is numbers and short labels; the
question text appears only in the problem lists, already PII-masked.
"""

import csv
import io
from datetime import datetime, timedelta, timezone

from app import usage
from app.config import get_settings
from app.insights import store

WINDOW_LIMIT = 2000  # most recent records considered per view

ANSWERED = {"answered", "answered_web", "small_talk"}
ERRORS_PREFIX = "llm_"

STEP_LABELS = {
    "rewrite": "Follow-up rewrite", "retrieval": "Document search", "grade": "Relevance check",
    "query_rewrite": "Search retry", "web_search": "Web search", "generate": "Writing the answer",
    "grounding": "Fact-check",
}

ERROR_FIXES = {
    "llm_auth_error": "Groq rejected the API key. Create a new key at console.groq.com/keys and add it as "
                      "a new version of the groq-api-key secret (docs/alerts.md).",
    "llm_rate_limited": "Groq rate limit or quota reached. Check usage at console.groq.com; lower "
                        "CHAT_REQUESTS_PER_DAY or upgrade the plan.",
    "llm_not_configured": "GROQ_API_KEY isn't set on the server. Add the groq-api-key secret to the service.",
    "error": "Unexpected failure. Click View to see its error type, then check the service logs.",
}


def is_error(outcome: str) -> bool:
    return outcome == "error" or outcome.startswith(ERRORS_PREFIX)


def group(outcome: str) -> str:
    if outcome in ANSWERED:
        return "answered"
    return "error" if is_error(outcome) else "not_answered"


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def _slowest_step(record: dict) -> tuple[str, float] | None:
    timings = {k: v for k, v in (record.get("timings_ms") or {}).items() if k in STEP_LABELS and v}
    if not timings:
        return None
    step = max(timings, key=timings.get)
    return step, timings[step]


def _brief(record: dict) -> dict:
    slow = _slowest_step(record)
    return {
        "id": record["id"], "created_at": record["created_at"], "question": record["question"],
        "outcome": record["outcome"], "latency_ms": record.get("latency_ms"),
        "slowest_step": STEP_LABELS.get(slow[0]) if slow else None,
        "slowest_step_ms": slow[1] if slow else None,
        "rating": record.get("rating"), "comment": record.get("comment"),
        "error_type": record.get("error_type"),
    }


def _buckets(records: list[dict], start: datetime, end: datetime, hours: int) -> list[dict]:
    step = timedelta(hours=1) if hours <= 48 else timedelta(days=1)
    if step == timedelta(days=1):
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start = start.replace(minute=0, second=0, microsecond=0)
    buckets, t = [], start
    while t <= end:
        buckets.append({"start": t, "answered": 0, "not_answered": 0, "error": 0, "latencies": []})
        t += step
    for r in records:
        idx = int((_parse(r["created_at"]) - start) / step)
        if 0 <= idx < len(buckets):
            b = buckets[idx]
            b[group(r["outcome"])] += 1
            if r.get("latency_ms") is not None and not r.get("cached"):
                b["latencies"].append(r["latency_ms"])
    return [{
        "start": b["start"].isoformat(), "answered": b["answered"], "not_answered": b["not_answered"],
        "error": b["error"], "p50_ms": percentile(b["latencies"], 0.5), "p95_ms": percentile(b["latencies"], 0.95),
    } for b in buckets]


def _cost(prompt_tokens: int, completion_tokens: int) -> float | None:
    s = get_settings()
    if not (s.llm_cost_per_million_input_tokens or s.llm_cost_per_million_output_tokens):
        return None
    return round(prompt_tokens / 1e6 * s.llm_cost_per_million_input_tokens
                 + completion_tokens / 1e6 * s.llm_cost_per_million_output_tokens, 4)


def recommendations(summary: dict, errors: list[dict], steps: list[dict], gaps: list[dict],
                    unverified: int, today: dict) -> list[dict]:
    """Plain-English suggestions, most urgent first. severity: critical | warning | info."""
    s = get_settings()
    recs = []
    by_type = {e["outcome"]: e["count"] for e in errors}
    if by_type.get("llm_auth_error"):
        recs.append({"severity": "critical", "title": "The Groq API key is being rejected",
                     "detail": ERROR_FIXES["llm_auth_error"]})
    if by_type.get("llm_rate_limited"):
        recs.append({"severity": "warning", "title": "Groq is rate-limiting the app",
                     "detail": ERROR_FIXES["llm_rate_limited"]})
    if summary["total"] and summary["errors"] / summary["total"] > 0.05:
        recs.append({"severity": "warning", "title": f"{summary['errors']} of {summary['total']} questions failed",
                     "detail": "See Errors below for the causes and fixes."})
    if summary["p95_ms"] and summary["p95_ms"] > s.slow_answer_ms and steps:
        top = steps[0]
        advice = {
            "web_search": "Web search is the slowest step. It runs only when the documents don't cover a "
                          "question: uploading documents for common questions avoids it.",
            "generate": "Writing the answer is the slowest step. A faster GROQ_MODEL or fewer chunks "
                        "(TOP_K) shortens it.",
            "grounding": "The fact-check is the slowest step. Check that CHECK_MODEL is a small, fast model.",
            "grade": "The relevance check is the slowest step. Check that CHECK_MODEL is a small, fast model.",
        }.get(top["step"], "Open a slow answer below to see its timings.")
        recs.append({"severity": "warning",
                     "title": f"The slowest 5% of answers take {summary['p95_ms'] / 1000:.1f} s",
                     "detail": f"{advice} ({top['label']} averages {top['avg_ms'] / 1000:.1f} s.)"})
    if gaps:
        top = ", ".join(f"“{g['question']}”" for g in gaps[:3])
        recs.append({"severity": "info", "title": f"{len(gaps)} questions the documents couldn't answer",
                     "detail": f"Most asked: {top}. Upload a document that covers them.", "action": "upload"})
    fb = summary["feedback"]
    if fb["up"] + fb["down"] >= 5 and fb["down"] / (fb["up"] + fb["down"]) >= 0.2:
        recs.append({"severity": "warning", "title": f"{fb['down']} answers marked 👎",
                     "detail": "Review them below; add real failures to the eval set with Export eval candidates."})
    if unverified:
        recs.append({"severity": "info", "title": f"{unverified} answers failed the fact-check",
                     "detail": "The documents may be unclear or contradict each other on these topics. "
                               "Open them below to see which sources were used."})
    for metric, value in today.items():
        if value["limit"] and value["used"] >= 0.8 * value["limit"]:
            recs.append({"severity": "warning", "title": f"Today's {metric.replace('_', ' ')}: "
                                                         f"{value['used']} of {value['limit']}",
                         "detail": "Daily usage is near its limit (docs/alerts.md > Usage limits)."})
    if s.storage_backend != "firestore":
        recs.append({"severity": "info", "title": "History is stored in a local file",
                     "detail": "On Cloud Run it resets when the app restarts. Run ops/storage/setup_storage.sh "
                               "to keep it permanently (docs/storage.md)."})
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(recs, key=lambda r: order[r["severity"]])


def monitoring(hours: int = 24) -> dict:
    s = get_settings()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    records = store.recent(limit=WINDOW_LIMIT, since=start)
    total = len(records)
    uncached = [r for r in records if not r.get("cached") and r.get("latency_ms") is not None]
    latencies = [r["latency_ms"] for r in uncached]
    groups = {"answered": 0, "not_answered": 0, "error": 0}
    for r in records:
        groups[group(r["outcome"])] += 1
    prompt = sum((r.get("tokens") or {}).get("prompt_tokens", 0) for r in records)
    completion = sum((r.get("tokens") or {}).get("completion_tokens", 0) for r in records)
    feedback = {"up": sum(r.get("rating") == "up" for r in records),
                "down": sum(r.get("rating") == "down" for r in records)}

    step_totals: dict[str, list[float]] = {}
    for r in uncached:
        for step, ms in (r.get("timings_ms") or {}).items():
            if step in STEP_LABELS and ms:
                step_totals.setdefault(step, []).append(ms)
    steps = sorted(({"step": k, "label": STEP_LABELS[k], "avg_ms": round(sum(v) / len(v), 1), "runs": len(v)}
                    for k, v in step_totals.items()), key=lambda x: -x["avg_ms"])

    error_groups: dict[str, dict] = {}
    for r in records:
        if is_error(r["outcome"]):
            g = error_groups.setdefault(r["outcome"], {"outcome": r["outcome"], "count": 0, "last_seen": r["created_at"],
                                                       "fix": ERROR_FIXES.get(r["outcome"], ERROR_FIXES["error"]),
                                                       "examples": []})
            g["count"] += 1
            g["last_seen"] = max(g["last_seen"], r["created_at"])
            if len(g["examples"]) < 5:
                g["examples"].append(_brief(r))
    errors = sorted(error_groups.values(), key=lambda g: -g["count"])

    gaps = store._unanswered(records)[:20]
    slow = sorted((r for r in uncached if r["latency_ms"] > s.slow_answer_ms), key=lambda r: -r["latency_ms"])
    summary = {
        "total": total,
        "answered": groups["answered"], "not_answered": groups["not_answered"], "errors": groups["error"],
        "answer_rate": round(groups["answered"] / total, 3) if total else None,
        "avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95),
        "max_ms": max(latencies) if latencies else None,
        "cache_hit_rate": round(sum(bool(r.get("cached")) for r in records) / total, 3) if total else None,
        "web_searches": sum(bool(r.get("web_searched")) for r in records),
        "feedback": feedback,
        "tokens": {"prompt": prompt, "completion": completion, "total": prompt + completion},
        "estimated_cost_usd": _cost(prompt, completion),
    }
    unverified = [r for r in records if r["outcome"] == "unverified"]
    today = usage.snapshot()
    return {
        "range_hours": hours, "from": start.isoformat(), "to": end.isoformat(),
        "truncated": total >= WINDOW_LIMIT, "storage": s.storage_backend,
        "slow_answer_ms": s.slow_answer_ms,
        "summary": summary,
        "timeline": _buckets(records, start, end, hours),
        "steps": steps,
        "today": today,
        "recommendations": recommendations(summary, errors, steps, gaps, len(unverified), today),
        "errors": errors,
        "slow": [_brief(r) for r in slow[:20]],
        "gaps": gaps,
        "downvoted": [_brief(r) for r in records if r.get("rating") == "down"][:20],
        "unverified": [_brief(r) for r in unverified][:20],
    }


CSV_FIELDS = ["created_at", "id", "session_id", "outcome", "latency_ms", "question", "answer", "sources",
              "decision", "grounded", "web_searched", "cached", "llm_model", "check_model", "prompt_version",
              "prompt_tokens", "completion_tokens", "rating", "comment", "pii_masked", "error_type"]


def audit_csv(hours: int) -> str:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for r in store.recent(limit=WINDOW_LIMIT * 5, since=since):
        tokens = r.get("tokens") or {}
        writer.writerow({
            **r,
            "sources": "; ".join(f"{s.get('filename')} p.{s.get('page')}" if not s.get("url") else s["url"]
                                 for s in r.get("sources") or []),
            "pii_masked": ",".join(r.get("pii_masked") or []),
            "prompt_tokens": tokens.get("prompt_tokens"), "completion_tokens": tokens.get("completion_tokens"),
        })
    return out.getvalue()
