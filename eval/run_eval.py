"""Evaluate retrieval (and optionally answers) on the Thistlewood Bank question set.

Builds a throwaway index from the sample PDFs, so your real backend/storage is never touched.

Usage (from the project root, with the backend virtualenv active):
    python eval/run_eval.py                       # retrieval + threshold sweep, no API key needed
    python eval/run_eval.py --llm                 # also run the full pipeline (needs GROQ_API_KEY)
    python eval/run_eval.py --docs sample-docs/thistlewood path/to/RBI-KYC-Master-Direction.pdf
    python eval/run_eval.py --compare             # before/after table for the retrieval variants
    SELF_CORRECTION=false python eval/run_eval.py --llm   # answers without the agent checks

Retrieval uses the app's own search (backend/app/rag/retrieval.py), so the results reflect
CHUNK_SIZE, STRIP_REPEATED_LINES and RETRIEVAL_MODE from backend/.env unless --compare is used.

Metrics:
    hit@k         in-scope questions where a chunk from the expected file/page passes the threshold
    top-1 hit     ...where the highest-scoring chunk is from the expected file/page
    refusal       out-of-scope questions where no chunk passes the threshold (bot says "not found")
    balanced      average of hit@k and refusal; used to suggest a threshold
With --llm:
    answered      in-scope questions that got an answer rather than "not found"
    citations ok  answers whose [n] citations all point at a returned source
    cited right   answers citing at least one expected file/page
    keywords      answers containing at least one expected keyword (rough correctness check)
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [round(0.05 * i, 2) for i in range(0, 15)]  # 0.00 .. 0.70

# Each step adds one improvement, so the table shows what each change contributes.
COMPARE_CONFIGS = [
    ("baseline (1000 chars, vector)", dict(chunk_size=1000, chunk_overlap=200,
                                           strip_repeated_lines=False, retrieval_mode="vector")),
    ("+ strip headers/footers", dict(chunk_size=1000, chunk_overlap=200,
                                     strip_repeated_lines=True, retrieval_mode="vector")),
    ("+ 500-char chunks", dict(chunk_size=500, chunk_overlap=100,
                               strip_repeated_lines=True, retrieval_mode="vector")),
    ("+ hybrid BM25 (500 chars)", dict(chunk_size=500, chunk_overlap=100,
                                       strip_repeated_lines=True, retrieval_mode="hybrid")),
    ("hybrid BM25 (1000 chars)", dict(chunk_size=1000, chunk_overlap=200,
                                      strip_repeated_lines=True, retrieval_mode="hybrid")),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", nargs="+", default=[str(ROOT / "sample-docs" / "thistlewood")],
                   help="PDF files or folders to index (name the RBI file with 'rbi' in it)")
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    p.add_argument("--threshold", type=float, help="threshold for --llm and the detail report "
                   "(default: RELEVANCE_THRESHOLD from backend/.env)")
    p.add_argument("--llm", action="store_true", help="also generate answers with the full LangGraph pipeline")
    p.add_argument("--compare", action="store_true",
                   help="evaluate each retrieval variant in COMPARE_CONFIGS and print a comparison table")
    p.add_argument("--out", default=str(ROOT / "eval" / "results" / "latest.json"))
    return p.parse_args()


def setup_app(tmp: Path):
    # Point the app at a temporary index before its settings are first loaded.
    os.environ["CHROMA_DIR"] = str(tmp / "chroma")
    os.environ["UPLOAD_DIR"] = str(tmp / "uploads")
    os.environ["REGISTRY_PATH"] = str(tmp / "documents.json")
    sys.path.insert(0, str(ROOT / "backend"))
    from app.config import get_settings  # still reads backend/.env for model, top_k, keys
    get_settings.cache_clear()
    return get_settings()


def collect_pdfs(paths):
    pdfs = []
    for p in map(Path, paths):
        pdfs += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    return pdfs


def matches(meta_file: str, meta_page: int, text: str, q: dict) -> bool:
    for file_key, page in q.get("sources", []):
        if file_key.lower() in meta_file.lower() and (page is None or page == meta_page):
            kws = q.get("chunk_keywords")
            if not kws or any(k.lower() in text.lower() for k in kws):
                return True
    return False


def kept_chunks(row, t):
    """Chunks the app would use at threshold t (same rule as app.rag.retrieval.search)."""
    if row["best"] < t:
        return []
    if row["mode"] == "hybrid":
        return row["chunks"]
    return [c for c in row["chunks"] if c["score"] is not None and c["score"] >= t]


def retrieval_metrics(rows, t):
    ins = [r for r in rows if r["type"] == "in_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    hit = top1 = 0
    for r in ins:
        kept = kept_chunks(r, t)
        hit += any(c["match"] for c in kept)
        top1 += bool(kept) and kept[0]["match"]
    refused = sum(not kept_chunks(r, t) for r in oos)
    m = {
        "threshold": t,
        "hit_at_k": hit / len(ins) if ins else 0.0,
        "top1_hit": top1 / len(ins) if ins else 0.0,
        "refusal": refused / len(oos) if oos else 0.0,
    }
    m["balanced"] = (m["hit_at_k"] + m["refusal"]) / 2
    return m


def best_of(sweep):
    return max(sweep, key=lambda m: (round(m["balanced"], 4), m["hit_at_k"]))


def pct(x):
    return f"{x * 100:5.1f}%"


def build_index(settings, pdfs, tmp: Path, overrides: dict | None = None, quiet=False) -> int:
    """(Re)build a throwaway index with the given setting overrides; returns the chunk count."""
    from app.rag.ingestion import ingest_pdf
    from app.rag.retrieval import invalidate_keyword_index
    from app.rag.vectorstore import get_vectorstore

    for key, value in (overrides or {}).items():
        setattr(settings, key, value)
    run = tmp / uuid.uuid4().hex[:8]
    settings.chroma_dir, settings.upload_dir = run / "chroma", run / "uploads"
    settings.registry_path = run / "documents.json"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    get_vectorstore.cache_clear()  # new Chroma folder; the embedding model stays loaded
    invalidate_keyword_index()

    total = 0
    for pdf in pdfs:
        doc_id = uuid.uuid4().hex
        dest = settings.upload_dir / f"{doc_id}.pdf"
        shutil.copy(pdf, dest)
        rec = ingest_pdf(dest, doc_id, pdf.name)
        total += rec.chunks
        if not quiet:
            print(f"  {pdf.name}: {rec.pages} pages, {rec.chunks} chunks")
    return total


def collect_rows(settings, questions):
    """Search every question with the threshold disabled, so any threshold can be scored later."""
    from app.rag.retrieval import search

    saved, settings.relevance_threshold = settings.relevance_threshold, 0.0
    try:
        rows = []
        for q in questions:
            hits, best = search(q["question"])
            chunks = [
                {"file": h.doc.metadata.get("filename"), "page": h.doc.metadata.get("page"),
                 "score": None if h.vector_score is None else round(h.vector_score, 4),
                 "match": matches(h.doc.metadata.get("filename", ""), h.doc.metadata.get("page"),
                                  h.doc.page_content, q)}
                for h in hits
            ]
            rows.append({**q, "mode": settings.retrieval_mode, "best": round(best, 4), "chunks": chunks})
        return rows
    finally:
        settings.relevance_threshold = saved


def load_questions(path, pdfs):
    indexed = " ".join(p.name.lower() for p in pdfs)
    questions = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    skipped = [q["id"] for q in questions if q.get("requires") and q["requires"] not in indexed]
    return [q for q in questions if q["id"] not in skipped], skipped


def run_compare(args, settings, pdfs, tmp):
    questions, skipped = load_questions(args.questions, pdfs)
    print(f"Comparing {len(COMPARE_CONFIGS)} retrieval variants on {len(questions)} questions "
          f"(top_k={settings.top_k}, embeddings={settings.embedding_model})\n")
    results = []
    for name, overrides in COMPARE_CONFIGS:
        chunks = build_index(settings, pdfs, tmp, overrides, quiet=True)
        rows = collect_rows(settings, questions)
        sweep = [retrieval_metrics(rows, t) for t in THRESHOLDS]
        results.append({"name": name, "settings": overrides, "chunks": chunks,
                        "no_threshold": sweep[0], "best": best_of(sweep), "sweep": sweep})
        print(f"  done: {name}")

    print("\n| Variant | Chunks | Hit@k (no threshold) | Top-1 (no threshold) | Best threshold "
          "| Hit@k | Top-1 | Refusal | Balanced |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        n, b = r["no_threshold"], r["best"]
        print(f"| {r['name']} | {r['chunks']} | {pct(n['hit_at_k'])} | {pct(n['top1_hit'])} | "
              f"{b['threshold']:.2f} | {pct(b['hit_at_k'])} | {pct(b['top1_hit'])} | {pct(b['refusal'])} | "
              f"{pct(b['balanced'])} |")
    print("\nHit@k: right page among the chunks sent to the LLM. Top-1: right page ranked first."
          "\nRefusal: off-topic questions answered with 'not found'. Balanced: average of Hit@k and Refusal.")
    return {"mode": "compare", "top_k": settings.top_k, "model": settings.embedding_model,
            "skipped": skipped, "variants": results}


def main():
    args = parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    try:
        settings = setup_app(tmp)
        pdfs = collect_pdfs(args.docs)
        if not pdfs:
            sys.exit("No PDFs found to index.")

        if args.compare:
            report = run_compare(args, settings, pdfs, tmp)
        else:
            report = run_single(args, settings, pdfs, tmp)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nDetails written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_single(args, settings, pdfs, tmp):
    print(f"Indexing {len(pdfs)} PDF(s) with {settings.embedding_model} "
          f"(chunk_size={settings.chunk_size}, strip_repeated_lines={settings.strip_repeated_lines}, "
          f"retrieval_mode={settings.retrieval_mode}) ...")
    build_index(settings, pdfs, tmp)
    questions, skipped = load_questions(args.questions, pdfs)
    rows = collect_rows(settings, questions)

    sweep = [retrieval_metrics(rows, t) for t in THRESHOLDS]
    best = best_of(sweep)
    current = args.threshold if args.threshold is not None else settings.relevance_threshold

    n_in = sum(r["type"] == "in_scope" for r in rows)
    print(f"\nQuestions: {n_in} in-scope, {len(rows) - n_in} out-of-scope, top_k={settings.top_k}"
          + (f", skipped {len(skipped)} needing missing docs: {', '.join(skipped)}" if skipped else ""))
    print("\nThreshold sweep (retrieval only)")
    print("threshold | hit@k  | top-1  | refusal | balanced")
    for m in sweep:
        mark = "  <- best" if m is best else ("  <- current" if abs(m["threshold"] - current) < 1e-9 else "")
        print(f"  {m['threshold']:.2f}    | {pct(m['hit_at_k'])} | {pct(m['top1_hit'])} | {pct(m['refusal'])}  | "
              f"{pct(m['balanced'])}{mark}")
    print(f"\nSuggested RELEVANCE_THRESHOLD={best['threshold']:.2f} (current {current:.2f})")

    cur = retrieval_metrics(rows, current)
    misses = [r for r in rows if r["type"] == "in_scope" and not any(c["match"] for c in kept_chunks(r, current))]
    leaks = [r for r in rows if r["type"] == "out_of_scope" and kept_chunks(r, current)]
    if misses or leaks:
        print(f"\nFailures at threshold {current:.2f}:")
        for r in misses:
            top = r["chunks"][0] if r["chunks"] else None
            got = f"{top['file']} p.{top['page']}" if top else "nothing"
            print(f"  MISS    {r['id']}: {r['question']}  -> top: {got} (best score {r['best']:.2f})")
        for r in leaks:
            print(f"  ANSWERS {r['id']}: {r['question']}  -> best score {r['best']:.2f}")

    report = {"model": settings.embedding_model, "top_k": settings.top_k, "threshold": current,
              "chunk_size": settings.chunk_size, "strip_repeated_lines": settings.strip_repeated_lines,
              "retrieval_mode": settings.retrieval_mode, "retrieval": cur, "sweep": sweep,
              "suggested_threshold": best["threshold"], "skipped": skipped, "questions": rows}
    if args.llm:
        report["answers"] = run_llm(rows, current, settings)
    return report


def run_llm(rows, threshold, settings):
    from app.rag.graph import get_graph

    settings.relevance_threshold = threshold
    graph = get_graph()
    results, latencies = [], []
    print(f"\nRunning full pipeline with {settings.groq_model} ...")
    for r in rows:
        start = time.perf_counter()
        state = graph.invoke({"question": r["question"], "history": []})
        latencies.append(time.perf_counter() - start)
        answer, sources = state["answer"], state.get("sources") or []
        outcome = state.get("outcome") or ("answered" if sources else "not_found")
        refused = outcome not in ("answered", "answered_web") or "couldn't find" in answer.lower()
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        res = {"id": r["id"], "type": r["type"], "answer": answer, "refused": refused, "outcome": outcome,
               "sources": [(s["filename"], s["page"]) for s in sources]}
        if r["type"] == "in_scope":
            res["citations_ok"] = bool(cited) and all(1 <= n <= len(sources) for n in cited)
            res["cited_right"] = any(
                1 <= n <= len(sources) and matches(sources[n - 1]["filename"], sources[n - 1]["page"],
                                                   " ".join(sources[n - 1]["chunks"]), r)
                for n in cited)
            kws = r.get("answer_keywords") or []
            res["keywords"] = not kws or any(k.lower() in answer.lower() for k in kws)
        results.append(res)
        time.sleep(0.5)  # stay under Groq free-tier rate limits

    ins = [x for x in results if x["type"] == "in_scope"]
    oos = [x for x in results if x["type"] == "out_of_scope"]
    rate = lambda xs, key: sum(bool(x.get(key)) for x in xs) / len(xs) if xs else 0.0
    summary = {
        "answered": sum(not x["refused"] for x in ins) / len(ins) if ins else 0.0,
        "citations_ok": rate(ins, "citations_ok"),
        "cited_right": rate(ins, "cited_right"),
        "keywords": rate(ins, "keywords"),
        "refused_out_of_scope": rate(oos, "refused"),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
    }
    outcomes = {}
    for x in results:
        outcomes[x["outcome"]] = outcomes.get(x["outcome"], 0) + 1
    print(f"Answer metrics (self_correction={settings.self_correction}, "
          f"grounding_check={settings.grounding_check})")
    for k, v in summary.items():
        print(f"  {k:22s} {v if k == 'avg_latency_s' else pct(v)}")
    print("  outcomes               " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    for x in results:
        if (x["type"] == "in_scope" and not (x["citations_ok"] and x["keywords"])) or \
           (x["type"] == "out_of_scope" and not x["refused"]):
            print(f"  CHECK {x['id']}: {x['answer'][:150]!r}")
    return {"summary": summary, "outcomes": outcomes, "results": results}


if __name__ == "__main__":
    main()
