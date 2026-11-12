"""Evaluate retrieval (and optionally answers) on the Thistlewood Bank question set.

Builds a throwaway index from the sample PDFs, so your real backend/storage is never touched.

Usage (from the project root, with the backend virtualenv active):
    python eval/run_eval.py                       # retrieval + threshold sweep, no API key needed
    python eval/run_eval.py --llm                 # also run the full pipeline (needs GROQ_API_KEY)
    python eval/run_eval.py --docs sample-docs/thistlewood path/to/RBI-KYC-Master-Direction.pdf

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", nargs="+", default=[str(ROOT / "sample-docs" / "thistlewood")],
                   help="PDF files or folders to index (name the RBI file with 'rbi' in it)")
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    p.add_argument("--threshold", type=float, help="threshold for --llm and the detail report "
                   "(default: RELEVANCE_THRESHOLD from backend/.env)")
    p.add_argument("--llm", action="store_true", help="also generate answers with the full LangGraph pipeline")
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


def retrieval_metrics(rows, t):
    ins = [r for r in rows if r["type"] == "in_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    hit = sum(any(c["match"] and c["score"] >= t for c in r["chunks"]) for r in ins)
    top1 = sum(bool(r["chunks"]) and r["chunks"][0]["score"] >= t and r["chunks"][0]["match"] for r in ins)
    refused = sum(all(c["score"] < t for c in r["chunks"]) for r in oos)
    m = {
        "threshold": t,
        "hit_at_k": hit / len(ins) if ins else 0.0,
        "top1_hit": top1 / len(ins) if ins else 0.0,
        "refusal": refused / len(oos) if oos else 0.0,
    }
    m["balanced"] = (m["hit_at_k"] + m["refusal"]) / 2
    return m


def pct(x):
    return f"{x * 100:5.1f}%"


def main():
    args = parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    try:
        settings = setup_app(tmp)
        from app.rag.ingestion import ingest_pdf
        from app.rag.vectorstore import get_vectorstore

        pdfs = collect_pdfs(args.docs)
        if not pdfs:
            sys.exit("No PDFs found to index.")
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        print(f"Indexing {len(pdfs)} PDF(s) with {settings.embedding_model} ...")
        for pdf in pdfs:
            doc_id = uuid.uuid4().hex
            dest = settings.upload_dir / f"{doc_id}.pdf"
            shutil.copy(pdf, dest)
            rec = ingest_pdf(dest, doc_id, pdf.name)
            print(f"  {pdf.name}: {rec.pages} pages, {rec.chunks} chunks")
        indexed = " ".join(p.name.lower() for p in pdfs)

        questions = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
        skipped = [q["id"] for q in questions if q.get("requires") and q["requires"] not in indexed]
        questions = [q for q in questions if q["id"] not in skipped]

        store = get_vectorstore()
        rows = []
        for q in questions:
            results = store.similarity_search_with_relevance_scores(q["question"], k=settings.top_k)
            chunks = [
                {"file": d.metadata.get("filename"), "page": d.metadata.get("page"), "score": round(s, 4),
                 "match": matches(d.metadata.get("filename", ""), d.metadata.get("page"), d.page_content, q)}
                for d, s in results
            ]
            rows.append({**q, "chunks": chunks})

        sweep = [retrieval_metrics(rows, t) for t in THRESHOLDS]
        best = max(sweep, key=lambda m: (round(m["balanced"], 4), m["hit_at_k"]))
        current = args.threshold if args.threshold is not None else settings.relevance_threshold

        n_in = sum(r["type"] == "in_scope" for r in rows)
        n_oos = len(rows) - n_in
        print(f"\nQuestions: {n_in} in-scope, {n_oos} out-of-scope, top_k={settings.top_k}"
              + (f", skipped {len(skipped)} needing missing docs: {', '.join(skipped)}" if skipped else ""))
        print("\nThreshold sweep (retrieval only)")
        print("threshold | hit@k  | top-1  | refusal | balanced")
        for m in sweep:
            mark = "  <- best" if m is best else ("  <- current" if abs(m["threshold"] - current) < 1e-9 else "")
            print(f"  {m['threshold']:.2f}    | {pct(m['hit_at_k'])} | {pct(m['top1_hit'])} | {pct(m['refusal'])}  | "
                  f"{pct(m['balanced'])}{mark}")
        print(f"\nSuggested RELEVANCE_THRESHOLD={best['threshold']:.2f} (current {current:.2f})")

        cur = retrieval_metrics(rows, current)
        misses = [r for r in rows if r["type"] == "in_scope"
                  and not any(c["match"] and c["score"] >= current for c in r["chunks"])]
        leaks = [r for r in rows if r["type"] == "out_of_scope" and any(c["score"] >= current for c in r["chunks"])]
        if misses or leaks:
            print(f"\nFailures at threshold {current:.2f}:")
            for r in misses:
                top = r["chunks"][0] if r["chunks"] else None
                got = f"{top['file']} p.{top['page']} ({top['score']:.2f})" if top else "nothing"
                print(f"  MISS    {r['id']}: {r['question']}  -> top: {got}")
            for r in leaks:
                print(f"  ANSWERS {r['id']}: {r['question']}  -> top score {r['chunks'][0]['score']:.2f}")

        report = {"model": settings.embedding_model, "top_k": settings.top_k, "threshold": current,
                  "retrieval": cur, "sweep": sweep, "suggested_threshold": best["threshold"],
                  "skipped": skipped, "questions": rows}

        if args.llm:
            report["answers"] = run_llm(rows, current, settings)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nDetails written to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_llm(rows, threshold, settings):
    from app.rag.graph import NO_CONTEXT_ANSWER, get_graph

    settings.relevance_threshold = threshold
    graph = get_graph()
    results, latencies = [], []
    print(f"\nRunning full pipeline with {settings.groq_model} ...")
    for r in rows:
        start = time.perf_counter()
        state = graph.invoke({"question": r["question"], "history": []})
        latencies.append(time.perf_counter() - start)
        answer, sources = state["answer"], state.get("sources") or []
        refused = answer == NO_CONTEXT_ANSWER or "couldn't find" in answer.lower()
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
        res = {"id": r["id"], "type": r["type"], "answer": answer, "refused": refused,
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
    print("Answer metrics")
    for k, v in summary.items():
        print(f"  {k:22s} {v if k == 'avg_latency_s' else pct(v)}")
    for x in results:
        if (x["type"] == "in_scope" and not (x["citations_ok"] and x["keywords"])) or \
           (x["type"] == "out_of_scope" and not x["refused"]):
            print(f"  CHECK {x['id']}: {x['answer'][:150]!r}")
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    main()
