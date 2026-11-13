# Thistlewood Bank Assistant: a self-correcting RAG agent

A banking help assistant that answers customer questions (KYC, Video KYC, fees, account rules)
from the bank's own documents, with citations. It is built as a **LangGraph agent** that grades
what it retrieved, retries the search, checks its answer for unsupported claims, and refuses
out-of-scope requests, with PII masking, observability and an evaluation harness around it.

**Live demo:** https://thistlewood-assistant-399551187226.us-central1.run.app
(first load after a quiet period takes 20–40 seconds while the container starts).

> Thistlewood Bank is fictional. Portfolio demo, not affiliated with any bank or the RBI,
> and not financial advice.

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph (conditional edges, bounded retry loops) |
| LLM | Groq (`openai/gpt-oss-120b` by default, set with `GROQ_MODEL`) |
| Retrieval | Chroma vector search + optional BM25 keyword search, fused with reciprocal rank fusion |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (local, open source) |
| PDF loading / chunking | LangChain `PyPDFLoader`, `RecursiveCharacterTextSplitter`, header/footer stripping |
| Backend | FastAPI (`backend/`), SQLite for interactions and feedback |
| Frontend | React + Vite (`frontend/`) |
| Deployment | Docker, Google Cloud Run, Cloud Build, Artifact Registry, Secret Manager |
| Observability | Structured JSON logs, Cloud Logging log-based metrics, Cloud Monitoring dashboard |

## How it works

```
Admin upload ─► PyPDFLoader ─► strip headers/footers ─► text splitter ─► embeddings ─► Chroma (+ BM25)

Question ─► PII masking ─► [contextualize] ─► [retrieve] ─┬─ nothing relevant ──────────────────► [refuse]
            (Aadhaar, PAN,   follow-ups use                └─ chunks ─► [grade] (LLM)
             cards, OTPs)    server-side history                          ├─ answerable ─► [generate] ─► [check_grounding] (LLM)
                                                                          │                                  ├─ grounded ─► answer + sources
                                                                          │                                  └─ unsupported ─► regenerate once ─► or [refuse: unverified]
                                                                          ├─ not in documents ─► [rewrite_query] ─► [retrieve] (once)
                                                                          └─ account data / advice / off-topic / unsafe ─► [refuse] (specific message)
```

- **small_talk:** greetings, thanks, goodbyes and "what can you do?" get a short friendly reply
  with example questions, without a search or an LLM call (`backend/app/rag/smalltalk.py`).
- **retrieve:** vector search (or hybrid vector + BM25). Chunks below `RELEVANCE_THRESHOLD`
  are dropped, so clearly off-topic questions never reach the LLM. Chunks are grouped by
  (document, page) and numbered once, so citation `[n]` always matches source `n` in the UI.
- **grade:** the LLM decides whether the excerpts answer the question, or whether it needs
  account access, asks for financial advice, is off-topic, or tries to override instructions.
  This catches questions that look similar to the documents but aren't covered by them (for
  example "home loan rate" next to a savings-rate table), which a similarity threshold cannot.
- **rewrite_query:** if the documents seem not to cover it, retry once with wording closer to
  how the documents phrase things.
- **check_grounding:** the LLM verifies every factual claim (fees, limits, timelines, steps)
  against the excerpts; unsupported answers are regenerated once, then refused as "unverified".

The agent is in `backend/app/rag/graph.py`, retrieval in `backend/app/rag/retrieval.py`, and
ingestion in `backend/app/rag/ingestion.py`. Set `SELF_CORRECTION=false` for the plain
retrieve-then-generate pipeline (fewer LLM calls), or `GROUNDING_CHECK=false` to skip only the
grounding check.

## Setup

### Backend

```powershell
# from the project root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt

copy backend\.env.example backend\.env   # then set GROQ_API_KEY and ADMIN_API_KEY

cd backend
python run.py
```

The first start downloads the embedding model (about 90 MB). API docs are at http://localhost:8731/docs.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (http://localhost:5175 by default). The **Admin** tab asks for
`ADMIN_API_KEY`. Use it to upload and delete PDFs. The **Chat** tab is the user-facing chatbot.

### Start both with one command (Windows)

After the one-time setup above, start everything with:

```powershell
.\start-dev.ps1
```

or double-click `start-dev.cmd`. It opens two windows, "ragbased backend" and
"ragbased frontend", waits for the backend to be ready, then starts the frontend (running
`npm install` on the first run). Press Ctrl+C or close a window to stop that server. If
PowerShell refuses to run scripts, use `start-dev.cmd`, or run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Ports

No ports are hard-coded; they are settings with defaults:

| Setting | File | Default | Notes |
|---|---|---|---|
| `PORT` | `backend/.env` | `8731` | Backend port used by `python run.py` |
| `FRONTEND_PORT` | `frontend/.env` | `5175` | If busy, Vite uses the next free port and prints it |
| `BACKEND_URL` | `frontend/.env` | `http://localhost:8731` | Must match the backend `PORT` |

To change them, copy `frontend/.env.example` to `frontend/.env` and edit, and set `PORT` in
`backend/.env`. CORS allows any `localhost` port, so it needs no change.

## Guardrails

| Guardrail | Where |
|---|---|
| **PII masking**: Aadhaar, PAN, card numbers (Luhn-checked), account and phone numbers, emails, UPI IDs and OTP/PIN/CVV values are replaced with placeholders such as `[AADHAAR]` before the question or history reaches the LLM, the vector search or the logs; the user sees a privacy notice | `backend/app/guardrails/pii.py`, tests in `backend/tests/` |
| **Scope grader**: refuses account-data requests, financial advice, off-topic and instruction-override attempts with a specific message | `grade` node, `backend/app/rag/graph.py` |
| **Grounding check**: every factual claim verified against the sources; regenerate once, otherwise refuse | `check_grounding` node |
| **Prompt-injection resistance**: chat history kept on the server (clients only send a session id, so they can't forge assistant turns); documents wrapped in a `<documents>` block and treated as data, never instructions | `backend/app/api/sessions.py`, prompts in `graph.py` |
| **Grounded answers**: answer only from retrieved context, cite sources, say "not found" otherwise; never ask for personal details | `backend/app/rag/graph.py` (prompt) |
| **Relevance threshold**: off-topic questions get "not found" without calling the LLM | `RELEVANCE_THRESHOLD` |
| **Input limits**: question up to 2,000 characters, last 10 history messages | `backend/app/api/chat.py` |
| **Rate limits**: per visitor per minute and a daily cap | `CHAT_REQUESTS_PER_MINUTE_PER_IP`, `CHAT_REQUESTS_PER_DAY` |
| **Admin protection**: admin key (constant-time compare), PDF only, size limit | `backend/app/api/` |
| **Safe rendering**: raw HTML in answers is never rendered | `frontend/src/components/Chat.jsx` |

Run the tests from `backend/` with `pip install pytest` and `python -m pytest`.

## Feedback and Insights

Every answer has 👍/👎 buttons (👎 asks for an optional comment, which is PII-masked). Each
question (already PII-masked), answer, outcome and score is stored in SQLite
(`STORE_INTERACTIONS`, `backend/storage/interactions.db`).

**Admin → Insights** shows total questions, answer rate, 👍/👎 counts, average response time,
outcomes, **knowledge gaps** (questions the documents couldn't answer, most frequent first) and
downvoted answers. **Export eval candidates** downloads those failures as JSON lines in the
eval format, to label and add to `eval/questions.jsonl`, so every real failure becomes a test.

On Cloud Run the disk is wiped on restart, so this history is temporary there.

## Deployment

The app ships as one container: FastAPI serves the API and the built React app, and indexes
the Thistlewood sample PDFs at startup. See [docs/deploy-cloud-run.md](docs/deploy-cloud-run.md)
for a step-by-step Google Cloud Run guide.

**Observability:** every chat request writes one structured JSON log line (outcome, latency per
pipeline step, retrieval scores, Groq token usage, PII-masking flags; never question text).
[docs/observability.md](docs/observability.md) shows how to query the logs and create
log-based metrics, a Cloud Monitoring dashboard and an error alert from `ops/monitoring/`.

## Sample documents (Thistlewood Bank)

`sample-docs/thistlewood/` contains four short PDFs for a **fictional** bank, Thistlewood Bank:
account opening and KYC, Video KYC, re-KYC and updating details, and fees and support. The
Markdown files next to them are the sources; rebuild the PDFs with
`pip install reportlab` and `python scripts/build_sample_pdfs.py`.

Upload them from the Admin tab to try the chatbot. You can also add the RBI Master Direction
on KYC (download it from rbi.org.in, and keep `rbi` in the filename so the eval finds it).

> Thistlewood Bank is fictional. This is a portfolio demo, not affiliated with any bank or
> the Reserve Bank of India, and not financial advice.

## Results

Baseline, measured with `python eval/run_eval.py` on the 46-question set (MiniLM embeddings,
top 4 chunks, 1000-character chunks, vector search):

| Metric | Baseline (threshold 0.30) |
|---|---|
| Right page among the 4 chunks sent to the LLM (hit@4) | 89.2% |
| Right page ranked first (top-1) | 56.8% |
| Right page retrieved at all, no threshold | 97.3% |
| Off-topic questions refused by the threshold alone | 55.6% |

What the baseline showed: retrieval almost always finds the answer, but ranks it first only
about 60% of the time, and **no similarity threshold can separate hard negatives** ("home loan
rate at Thistlewood", 0.57) from real questions (many score around 0.3). That motivated the
LLM grader and the retrieval improvements below.

### Retrieval improvements

Measured with `python eval/run_eval.py --compare` (37 answerable + 9 out-of-scope questions,
top 4 chunks). "No threshold" columns measure ranking quality alone.

| Variant | Chunks | Hit@4 (no threshold) | Top-1 (no threshold) | Best threshold | Hit@4 | Top-1 | Refusal |
|---|---|---|---|---|---|---|---|
| Baseline (1000 chars, vector) | 23 | 97.3% | 62.2% | 0.30 | 89.2% | 56.8% | 55.6% |
| + strip headers/footers | 20 | **100.0%** | 75.7% | 0.15 | 100.0% | 75.7% | 44.4% |
| + 500-character chunks | 37 | 94.6% | 75.7% | 0.60 | 59.5% | 54.1% | 100.0% |
| + hybrid BM25 (500 chars) | 37 | 94.6% | 75.7% | 0.60 | 59.5% | 51.4% | 100.0% |
| **Hybrid BM25 (1000 chars), chosen** | 20 | **100.0%** | **78.4%** | 0.35 | 89.2% | 67.6% | 55.6% |

What we learned:

- **Stripping the repeated footer** was the biggest single win: top-1 from 62.2% to 75.7% and
  every answer page retrieved. The footer ("Thistlewood Bank is fictional…") was in every chunk,
  so it diluted the real content and boosted any question that mentioned the bank.
- **Smaller chunks hurt**: with 500-character chunks some answers were split away from their
  context (hit@4 fell to 94.6%), so the default stays at 1000 characters.
- **Hybrid BM25 + vector** ranked the right page first most often (**78.4%**, +16 points over the
  baseline), helping exact-term questions such as "charge for closing my account".
- **The threshold alone can't refuse well**: the only setting that refused every off-topic
  question (0.60) also dropped 40% of real questions. So refusal moved to the **LLM grader**, and
  the threshold was retuned for recall: **0.15 keeps 100% of real questions** and only filters
  obvious noise.

**Defaults now:** `STRIP_REPEATED_LINES=true`, `RETRIEVAL_MODE=hybrid`, `CHUNK_SIZE=1000`,
`RELEVANCE_THRESHOLD=0.15`, with the LLM grader and grounding check on.

### Answer quality with the agent

Run `python eval\run_eval.py --llm` (agent on), and with `SELF_CORRECTION=false` for the plain
pipeline, to compare answered rate, citation accuracy and out-of-scope refusal:

```powershell
python eval\run_eval.py --llm
$env:SELF_CORRECTION="false"; python eval\run_eval.py --llm; Remove-Item Env:SELF_CORRECTION
```

| Pipeline | Answered (in-scope) | Cited the right page | Refused out-of-scope | Avg latency |
|---|---|---|---|---|
| Plain retrieve → generate | _run --llm_ | | | |
| Agent (grader + grounding check) | | | | |

## Evaluation

`eval/questions.jsonl` has 37 questions answered by the sample documents (each tagged with the
file and page that holds the answer), 9 questions the bot should refuse (including tricky ones
such as "What is the home loan interest rate at Thistlewood Bank?"), and 4 optional questions
that need the RBI document.

```powershell
# from the project root, with the backend virtualenv active
python eval\run_eval.py          # retrieval metrics + threshold sweep, no API key needed
python eval\run_eval.py --llm    # also generate answers with Groq and check citations
python eval\run_eval.py --docs sample-docs\thistlewood path\to\rbi-kyc-master-direction.pdf
python eval\run_eval.py --compare  # before/after table for the retrieval improvements
```

`--compare` rebuilds the index for each retrieval variant (baseline, stripped headers/footers,
500-character chunks, hybrid BM25 + vector search) and prints one comparison table. The
variants are settings, so the winner can be switched on in `backend/.env` with
`STRIP_REPEATED_LINES`, `CHUNK_SIZE` / `CHUNK_OVERLAP` and `RETRIEVAL_MODE=hybrid`.

The script builds a temporary index, so it never touches `backend/storage/`. It reports
retrieval hit rate, top-1 hit rate and out-of-scope refusal rate for thresholds 0.00 to 0.70,
suggests a `RELEVANCE_THRESHOLD`, and lists every miss. With `--llm` it also checks that
citations point at returned sources and at the expected page. Full results are written to
`eval/results/latest.json`.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/chat` | none | `{question, session_id?}` → `{answer, sources, outcome, session_id, message_id, notice}` |
| POST | `/api/feedback` | none | `{message_id, rating: "up" \| "down", comment?}` |
| GET | `/api/admin/documents` | `X-Admin-Key` | List indexed documents |
| POST | `/api/admin/documents` | `X-Admin-Key` | Upload one or more PDFs (multipart `files`) |
| DELETE | `/api/admin/documents/{id}` | `X-Admin-Key` | Remove a document and its chunks |
| GET | `/api/admin/insights` | `X-Admin-Key` | Usage, answer rate, feedback, knowledge gaps, downvoted answers |
| GET | `/api/admin/insights/eval-candidates` | `X-Admin-Key` | Unanswered and downvoted questions as eval-format JSON lines |
| GET | `/api/health` | none | Health check |

Uploaded files, the Chroma index, the document registry and the interaction database are
stored in `backend/storage/`. `outcome` is one of `answered`, `small_talk`, `not_found`, `unverified`,
`needs_account_access`, `financial_advice`, `off_topic` or `unsafe`.
