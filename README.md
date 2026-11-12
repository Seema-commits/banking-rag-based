# RAG Chatbot

Admins upload PDFs, which are chunked, embedded and stored in a vector database. Users ask
questions, and the chatbot answers from the retrieved chunks and cites its sources.

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| PDF loading / chunking | LangChain `PyPDFLoader`, `RecursiveCharacterTextSplitter` |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (local, open source) |
| Vector DB | Chroma (`langchain-chroma`, persisted to disk) |
| LLM | Groq (`openai/gpt-oss-120b` by default, set with `GROQ_MODEL`) |
| Backend | FastAPI (`backend/`) |
| Frontend | React + Vite (`frontend/`) |

## Pipeline

```
Admin upload ─► PyPDFLoader ─► text splitter ─► HF embeddings ─► Chroma

User question ─► [contextualize] ─► [retrieve] ─┬─► [generate]   ─► answer + sources
                  rewrite follow-ups            └─► [no_context] (nothing relevant found)
                  using chat history
```

`retrieve` drops chunks whose relevance score (0–1) is below `RELEVANCE_THRESHOLD`, so
off-topic questions reach `no_context` instead of being answered from unrelated text.
Retrieved chunks are grouped by (document, page) and numbered once; the same numbers are
used for the LLM's inline citations and the source list returned to the UI.

The LangGraph graph is in `backend/app/rag/graph.py`, and ingestion is in `backend/app/rag/ingestion.py`.

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

### Ports

No ports are hard-coded; they are settings with defaults:

| Setting | File | Default | Notes |
|---|---|---|---|
| `PORT` | `backend/.env` | `8731` | Backend port used by `python run.py` |
| `FRONTEND_PORT` | `frontend/.env` | `5175` | If busy, Vite uses the next free port and prints it |
| `BACKEND_URL` | `frontend/.env` | `http://localhost:8731` | Must match the backend `PORT` |

To change them, copy `frontend/.env.example` to `frontend/.env` and edit, and set `PORT` in
`backend/.env`. CORS allows any `localhost` port, so it needs no change.

## Sample documents (Thistlewood Bank)

`sample-docs/thistlewood/` contains four short PDFs for a **fictional** bank, Thistlewood Bank:
account opening and KYC, Video KYC, re-KYC and updating details, and fees and support. The
Markdown files next to them are the sources; rebuild the PDFs with
`pip install reportlab` and `python scripts/build_sample_pdfs.py`.

Upload them from the Admin tab to try the chatbot. You can also add the RBI Master Direction
on KYC (download it from rbi.org.in, and keep `rbi` in the filename so the eval finds it).

> Thistlewood Bank is fictional. This is a portfolio demo, not affiliated with any bank or
> the Reserve Bank of India, and not financial advice.

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
```

The script builds a temporary index, so it never touches `backend/storage/`. It reports
retrieval hit rate, top-1 hit rate and out-of-scope refusal rate for thresholds 0.00 to 0.70,
suggests a `RELEVANCE_THRESHOLD`, and lists every miss. With `--llm` it also checks that
citations point at returned sources and at the expected page. Full results are written to
`eval/results/latest.json`.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/chat` | none | `{question, history}` → `{answer, sources}` |
| GET | `/api/admin/documents` | `X-Admin-Key` | List indexed documents |
| POST | `/api/admin/documents` | `X-Admin-Key` | Upload one or more PDFs (multipart `files`) |
| DELETE | `/api/admin/documents/{id}` | `X-Admin-Key` | Remove a document and its chunks |
| GET | `/api/health` | none | Health check |

Uploaded files, the Chroma index and the document registry are stored in `backend/storage/`.
