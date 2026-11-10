# RAG Chatbot

Admins upload PDFs, which are chunked, embedded and stored in a vector database. Users ask
questions, and the chatbot answers from the retrieved chunks and cites its sources.

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| PDF loading / chunking | LangChain `PyPDFLoader`, `RecursiveCharacterTextSplitter` |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (local, open source) |
| Vector DB | Chroma (`langchain-chroma`, persisted to disk) |
| LLM | Groq (`llama-3.3-70b-versatile` by default) |
| Backend | FastAPI (`backend/`) |
| Frontend | React + Vite (`frontend/`) |

## Pipeline

```
Admin upload ─► PyPDFLoader ─► text splitter ─► HF embeddings ─► Chroma

User question ─► [contextualize] ─► [retrieve] ─┬─► [generate]   ─► answer + sources
                  rewrite follow-ups            └─► [no_context] (nothing relevant found)
                  using chat history
```

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
uvicorn app.main:app --reload --port 8000
```

The first start downloads the embedding model (about 90 MB). API docs are at http://localhost:8000/docs.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The **Admin** tab asks for `ADMIN_API_KEY`. Use it to upload and delete PDFs.
The **Chat** tab is the user-facing chatbot.

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/chat` | none | `{question, history}` → `{answer, sources}` |
| GET | `/api/admin/documents` | `X-Admin-Key` | List indexed documents |
| POST | `/api/admin/documents` | `X-Admin-Key` | Upload one or more PDFs (multipart `files`) |
| DELETE | `/api/admin/documents/{id}` | `X-Admin-Key` | Remove a document and its chunks |
| GET | `/api/health` | none | Health check |

Uploaded files, the Chroma index and the document registry are stored in `backend/storage/`.
