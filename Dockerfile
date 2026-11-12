# One container: FastAPI serves the API and the built React app.
# Build:  docker build -t thistlewood-assistant .
# Run:    docker run -p 8080:8080 -e GROQ_API_KEY=... -e ADMIN_API_KEY=... thistlewood-assistant

# ---- 1. Build the React frontend ----
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. Python backend ----
FROM python:3.12-slim

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    EMBEDDING_MODEL=${EMBEDDING_MODEL}

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only PyTorch first, so sentence-transformers doesn't pull the much larger CUDA build.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r backend/requirements.txt

# Download the embedding model at build time, so containers start without internet access.
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'])" \
 && chown -R app /app/.cache

COPY backend/ backend/
COPY sample-docs/thistlewood/*.pdf sample-docs/thistlewood/
COPY --from=frontend /app/frontend/dist frontend/dist
RUN mkdir -p backend/storage && chown -R app backend/storage

USER app
WORKDIR /app/backend

# Cloud Run sets PORT (8080 by default). Secrets (GROQ_API_KEY, ADMIN_API_KEY) are passed at runtime.
ENV HOST=0.0.0.0 \
    LOG_FORMAT=json \
    PORT=8080 \
    RELOAD=false \
    HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False \
    SEED_DOCS_DIR=/app/sample-docs/thistlewood

EXPOSE 8080
CMD ["python", "run.py"]
