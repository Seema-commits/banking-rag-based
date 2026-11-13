# Observability on Google Cloud

The app writes **one structured log line per chat request** (and a few other events). On
Cloud Run these lines are JSON (`LOG_FORMAT=json` in the Docker image), so Cloud Logging
stores each field separately. From them we build **log-based metrics**, a **dashboard**
and an **error alert**.

Logs never contain question text, answers or document contents: only numbers and flags.
Masked personal data is reported by type only (for example `"pii_masked": ["AADHAAR"]`).

## What gets logged

| Event | When | Main fields |
|---|---|---|
| `chat` | Every chat request | `outcome`, `latency_ms`, `retrieval_ms`, `rewrite_ms`, `generate_ms`, `llm_ms`, `llm_calls`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `top_score`, `chunks_kept`, `keyword_only_chunks`, `retrieval_mode`, `sources`, `rewritten`, `history_messages`, `question_chars`, `pii_masked`, `llm_model`, `error_type` |
| `rate_limited` | A request is blocked by the rate limiter | `scope` (`per_ip_minute` or `daily`) |
| `feedback` | A user clicks 👍 or 👎 | `rating`, `has_comment` |
| `startup` | A new container starts | `duration_ms`, `embedding_model`, `llm_model`, `top_k`, `relevance_threshold`, `documents` |
| `seed` | Sample PDFs indexed at startup | `documents`, `chunks`, `duration_ms` |
| `document_indexed` / `document_rejected` | Admin upload | `pages`, `chunks`, `size_kb`, `duration_ms` / `reason` |

`outcome` is one of: `answered`, `not_found` (no relevant documents), `unverified` (answer
failed the grounding check), `needs_account_access`, `financial_advice`, `off_topic`, `unsafe`
(refused by the grader), `llm_auth_error` (bad Groq key), `llm_rate_limited` (Groq quota),
`llm_not_configured`, `error` (anything else; see `error_type`).

Agent fields: `decision` (grader verdict), `grade_ms`, `query_rewrites`, `query_rewrite_ms`,
`grounded`, `grounding_ms`, `unsupported_claims`, `regenerations`, and `grader_error` /
`grounding_error` when an LLM check returned something unusable (the agent then fails open).

Locally, logs stay readable text with the same fields appended as `key=value`.

## 1. Deploy the new version

In Cloud Shell:

```bash
cd ~/banking-rag-based
git pull
gcloud run deploy thistlewood-assistant --source . --region us-central1
```

Open the site and ask a few questions so there is something to see.

## 2. Look at the logs

Console → **Logging** → **Logs Explorer**. Paste a query and click **Run query**:

| To see | Query |
|---|---|
| All chat requests | `jsonPayload.event="chat"` |
| Questions with no relevant documents | `jsonPayload.outcome="not_found"` |
| Questions the grader refused | `jsonPayload.event="chat" AND jsonPayload.decision!="answer"` |
| Answers that failed the grounding check | `jsonPayload.grounded=false` |
| Errors only | `jsonPayload.event="chat" AND severity>=ERROR` |
| Slow answers (over 5 s) | `jsonPayload.latency_ms>5000` |
| Messages where an Aadhaar number was masked | `jsonPayload.pii_masked="AADHAAR"` |
| Container starts (cold starts) | `jsonPayload.event="startup"` |

Click a line and expand `jsonPayload` to see every field.

## 3. Create the log-based metrics

These turn log fields into numbers Google can chart and alert on. Run once, in Cloud Shell,
from the project folder:

```bash
cd ~/banking-rag-based
for m in chat_requests chat_latency llm_tokens retrieval_top_score rate_limited; do gcloud logging metrics create $m --config-from-file=ops/monitoring/$m.yaml; done
```

| Metric | What it shows |
|---|---|
| `chat_requests` | Number of chat requests, split by `outcome` |
| `chat_latency` | Distribution of end-to-end response time (ms) |
| `llm_tokens` | Distribution of Groq tokens per request |
| `retrieval_top_score` | Distribution of the best relevance score per question |
| `rate_limited` | Requests blocked by the rate limiter, split by `scope` |

Metrics only count log lines written **after** they are created, so charts start empty.

## 4. Create the dashboard

```bash
gcloud monitoring dashboards create --config-from-file=ops/monitoring/dashboard.json
```

Open it from Console → **Monitoring** → **Dashboards** → **Thistlewood Assistant**. It
shows requests by outcome, latency (median and 95th percentile), tokens per request,
retrieval scores and rate-limited requests.

If the command reports an error, you can build the same charts by hand: **Monitoring** →
**Metrics explorer** → search for `logging/user/chat_requests` (and the others) → **Save
chart** to a new dashboard.

## 5. Add an email alert for errors

Console → **Monitoring** → **Alerting** → **Create policy**:

1. **Select a metric**: search `chat_requests` → choose **logging/user/chat_requests** → **Apply**.
2. **Add filter**: `outcome` **=~** `llm_.*|error` (the failure outcomes; refusals like
   `off_topic` or `not_found` are normal behaviour, not errors).
3. Rolling window **5 min**, function **sum**.
4. **Configure trigger**: threshold, **Above threshold**, value **0**.
5. **Notifications**: **Manage notification channels** → add **Email** with your address →
   come back and select it.
6. Name it `Chat errors` → **Create policy**.

You get an email whenever a chat request fails (for example, the Groq key stops working).

## Cost

All of this fits in the free allowances for a demo: Cloud Logging includes the first 50 GiB
of logs per month, and Cloud Monitoring includes a free monthly allowance for log-based
metrics. Dashboards and alert policies are free.

Avoid **uptime checks** for this service: pinging it every few minutes keeps waking the
container (each cold start loads the embedding model), which uses up the free Cloud Run
allowance. The error alert above covers failures without extra traffic.
