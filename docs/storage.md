# Permanent storage: audit log and uploaded PDFs

On Cloud Run the container's disk is wiped when it restarts or scales to zero. Without this
setup, feedback, Insights history and PDFs uploaded in Admin disappear at the next restart.

With it:

| What | Where | Kept |
|---|---|---|
| **Audit log**: every question (PII-masked), the answer shown, sources, outcome, the agent's decisions, time per step, tokens, model and prompt version, 👍/👎 and comments | **Firestore** collection `interactions` | 90 days (`INTERACTION_RETENTION_DAYS`), then deleted automatically |
| **Uploaded PDFs** | **Cloud Storage** bucket `<project>-documents` (private) | Until you delete them in Admin |
| Sample Thistlewood PDFs | Built into the container image | Always |
| Vector index and keyword index | Container disk | Rebuilt at startup from the PDFs above |

Locally nothing changes: the app keeps using `backend/storage/interactions.db` (SQLite) and the
local uploads folder unless you set `STORAGE_BACKEND` and `DOCUMENTS_BUCKET`.

## Set up (once)

1. Deploy the code that includes this feature (it adds the Google Cloud client libraries):

   ```bash
   cd ~/banking-rag-based
   git pull
   gcloud run deploy thistlewood-assistant --source . --region us-central1
   ```

2. Create the storage and switch the app to it:

   ```bash
   bash ops/storage/setup_storage.sh
   ```

   It enables Firestore and Cloud Storage, creates the database (in `us-central1`) and a private
   bucket, turns on automatic deletion of old records, gives the app's service account access,
   and updates the Cloud Run service with `STORAGE_BACKEND=firestore` and `DOCUMENTS_BUCKET`.
   It is safe to run again. `DRY_RUN=1 bash ops/storage/setup_storage.sh` prints the commands
   without running them.

3. Check it worked: ask a question on the site, then open **Admin → Insights**. It should say
   **"History: permanent (Firestore)"**. Upload a PDF, then confirm it's listed in the console
   under **Cloud Storage → Buckets → `<project>-documents` → documents/**.

## What's in each audit record

`created_at`, `session_id`, `question` (PII already masked), `answer`, `sources`, `outcome`,
`top_score`, `latency_ms`, `timings_ms` (per step), `tokens`, `decision` (grader verdict),
`grounded`, `regenerations`, `web_searched`, `cached`, `llm_model`, `check_model`,
`prompt_version` (a fingerprint of all prompts, so you can tell which wording produced an answer),
`retrieval_mode`, `relevance_threshold`, `pii_masked` (types only), `rating`, `comment`,
`feedback_at`, `expire_at`.

Browse them in the console: **Firestore → Data → interactions**.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `STORAGE_BACKEND` | `sqlite` | `firestore` for permanent storage |
| `FIRESTORE_DATABASE` | `(default)` | Firestore database id |
| `FIRESTORE_COLLECTION` | `interactions` | Collection for audit records |
| `INTERACTION_RETENTION_DAYS` | `90` | Records are deleted after this many days (0 = keep forever) |
| `DOCUMENTS_BUCKET` | empty | Bucket for uploaded PDFs; empty keeps them on local disk only |

## Cost

For a portfolio demo this stays within the free allowances. At the time of writing, Firestore
includes a free daily quota of reads, writes and storage for one database per project, and
Cloud Storage includes a few GB per month in `us-central1`. Check the current limits on Google
Cloud's pricing pages. Each question writes one record; opening Insights reads up to 2,000.

## Turn it off

```bash
gcloud run services update thistlewood-assistant --region us-central1 --remove-env-vars STORAGE_BACKEND,DOCUMENTS_BUCKET
```

The data stays in Firestore and the bucket until you delete it there.
