# Deploying to Google Cloud Run

This guide puts the Thistlewood Bank Assistant on a public URL using Google Cloud Run.
Everything runs in **Cloud Shell**, a terminal in your browser, so nothing needs to be
installed on your computer. Plan about 30–45 minutes the first time; most of it is waiting
for the first build.

**What gets deployed:** one container in which FastAPI serves both the API and the React app.
At startup it indexes the four Thistlewood PDFs from `sample-docs/thistlewood/`.

**Cost:** Cloud Run has a monthly free allowance (in `us-central1`) that a portfolio demo
stays well inside. The container image is stored in Artifact Registry, which may cost a few
cents a month beyond its free 0.5 GB. A budget alert (step 2) warns you if anything starts
costing money.

---

## 1. Create a Google Cloud project

1. Go to https://console.cloud.google.com and sign in with your Google account.
2. Accept the terms. If you are offered the free trial, you can accept it (it gives credits).
3. At the top, click the project picker → **New Project**.
   Name it `thistlewood-assistant` and click **Create**.
4. Note the **Project ID** shown under the name (it may have numbers added, for example
   `thistlewood-assistant-123456`). You need it below.

## 2. Turn on billing and set a budget alert

Cloud Run needs a billing account even when you stay in the free tier.

1. Menu (☰) → **Billing** → link a billing account to the project (add your card if asked).
2. **Billing** → **Budgets & alerts** → **Create budget**.
   - Scope: your `thistlewood-assistant` project.
   - Amount: a small number, for example ₹100 (or $1).
   - Alert thresholds: 50%, 90%, 100%.
   - Click **Finish**.

A budget sends you an email; it does not stop services. If you ever get an alert you don't
expect, delete the service (see "Stop or remove everything" at the end).

## 3. Open Cloud Shell

Click the **Activate Cloud Shell** icon (`>_`) in the top-right corner of the console.
A terminal opens at the bottom of the page. Run every command below in it.

Set your project (replace with your Project ID):

```bash
gcloud config set project YOUR_PROJECT_ID
```

## 4. Enable the services the app needs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

This takes a minute.

## 5. Store your keys as secrets

Secrets are stored encrypted in Secret Manager and given to the app at runtime. They are
never in the code or the container image.

**Groq API key.** The command waits for you to paste the key; nothing is shown as you paste.
Press Enter afterwards.

```bash
read -rs -p "Paste your Groq API key: " KEY && printf '%s' "$KEY" | gcloud secrets create groq-api-key --data-file=- && unset KEY
```

**Admin key** (the password for the Admin tab). Choose a long, random password that you
don't use anywhere else, and save it in your password manager.

```bash
read -rs -p "Type a new admin password: " KEY && printf '%s' "$KEY" | gcloud secrets create admin-api-key --data-file=- && unset KEY
```

**Let the app read the secrets, and let the build run.** Cloud Run uses the project's
default compute service account:

```bash
PROJECT_NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for s in groq-api-key admin-api-key; do gcloud secrets add-iam-policy-binding $s --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor; done
gcloud projects add-iam-policy-binding $(gcloud config get-value project) --member="serviceAccount:$SA" --role=roles/run.builder
```

## 6. Get the code into Cloud Shell

**If the GitHub repository is public:**

```bash
git clone https://github.com/Seema-commits/banking-rag-based.git
cd banking-rag-based
```

**If it is private**, create a read-only token first: GitHub → Settings → Developer
settings → Personal access tokens → **Fine-grained tokens** → Generate new token.
Give it access to only this repository, with **Contents: Read-only**, and a short expiry.
Then run the clone command above; when asked, enter your GitHub username, and paste the
token as the password.

## 7. Deploy

```bash
gcloud run deploy thistlewood-assistant \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 120 \
  --set-secrets GROQ_API_KEY=groq-api-key:latest,ADMIN_API_KEY=admin-api-key:latest
```

- If asked to create an Artifact Registry repository, answer **Y**.
- The first build takes about 10–15 minutes (it installs PyTorch and downloads the embedding
  model into the image). Later builds are faster.
- When it finishes it prints a **Service URL** like
  `https://thistlewood-assistant-xxxxxxxx-uc.a.run.app`. That is your public link.

What the options mean:

| Option | Why |
|---|---|
| `--region us-central1` | A region covered by the Cloud Run free tier |
| `--allow-unauthenticated` | Anyone with the link can open the demo |
| `--memory 2Gi --cpu 1` | Enough for the embedding model |
| `--cpu-boost` | Faster cold starts |
| `--min-instances 0` | Scales to zero when idle, so idle time costs nothing |
| `--max-instances 1` | Caps cost; keeps rate limits and uploaded documents on one instance |
| `--set-secrets` | Passes the keys from Secret Manager as environment variables |

## 8. Test it

1. Open the Service URL. The first visit after a quiet period takes 20–40 seconds while the
   container starts and indexes the documents. After that it's fast.
2. **Chat:** ask "Can I do video KYC on a Sunday?" and check the answer cites the Video KYC
   guide. Ask "What's the weather?" and check it says it couldn't find anything.
3. **Admin:** sign in with the admin password from step 5. The four Thistlewood PDFs should
   be listed.

## Updating after code changes

```bash
cd ~/banking-rag-based
git pull
gcloud run deploy thistlewood-assistant --source . --region us-central1
```

The other settings and secrets are kept from the first deploy.

## Changing a key

```bash
read -rs -p "Paste the new Groq API key: " KEY && printf '%s' "$KEY" | gcloud secrets versions add groq-api-key --data-file=- && unset KEY
gcloud run deploy thistlewood-assistant --source . --region us-central1
```

## Optional: turn on the web search tool

The agent can search official websites (rbi.org.in, uidai.gov.in, incometax.gov.in,
npci.org.in) when the bank's documents don't answer a question. It uses Tavily
(https://tavily.com, free tier with no card). Create an API key there, then in Cloud Shell:

```bash
read -rs KEY
```

Paste the Tavily key (nothing is shown) and press Enter, then:

```bash
printf %s $KEY | gcloud secrets create tavily-api-key --data-file=- && unset KEY
SA=$(gcloud projects describe $(gcloud config get-value project) --format=value\(projectNumber\))-compute@developer.gserviceaccount.com
gcloud secrets add-iam-policy-binding tavily-api-key --member=serviceAccount:$SA --role=roles/secretmanager.secretAccessor
gcloud run services update thistlewood-assistant --region us-central1 --update-secrets TAVILY_API_KEY=tavily-api-key:latest
```

To change the allowed sites: `--update-env-vars '^|^WEB_SEARCH_DOMAINS=["rbi.org.in","npci.org.in"]'`.
To turn it off: `--update-env-vars WEB_SEARCH_ENABLED=false`.

## Good to know

- **Uploads are temporary.** Cloud Run's disk is wiped when the instance restarts or scales
  to zero. The Thistlewood PDFs are re-indexed automatically; documents uploaded from the
  Admin tab disappear. That's fine for a demo. Persistent storage would need a database or
  Cloud Storage.
- **Limits:** each visitor can send 10 messages a minute, and the whole demo answers up to
  500 questions a day, to protect your Groq quota. Change them with environment variables,
  for example `--update-env-vars CHAT_REQUESTS_PER_DAY=1000`.
- **Settings:** any setting in `backend/.env.example` can be set the same way, for example
  `--update-env-vars RELEVANCE_THRESHOLD=0.3,TOP_K=4`.

## Troubleshooting

See the logs:

```bash
gcloud run services logs read thistlewood-assistant --region us-central1 --limit 50
```

| Problem | Likely cause and fix |
|---|---|
| Build fails with a permission error | Re-run the `roles/run.builder` command from step 5, wait a minute, deploy again |
| Chat says "The AI service rejected the API key" | Wrong Groq key: update it (see "Changing a key") |
| Admin says "Invalid admin key" | Use the admin password from step 5, or set a new one the same way |
| Container fails to start / "memory limit exceeded" in logs | Redeploy with `--memory 4Gi` |
| Page is slow the first time | Normal cold start; set `--min-instances 1` to avoid it, but that is billed continuously |

## Stop or remove everything

Delete the service (stops all Cloud Run charges):

```bash
gcloud run services delete thistlewood-assistant --region us-central1
```

To remove everything, including stored images and secrets, delete the whole project:
console → **IAM & Admin** → **Settings** → **Shut down**.
