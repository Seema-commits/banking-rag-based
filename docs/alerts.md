# Owner alerts

Emails to the owner when something needs attention: an API key stops working, a quota is hit,
answers fail or slow down, or daily usage nears its limit. Plus the Google Cloud budget alert
for cloud costs.

All of this is free within Google Cloud's monitoring allowance.

## Set up (once)

In Cloud Shell:

```bash
cd ~/banking-rag-based
git pull
python3 ops/alerts/setup_alerts.py --email you@example.com
```

It creates an email channel, six log-based metrics and six alert policies (all named
`[Thistlewood] …`), and checks your budget. Running it again is safe: existing items are
skipped. Add `--dry-run` to see the plan first.

You can see and edit everything in the console: **Monitoring → Alerting** (policies and
incidents) and **Logging → Log-based metrics**. To get push notifications on your phone, install
the Google Cloud app and add it as a notification channel in Monitoring → Alerting → Edit
notification channels.

## What each email means, and what to do

| Email | Meaning | What to do |
|---|---|---|
| **Groq API key rejected** | Groq returned 401: the key is wrong, expired or revoked. The bot can't answer anything. | Create a new key at console.groq.com/keys, then update the secret (commands below). |
| **Groq rate limit or quota reached** | Groq refused requests (429). Free-tier limits were reached or traffic spiked. | Check usage at console.groq.com. Lower `CHAT_REQUESTS_PER_DAY`, wait for the limit to reset, or upgrade the plan. |
| **Chat errors** | More than 3 questions failed in 10 minutes. | Logs Explorer: `jsonPayload.event="chat" AND severity>=ERROR`. Look at `outcome` and `error_type`. |
| **Web search failing** | More than one Tavily search failed in 15 minutes. | Check `reason` in the `web_search_failed` log line: **auth** = update the Tavily key; **quota** = credits used up (tavily.com dashboard); **timeout / network** = Tavily is down, usually temporary. |
| **Slow answers** | The slowest 5% of answers took longer than 10 seconds over 15 minutes. | Logs Explorer: `jsonPayload.event="chat"`, compare `grade_ms`, `web_search_ms`, `generate_ms`, `grounding_ms` to find the slow step. Cold starts also show up here. |
| **Daily usage near limit** | A daily counter reached 80% or 100% of its limit. | The `usage_threshold` log line shows `metric` (questions, llm_tokens, web_searches), `used` and `limit`. At 100% of questions, the bot stops answering until midnight UTC. |
| **Budget alert** (from Google Billing) | Google Cloud costs reached 50 / 90 / 100% of your budget. | Billing → Reports shows which service. A budget email does **not** stop spending; delete the Cloud Run service if needed. |

When the problem stops, Google sends a second email saying the incident is resolved.

## Update a key

```bash
read -rs KEY
```

Paste the new key (nothing is shown) and press Enter, then (use `tavily-api-key` for Tavily):

```bash
printf %s $KEY | gcloud secrets versions add groq-api-key --data-file=- && unset KEY
gcloud run services update thistlewood-assistant --region us-central1 --update-secrets GROQ_API_KEY=groq-api-key:latest
```

## Usage limits

Set these to match your Groq and Tavily plans (Google Cloud can't see their usage, so the app
counts it). Counters reset at midnight UTC.

| Setting | Default | Meaning |
|---|---|---|
| `CHAT_REQUESTS_PER_DAY` | 500 | Questions per day; the bot stops answering at 100% |
| `DAILY_LLM_TOKEN_WARNING` | 300000 | Groq tokens per day that trigger the warning (no blocking) |
| `DAILY_WEB_SEARCH_WARNING` | 30 | Tavily searches per day (the free tier is about 1,000 a month) |

```bash
gcloud run services update thistlewood-assistant --region us-central1 --update-env-vars DAILY_LLM_TOKEN_WARNING=200000
```

## Send yourself a test alert

Temporarily set a tiny token limit, ask one question, and wait 5–10 minutes for the email:

```bash
gcloud run services update thistlewood-assistant --region us-central1 --update-env-vars DAILY_LLM_TOKEN_WARNING=10
```

Open the site and ask any question (not a greeting). Once the email arrives, remove the test
setting:

```bash
gcloud run services update thistlewood-assistant --region us-central1 --remove-env-vars DAILY_LLM_TOKEN_WARNING
```

## Limits of these alerts

- Usage counters are kept in memory per instance and reset when the container restarts, so
  they are an early warning, not an exact bill. Check console.groq.com and the Tavily dashboard
  for exact usage.
- Log-based metrics take a few minutes to update, so alerts arrive a few minutes after the event.
