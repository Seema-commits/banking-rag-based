# Monitoring tab (Admin → Monitoring)

A dashboard for the bot's owner: is it healthy, is it fast, what fails, and what to do about it.
It reads the audit log, so with permanent storage on (docs/storage.md) it covers every question,
including after restarts.

## What it shows

- **Time range:** last 24 hours, 7 days or 30 days.
- **Health tiles:** questions, answer rate, errors, typical (median) and slowest-5% response time
  (cached answers excluded), cache hit rate, 👍/👎, Groq tokens (and estimated cost if you set
  `LLM_COST_PER_MILLION_INPUT_TOKENS` / `LLM_COST_PER_MILLION_OUTPUT_TOKENS` from your Groq plan),
  web searches.
- **Today's usage against limits:** questions, Groq tokens and web searches against the alert
  limits (docs/alerts.md), turning amber at 80% and red at 100%.
- **Suggested actions:** plain-English problems with what to do, most urgent first. For example
  a rejected API key, a high error rate, which step makes answers slow, frequent unanswered
  questions (with an Upload button), many 👎, failed fact-checks, or temporary storage.
- **Charts** (each has "Show as table"):
  - *Questions over time*, by outcome: answered, not answered, errors.
  - *Response time*: median and slowest 5% per hour or day.
  - *Where the time goes*: average time per agent step, slowest first.
- **Problem lists:** errors grouped by cause with the fix, slow answers with their slowest step,
  knowledge gaps, answers marked 👎 with comments, and answers that failed the fact-check.
- **Question details:** click a row to see the full audit record: question, answer shown,
  sources, what the agent did (relevance check, web search, fact-check, cache), time per step,
  model, prompt version, tokens and feedback.

## Actions

- **Clear answer cache:** after fixing a document or prompt, so old answers aren't reused.
- **Export eval candidates:** unanswered and 👎 questions as JSON lines, to label and add to
  `eval/questions.jsonl`.
- **Download audit log (CSV):** every record in the selected period (questions are PII-masked),
  including reviewed and excluded ones, with `reviewed`, `review_note` and `excluded` columns.

## Reviewing problems and excluding test questions

Nothing is ever deleted from the audit log. These only change what the dashboard shows:

| Action | Where | Effect |
|---|---|---|
| **✓ Reviewed** | A row in Slow answers, Knowledge gaps, Answers marked 👎, Failed fact-check; or a record's details, with an optional note | Leaves its list and the suggestions; **still counted** in every number and chart. A knowledge gap marks every question in that group |
| **Exclude from stats** | A record's details | Left out of **every** number, chart and the eval candidates, for your own test questions |
| **Exclude this whole conversation** | A record's details | The same for every question in that chat |
| **Show hidden** | Toolbar | Shows reviewed rows again (greyed, with **Undo**) and lists the excluded questions with **Include again** |

API: `POST /api/admin/interactions/mark` with `{"ids": [...]}` or `{"session_id": "..."}` and
`"reviewed"` and/or `"excluded"` (true to set, false to undo), plus an optional `"note"`.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `SLOW_ANSWER_MS` | 10000 | Answers slower than this are listed as slow and trigger a suggestion |
| `LLM_COST_PER_MILLION_INPUT_TOKENS` | 0 | Groq price per million input tokens, to estimate cost (0 = off) |
| `LLM_COST_PER_MILLION_OUTPUT_TOKENS` | 0 | Groq price per million output tokens |

Each view reads up to the latest 2,000 records in the selected period. Usage counters for
"today" are kept in memory and reset when the app restarts.
