# MCP server: the knowledge base as tools for AI agents

**MCP (Model Context Protocol)** is an open standard for connecting AI apps to tools. A tool
is wrapped once as an MCP **server**, and any MCP **client** can use it: Claude Desktop, Claude
Code, VS Code, other agents.

This app is an MCP server at `/mcp` on the same Cloud Run service. An agent connected to it can
ask Thistlewood Bank questions and gets the same checked, cited answers as the chat page:

| Tool | What it does | LLM calls |
|---|---|---|
| `ask_thistlewood(question)` | The full agent: PII masking → cache → hybrid search → relevance check → answer → fact-check. Returns the answer, outcome and numbered sources | Yes, like the chat |
| `search_bank_documents(query)` | Hybrid search only: the matching passages with document and page | None |
| `list_documents()` | What the knowledge base contains | None |

All three are **read-only**: nothing can change documents or see account data.

**Guardrails are the same as the chat:**
- personal details are masked
- questions are logged in the audit log with `channel: "mcp"`
- the daily question cap is shared with the chat
- MCP has its own per-minute limit (`MCP_REQUESTS_PER_MINUTE`, default 10)

Monitoring shows a **via MCP** count.

## Turn it on (once)

MCP is **off** until a token is set. Every request must send `Authorization: Bearer <token>`. In
Cloud Shell:

1. Create a random token and store it in Secret Manager. Nothing is printed:

   ```bash
   openssl rand -hex 24 | tr -d '\n' | gcloud secrets create mcp-token --data-file=-
   ```

2. Let the service read it, and pass it to the app:

   ```bash
   NUMBER=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
   gcloud secrets add-iam-policy-binding mcp-token --member="serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" --role=roles/secretmanager.secretAccessor
   gcloud run services update thistlewood-assistant --region us-central1 --update-secrets MCP_TOKEN=mcp-token:latest
   ```

3. The endpoint is your service URL plus `/mcp`, for example
   `https://thistlewood-assistant-…run.app/mcp`.

Treat the token like a password. Anyone with it can spend your Groq quota, within the limits above.
To change it, add a new secret version and run the `services update` command again.

To see the token when a client needs it:

```bash
gcloud secrets versions access latest --secret=mcp-token
```

## Test it

### 1. From Cloud Shell, the quickest check

```bash
cd ~/banking-rag-based && pip install -q "mcp>=2.3,<3"
URL=$(gcloud run services describe thistlewood-assistant --region us-central1 --format='value(status.url)')/mcp
MCP_TOKEN=$(gcloud secrets versions access latest --secret=mcp-token) python3 ops/mcp/try_mcp.py "$URL" "Can I do Video KYC on a Sunday?"
```

It connects like an agent would, lists the three tools, and calls each one. You should see the
document list, the matching passages with their pages, and a cited answer.

### 2. MCP Inspector, a visual test page (needs Node.js on your computer)

```bash
npx @modelcontextprotocol/inspector
```

In the page that opens:
1. Set **Transport** to *Streamable HTTP* and **URL** to your `/mcp` address.
2. Under **Authentication**, set the header `Authorization` to `Bearer <token>`.
3. Click **Connect**, then **Tools** → **List Tools**, pick a tool and click **Run**.

### 3. Claude Code

```bash
claude mcp add --transport http thistlewood https://<your-service>.run.app/mcp --header "Authorization: Bearer <token>"
```

Then ask Claude: *"Using the thistlewood tools, what documents does a 16-year-old need to open an
account?"* Claude calls `ask_thistlewood` and cites the bank's PDFs.

### 4. Claude Desktop

Settings → Developer → **Edit Config**. In `claude_desktop_config.json`, add the following, then
restart Claude Desktop:

```json
{
  "mcpServers": {
    "thistlewood": {
      "command": "npx",
      "args": ["mcp-remote", "https://<your-service>.run.app/mcp", "--header", "Authorization:${MCP_AUTH}"],
      "env": { "MCP_AUTH": "Bearer <token>" }
    }
  }
}
```

`mcp-remote` is a small bridge that lets Claude Desktop reach an HTTP MCP server that uses a bearer
token. It needs Node.js. The tools then appear under the tools icon in a new chat. A screenshot of
Claude answering with your bank's citations makes a good portfolio piece.

## How it's built

- `backend/app/mcp_server.py`: the three tools and the token check. It uses the official
  `mcp` Python SDK, v2 (`MCPServer`; v1 called it `FastMCP`).
- **Streamable HTTP in stateless JSON mode:** there's no server-side session, so it survives
  Cloud Run restarts and scaling.
- **Thread pool:** `ask_thistlewood` runs the agent in a worker thread, so a slow answer doesn't
  block the chat.
- **No DNS-rebinding guard:** the SDK's guard protects unauthenticated servers on a laptop. This
  server requires a token and must accept its public host name, so the guard is off.
- **Tests:** `backend/tests/test_mcp.py` runs the real app and the official MCP client over HTTP.
  It covers all three tools, PII masking, the audit `channel`, the token (401, or 404 when off), and
  the rate limit returned as a tool error.
