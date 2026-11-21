"""Try the MCP server like an AI agent would: list its tools, search, and ask a question.

    pip install "mcp>=2.3,<3"
    MCP_TOKEN=... python3 ops/mcp/try_mcp.py https://<your-service>.run.app/mcp "Can I do Video KYC on a Sunday?"

In Cloud Shell, read the token from Secret Manager instead of typing it:

    MCP_TOKEN=$(gcloud secrets versions access latest --secret=mcp-token) python3 ops/mcp/try_mcp.py <url>
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client


def show(title: str, result) -> None:
    print(f"\n=== {title} ===")
    if result.is_error:
        print("ERROR:", " ".join(getattr(c, "text", "") for c in result.content))
        return
    data = result.structured_content or json.loads(result.content[0].text)
    print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])


async def main(url: str, question: str) -> None:
    token = os.environ.get("MCP_TOKEN", "")
    if not token:
        sys.exit("Set MCP_TOKEN (see the docstring at the top of this file).")
    async with create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}) as http, \
            streamable_http_client(url, http_client=http) as (read, write, *_), \
            ClientSession(read, write) as session:
        init = await session.initialize()
        print(f"Connected to {init.server_info.name} {init.server_info.version}")
        tools = (await session.list_tools()).tools
        print("Tools:")
        for t in tools:
            print(f"  - {t.name}: {(t.description or '').splitlines()[0]}")
        show("list_documents", await session.call_tool("list_documents", {}))
        show(f"search_bank_documents: {question}", await session.call_tool("search_bank_documents", {"query": question}))
        show(f"ask_thistlewood: {question}", await session.call_tool("ask_thistlewood", {"question": question}))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    try:
        asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Can I do Video KYC on a Sunday?"))
    except Exception as exc:  # usually a wrong token (401), MCP not enabled (404) or a wrong URL
        sys.exit(f"Could not use the MCP server: {exc!r}\nCheck the URL ends in /mcp, that MCP_TOKEN is "
                 "the server's token, and that the service has MCP_TOKEN set (docs/mcp.md).")
