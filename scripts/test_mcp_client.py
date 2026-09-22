#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_mcp_client.py

Minimal MCP client for smoke-testing mcp_server.py over the streamable-http
transport (e.g. against `docker compose up`). Lists the exposed tools, then
calls `ask_procurement_law` with a sample Thai procurement question.

Usage:
    python scripts/test_mcp_client.py
    python scripts/test_mcp_client.py --url http://localhost:8000/mcp
    python scripts/test_mcp_client.py --question "หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงได้ไม่เกินวงเงินเท่าใด"
"""

import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_QUESTION = (
    "หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงเนื่องจากเป็นพัสดุที่มีวงเงินเล็กน้อย"
    "ตาม พ.ร.บ. ได้ไม่เกินวงเงินเท่าใด"
)


async def main(url: str, question: str) -> None:
    print(f"Connecting to MCP server at: {url}")
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("\nAvailable tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description.splitlines()[0] if tool.description else ''}")

            print("\nCalling healthcheck()...")
            health = await session.call_tool("healthcheck", {})
            print(json.dumps(_extract(health), ensure_ascii=False, indent=2))

            print(f"\nCalling ask_procurement_law(question={question!r})...")
            result = await session.call_tool("ask_procurement_law", {"question": question})
            print(json.dumps(_extract(result), ensure_ascii=False, indent=2))


def _extract(result) -> dict:
    """FastMCP tool results carry a structured payload in structuredContent
    (or, on older SDKs, as text JSON in content[0].text)."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    if result.content and hasattr(result.content[0], "text"):
        try:
            return json.loads(result.content[0].text)
        except json.JSONDecodeError:
            return {"raw_text": result.content[0].text}
    return {"raw": str(result)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the LegalGraphRAG MCP server")
    parser.add_argument("--url", default="http://localhost:8000/mcp", help="MCP streamable-http endpoint")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Thai procurement law question to ask")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.question))
