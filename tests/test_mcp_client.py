#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_mcp_client.py

Comprehensive test client for the LegalGraphRAG 4-Tier MCP Server.
Connects over streamable-http (default: http://localhost:8000/mcp),
inspects tools, resources, and prompts, and exercises atomic lookups,
compliance checks, and Q&A.

Usage:
    python tests/test_mcp_client.py
    python tests/test_mcp_client.py --url http://localhost:8000/mcp
    python tests/test_mcp_client.py --run-qa
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _extract(result) -> dict:
    """Extract dict payload from MCP tool call result."""
    data = None
    if getattr(result, "structuredContent", None):
        data = result.structuredContent
    elif result.content and hasattr(result.content[0], "text"):
        try:
            data = json.loads(result.content[0].text)
        except json.JSONDecodeError:
            return {"raw_text": result.content[0].text}
    else:
        return {"raw": str(result)}

    if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
        return data["result"]
    return data


async def main(url: str, run_qa: bool, question: str) -> None:
    print(f"=== Connecting to MCP server at: {url} ===")
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print(" Connected successfully!\n")

            # 1. Tools
            tools = await session.list_tools()
            print(f"--- 1. Available Tools ({len(tools.tools)}) ---")
            for t in tools.tools:
                desc = t.description.splitlines()[0] if t.description else ""
                print(f"  [Tool] {t.name:<32} : {desc}")

            # 2. Resources
            try:
                resources = await session.list_resources()
                print(f"\n--- 2. Available Resources ({len(resources.resources)}) ---")
                for r in resources.resources:
                    print(f"  [Resource] {str(r.uri):<32} : {r.name or ''}")

                # Read thresholds resource
                print("\n  Reading resource 'procurement://rules/thresholds' snippet:")
                res_content = await session.read_resource("procurement://rules/thresholds")
                if res_content and res_content.contents:
                    sample = res_content.contents[0].text[:180].replace('\n', ' ')
                    print(f"  > Content preview: {sample}...")
            except Exception as e:
                print(f"\n--- 2. Resources check: {e} ---")

            # 3. Prompts
            try:
                prompts = await session.list_prompts()
                print(f"\n--- 3. Available Prompts ({len(prompts.prompts)}) ---")
                for p in prompts.prompts:
                    print(f"  [Prompt] {p.name:<32} : {p.description or ''}")
            except Exception as e:
                print(f"\n--- 3. Prompts check: {e} ---")

            # 4. Tool Call: healthcheck
            print("\n--- 4. Testing healthcheck() ---")
            health = await session.call_tool("healthcheck", {})
            print(json.dumps(_extract(health), ensure_ascii=False, indent=2))

            # 5. Tool Call: get_statute_section (Atomic Tier 1)
            print("\n--- 5. Testing get_statute_section(section='มาตรา 56') ---")
            sec_res = await session.call_tool("get_statute_section", {"section": "มาตรา 56"})
            extracted_sec = _extract(sec_res)
            print(f"Found: {extracted_sec.get('found')}")
            print(f"Source: {extracted_sec.get('source_id')}")
            print(f"Snippet: {extracted_sec.get('focused_content', '')[:250]}...\n")

            # 6. Tool Call: verify_procurement_compliance (Tier 3)
            print("--- 6. Testing verify_procurement_compliance (Budget: 450,000 THB, Specific Method) ---")
            comp_res = await session.call_tool(
                "verify_procurement_compliance",
                {
                    "procurement_item": "จัดซื้อเครื่องคอมพิวเตอร์และอุปกรณ์ต่อพ่วง",
                    "estimated_budget": 450000.0,
                    "proposed_method": "เฉพาะเจาะจง",
                    "justification_reason": "วงเงินไม่เกิน 500,000 บาท"
                }
            )
            print(json.dumps(_extract(comp_res), ensure_ascii=False, indent=2))

            # 7. Tool Call: search_procurement_faqs (Tier 1)
            print("\n--- 7. Testing search_procurement_faqs(query='ขึ้นทะเบียนผู้ค้างานก่อสร้าง') ---")
            faq_res = await session.call_tool("search_procurement_faqs", {"query": "ขึ้นทะเบียนผู้ค้างานก่อสร้าง", "top_k": 1})
            print(json.dumps(_extract(faq_res), ensure_ascii=False, indent=2))

            # 8. Full Q&A (Optional)
            if run_qa:
                print(f"\n--- 8. Testing ask_procurement_law (mode='fast') ---")
                print(f"Question: {question}")
                qa_res = await session.call_tool("ask_procurement_law", {"question": question, "mode": "fast"})
                print(json.dumps(_extract(qa_res), ensure_ascii=False, indent=2))
            else:
                print("\n[Tip] Pass --run-qa to test generative ask_procurement_law inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test client for LegalGraphRAG 4-Tier MCP Server")
    parser.add_argument("--url", default="http://localhost:8000/mcp", help="FastMCP streamable-http URL")
    parser.add_argument("--run-qa", action="store_true", help="Run full generative ask_procurement_law call")
    parser.add_argument("--question", default="หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงได้ไม่เกินวงเงินเท่าใด", help="Test question")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.run_qa, args.question))
