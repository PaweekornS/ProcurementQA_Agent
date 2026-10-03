#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_mcp_client.py

End-to-end smoke test of the MCP surface exactly as a Super-Orchestrator sees it:
connects over streamable-http, sends the tenant in the X-Organization-Id header, lists
tools/resources/prompts, and calls every public tool. Exits non-zero on any failed check.

Usage:
    python tests/test_mcp_client.py
    python tests/test_mcp_client.py --url http://localhost:8000/mcp --org DGA
    python tests/test_mcp_client.py --run-qa
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

EXPECTED_TOOLS = {
    "get_statute_section",
    "search_procurement_clauses",
    "ask_procurement_law",
}

failures = []


def check(cond: bool, label: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        failures.append(label)


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


async def main(url: str, org: str, run_qa: bool, question: str) -> None:
    print(f"=== Connecting to MCP server at: {url} (X-Organization-Id: {org}) ===")
    async with streamablehttp_client(url, headers={"X-Organization-Id": org}) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print(f"Connected: {init.serverInfo.name}\n")

            print("--- 1. Discovery ---")
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            for t in tools.tools:
                print(f"  [Tool] {t.name:<30} params={list(t.inputSchema.get('properties', {}))}")
            check(EXPECTED_TOOLS <= tool_names, f"public tools exposed: {sorted(EXPECTED_TOOLS)}")
            check(
                all("ctx" not in t.inputSchema.get("properties", {}) for t in tools.tools),
                "request context is not leaked into tool schemas",
            )
            resources = await session.list_resources()
            check(len(resources.resources) >= 2, f"{len(resources.resources)} resources listed")
            thresholds = await session.read_resource("procurement://rules/thresholds")
            check(bool(thresholds.contents and thresholds.contents[0].text.strip()), "thresholds resource readable")
            prompts = await session.list_prompts()
            check(len(prompts.prompts) >= 2, f"{len(prompts.prompts)} prompts listed")

            print("\n--- 3. get_statute_section ---")
            sec = _extract(await session.call_tool("get_statute_section", {"section": "มาตรา 56"}))
            check(sec.get("found") is True, "มาตรา 56 found")
            check(str(sec.get("doc_title", "")).startswith("พระราชบัญญัติ"), f"resolved to the Act: {sec.get('doc_title')}")
            clause = _extract(await session.call_tool("get_statute_section", {"section": "ข้อ 2"}))
            check(clause.get("ambiguous") is True, f"bare 'ข้อ 2' flagged ambiguous ({len(clause.get('other_documents_with_same_number', []))} other docs)")

            print("\n--- 4. search_procurement_clauses ---")
            search = _extract(await session.call_tool("search_procurement_clauses", {
                "query": "วิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000 บาท", "top_k": 3,
            }))
            check(search.get("count", 0) > 0, f"{search.get('count')} clauses returned")
            check(search.get("org_id") == org, f"tenant taken from header (got {search.get('org_id')})")
            spoofed = _extract(await session.call_tool("search_procurement_clauses", {
                "query": "วิธีคัดเลือก", "top_k": 1, "org_id": "SOME_OTHER_ORG",
            }))
            check(spoofed.get("org_id") == org, "header wins over an LLM-supplied org_id argument")

            if run_qa:
                print(f"\n--- 5. ask_procurement_law ---\nQuestion: {question}")
                qa = _extract(await session.call_tool("ask_procurement_law", {"query": question}))
                print(f"  status={qa.get('status')} grounded={qa.get('grounded')} citations={[c.get('law') for c in qa.get('citations', [])]}")
                check(qa.get("status") != "ERROR" and bool(qa.get("answer")), "agentic answer returned")
                check(qa.get("org_id") == org, "QA scoped to header tenant")
                check(any(c.get("filename") and c.get("page") for c in qa.get("citations", [])), "citations carry source file and page")
            else:
                print("\n[Tip] Pass --run-qa to exercise the full agentic answer (~1-2 min).")

    print(f"\n=== {'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'} ===")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP smoke test from the Super-Orchestrator's point of view")
    parser.add_argument("--url", default="http://localhost:8000/mcp", help="FastMCP streamable-http URL")
    parser.add_argument("--org", default="DGA", help="Tenant sent in the X-Organization-Id header")
    parser.add_argument("--run-qa", action="store_true", help="Also run the full ask_procurement_law call")
    parser.add_argument("--question", default="หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงเนื่องจากเป็นพัสดุที่มีวงเงินเล็กน้อยตาม พ.ร.บ. ได้ไม่เกินวงเงินเท่าใด และต้องขอความเห็นชอบรายงานขอซื้อขอจ้างจากใครก่อนจัดซื้อ", help="Test question")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.org, args.run_qa, args.question))
