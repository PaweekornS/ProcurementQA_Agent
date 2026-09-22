#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mcp_server.py

Exposes LegalGraphRAG's Multi-Agent CRAG pipeline as an MCP (Model Context
Protocol) tool, so this project can act as a sub-agent that a larger
orchestrator (or any MCP-compatible client) calls directly, instead of a
human hitting a REST endpoint.

The RAG pipeline itself (core/crag/pipeline.py) is untouched -- this file is
a thin adapter: it loads LegalGraphRAG once at startup and forwards each
MCP tool call into LegalGraphRAG.analyze_case().

Transports:
    MCP_TRANSPORT=stdio              (default) for local MCP clients
                                      (Claude Desktop, Claude Code, `mcp dev`)
    MCP_TRANSPORT=streamable-http     for remote/orchestrator access over HTTP
                                      (binds MCP_HOST:MCP_PORT, default 0.0.0.0:8000)

Usage:
    python mcp_server.py
    MCP_TRANSPORT=streamable-http python mcp_server.py
"""

import os
import sys
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mcp.server.fastmcp import FastMCP

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig

mcp = FastMCP("legalgraphrag-procurement")

_rag: Optional[LegalGraphRAG] = None


def _get_rag() -> LegalGraphRAG:
    """Lazily initialize LegalGraphRAG once per process (loads the graph DB, model, and indexes)."""
    global _rag
    if _rag is None:
        dotenv_path = os.getenv("DOTENV_PATH", ".env")
        print(f"[mcp_server] Loading LegalGraphRAG config from: {dotenv_path}", file=sys.stderr)
        config = LegalGraphRAGConfig.from_env_file(dotenv_path)
        _rag = LegalGraphRAG(config=config)
        print("[mcp_server] LegalGraphRAG ready.", file=sys.stderr)
    return _rag


def _summarize_citation(law: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entry": law.get("entry", ""),
        "topics": law.get("crimes", law.get("crime", [])),
    }


@mcp.tool()
def ask_procurement_law(question: str) -> Dict[str, Any]:
    """
    Answer a Thai government procurement law question using the Multi-Agent
    CRAG pipeline (issue decomposition -> hybrid retrieval -> synthesis ->
    completeness audit -> targeted refinement retry).

    Args:
        question: A Thai (or English) question about Thai procurement law,
            e.g. "หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงได้ไม่เกินวงเงินเท่าใด".

    Returns:
        A dict with:
            status: "OK" or "NO_LAW_FOUND"
            direct_answer: concise plain-language answer
            legal_reasoning: detailed reasoning grounded in retrieved statutes
            applicable_laws: list of cited statute/section references
            exceptions_or_conditions: caveats or conditions on the answer
            citations: retrieved statute nodes actually used for synthesis
            crag_meta: diagnostics (sub-issues found, retries used, audit completeness)
    """
    if not question or not question.strip():
        return {
            "status": "ERROR",
            "direct_answer": "",
            "legal_reasoning": "",
            "applicable_laws": [],
            "exceptions_or_conditions": "",
            "citations": [],
            "crag_meta": {},
            "error": "`question` must be a non-empty string.",
        }

    rag = _get_rag()
    case = {"fact": question.strip(), "name": "ผู้สอบถาม"}

    try:
        results: List[Dict[str, Any]] = rag.analyze_case(case)
    except Exception as e:
        return {
            "status": "ERROR",
            "direct_answer": "",
            "legal_reasoning": "",
            "applicable_laws": [],
            "exceptions_or_conditions": "",
            "citations": [],
            "crag_meta": {},
            "error": f"{type(e).__name__}: {e}",
        }

    if not results:
        return {
            "status": "ERROR",
            "direct_answer": "",
            "legal_reasoning": "",
            "applicable_laws": [],
            "exceptions_or_conditions": "",
            "citations": [],
            "crag_meta": {},
            "error": "Pipeline returned no results.",
        }

    item = results[0]
    judge_result = item.get("judge_result", {}) or {}
    used_laws = item.get("used_laws", []) or []

    return {
        "status": judge_result.get("status", "OK"),
        "direct_answer": judge_result.get("direct_answer", ""),
        "legal_reasoning": judge_result.get("legal_reasoning", ""),
        "applicable_laws": judge_result.get("applicable_laws", []),
        "exceptions_or_conditions": judge_result.get("exceptions_or_conditions", ""),
        "citations": [_summarize_citation(law) for law in used_laws],
        "crag_meta": item.get("crag_meta", {}),
    }


@mcp.tool()
def healthcheck() -> Dict[str, Any]:
    """Report whether the LegalGraphRAG pipeline is loaded and ready to answer questions."""
    try:
        rag = _get_rag()
        return {
            "ready": True,
            "graph_db_path": rag.config.graph.graph_db_path,
            "model": rag.config.model.model_name,
            "crag_enabled": rag.config.crag.enabled,
            "crag_max_retry": rag.config.crag.max_retry,
        }
    except Exception as e:
        return {"ready": False, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.settings.host = os.getenv("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
