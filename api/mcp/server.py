# -*- coding: utf-8 -*-
"""
api/mcp/server.py

FastMCP Server Adapter for LegalGraphRAG Thai Procurement Law.
Exposes specialized tools, resources, and prompt templates for MCP clients
(e.g., Claude Desktop, Cursor, Super-Orchestrator MCP clients).
"""

import json
import os
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from api.dependencies import get_service

mcp = FastMCP("legalgraphrag-procurement")

_expose_internal = os.getenv("expose_internal_tools", "false").strip().lower() in ("true", "1", "yes")


# ==============================================================================
# TIER 1 & 2: ATOMIC RETRIEVAL & GRAPH TRAVERSAL TOOLS
# ==============================================================================

@mcp.tool()
def get_statute_section(section: str, doc_title: Optional[str] = None) -> Dict[str, Any]:
    """Exact verbatim statutory section lookup without generative overhead."""
    try:
        service = get_service()
        return service.lookup_section(section=section, doc_title=doc_title)
    except Exception as e:
        return {"found": False, "error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def search_procurement_clauses(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    org_id: Optional[str] = None
) -> Dict[str, Any]:
    """Direct hybrid search (Dense Vector + Thai BM25) over statutory clauses."""
    try:
        service = get_service()
        results = service.search_clauses(query=query, top_k=top_k, doc_filter=doc_filter, org_id=org_id)
        return {"query": query, "count": len(results), "results": results}
    except Exception as e:
        return {"query": query, "count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def search_procurement_faqs(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search historical consultation rulings and Comptroller General FAQs."""
    try:
        service = get_service()
        results = service.search_faqs(query=query, top_k=top_k)
        return {"query": query, "count": len(results), "results": results}
    except Exception as e:
        return {"query": query, "count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def get_related_regulations(article_name: str, max_hops: int = 1) -> Dict[str, Any]:
    """Traverse the statutory knowledge graph to find implementing regulations and related circulars."""
    try:
        service = get_service()
        results = service.get_related_clauses(article_name=article_name, max_hops=max_hops)
        return {"article": article_name, "count": len(results), "results": results}
    except Exception as e:
        return {"article": article_name, "count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}


# ==============================================================================
# TIER 3: REASONING & COMPLIANCE TOOLS
# ==============================================================================

@mcp.tool()
def ask_procurement_law(
    question: str,
    mode: str = "deep",
    org_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Submits a procurement inquiry to the Pure LangGraph Agentic RAG workflow.
    Returns direct answers, decisive statutory quotes, and per-issue breakdown.
    """
    try:
        service = get_service()
        result = service.ask_procurement_law(question=question, mode=mode, org_id=org_id)
        return {
            "status": result.get("status", "COMPLIANT"),
            "mode": result.get("mode", mode),
            "direct_answer": result.get("direct_answer", ""),
            "applicable_laws": result.get("applicable_laws", []),
            "decisive_quotes": result.get("decisive_quotes", []),
            "issues_breakdown": result.get("issues_breakdown", []),
            "exceptions_or_conditions": result.get("exceptions_or_conditions", ""),
            "organization_id": org_id or "DGA"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "mode": mode,
            "direct_answer": "",
            "applicable_laws": [],
            "decisive_quotes": [],
            "issues_breakdown": [],
            "error": f"{type(e).__name__}: {e}"
        }


@mcp.tool()
def check_procurement_threshold(
    item: str,
    budget: float,
    method: str,
    justification: str = ""
) -> Dict[str, Any]:
    """Rule-based verification of procurement method validity against statutory monetary thresholds."""
    try:
        service = get_service()
        return service.verify_compliance(
            procurement_item=item,
            estimated_budget=budget,
            proposed_method=method,
            justification_reason=justification
        )
    except Exception as e:
        return {"status": "ERROR", "is_compliant": False, "error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def healthcheck() -> Dict[str, Any]:
    """Report whether the LegalGraphRAG pipeline and services are loaded and ready."""
    try:
        service = get_service()
        rag = service.rag
        return {
            "ready": True,
            "graph_db_path": rag.config.graph.graph_db_path,
            "model": rag.config.model.model_name,
            "indexed_sections_count": len(service._section_index)
        }
    except Exception as e:
        return {"ready": False, "error": f"{type(e).__name__}: {e}"}


# ==============================================================================
# TIER 4: NATIVE MCP RESOURCES & PROMPTS
# ==============================================================================

@mcp.resource("procurement://rules/thresholds", name="Procurement Thresholds & Rules")
def resource_thresholds() -> str:
    """Summary of statutory monetary thresholds, methods, and appeal deadlines."""
    service = get_service()
    return service.get_thresholds_resource()


@mcp.resource("procurement://catalog/documents", name="Procurement Law Catalog")
def resource_catalog() -> str:
    """Summary of indexed statutes, regulations, and case database statistics."""
    service = get_service()
    return json.dumps(service.get_catalog_resource(), ensure_ascii=False, indent=2)


@mcp.prompt("audit_procurement_plan")
def prompt_audit_procurement_plan(
    project_name: str,
    estimated_budget: str,
    proposed_method: str,
    justification: str = ""
) -> List[Dict[str, Any]]:
    """Template to audit a draft procurement plan against Thai public procurement regulations."""
    instruction = (
        f"โปรดตรวจสอบความถูกต้องและข้อกำหนดทางกฎหมายของโครงการจัดซื้อจัดจ้างต่อไปนี้:\n"
        f"- ชื่อโครงการ/รายการ: {project_name}\n"
        f"- วงเงินงบประมาณ: {estimated_budget} บาท\n"
        f"- วิธีจัดซื้อจัดจ้างที่เสนอ: {proposed_method}\n"
        f"- เหตุผลความจำเป็น: {justification or 'ไม่มี'}\n\n"
        f"คำสั่งสำหรับ Agent:\n"
        f"1. เรียกใช้เครื่องมือ `check_procurement_threshold` เพื่อตรวจสอบเกณฑ์วงเงินและข้อห้าม\n"
        f"2. หากมีข้อสงสัยเกี่ยวกับมาตราที่เกี่ยวข้อง ให้ค้นหาเพิ่มเติมด้วย `get_statute_section` หรือ `search_procurement_clauses`\n"
        f"3. สรุปผลการตรวจสอบโดยระบุ: สถานะ (ผ่าน/มีความเสี่ยง/ขัดต่อกฎหมาย), ฐานกฎหมายที่รองรับ, ผู้มีอำนาจอนุมัติ, และข้อควรระวังเรื่องการแบ่งซื้อแบ่งจ้าง"
    )
    return [{"role": "user", "content": {"type": "text", "text": instruction}}]


@mcp.prompt("appeal_procedure_advisor")
def prompt_appeal_procedure_advisor(
    procurement_project: str,
    issue_description: str,
    announcement_date: str = ""
) -> List[Dict[str, Any]]:
    """Guidance template for assessing bidder appeal rights and statutory deadlines."""
    instruction = (
        f"โปรดให้คำปรึกษาขั้นตอนการอุทธรณ์ผลการจัดซื้อจัดจ้างภาครัฐ:\n"
        f"- โครงการ: {procurement_project}\n"
        f"- วันที่ประกาศผลในระบบ e-GP: {announcement_date or 'ไม่ระบุ'}\n"
        f"- ประเด็นข้อโต้แย้ง/ความไม่เป็นธรรม: {issue_description}\n\n"
        f"คำสั่งสำหรับ Agent:\n"
        f"1. ตรวจสอบเงื่อนไขการอุทธรณ์ตาม พ.ร.บ. จัดซื้อจัดจ้างฯ 2560 มาตรา 114 - 119\n"
        f"2. ตรวจสอบว่าประเด็นดังกล่าวเข้าข้อยกเว้นที่ห้ามอุทธรณ์ตามมาตรา 115 หรือไม่\n"
        f"3. คำนวณกำหนดเวลา 7 วันทำการและระบุขั้นตอนการยื่นต่อหน่วยงานของรัฐ"
    )
    return [{"role": "user", "content": {"type": "text", "text": instruction}}]
