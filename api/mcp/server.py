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
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from api.dependencies import get_service
from api.schemas import QAResponse


def _transport_security() -> TransportSecuritySettings:
    """
    FastMCP enables DNS-rebinding protection for localhost only by default, which rejects any
    orchestrator calling us by service name or public hostname. Set MCP_ALLOWED_HOSTS
    (e.g. "procurement-mcp:*,api.example.go.th") to enforce an explicit allow-list.
    """
    hosts = [h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    if not hosts:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts)


mcp = FastMCP(
    "legalgraphrag-procurement",
    instructions=(
        "Thai public procurement law service (พ.ร.บ.การจัดซื้อจัดจ้างฯ 2560, ระเบียบกระทรวงการคลังฯ, "
        "กฎกระทรวง, หนังสือเวียน). Use ask_procurement_law for full legal answers, "
        "get_statute_section for verbatim text of a known มาตรา/ข้อ, search_procurement_clauses for "
        "retrieval without synthesis, and check_procurement_threshold for budget/method checks. "
        "Send the tenant in the X-Organization-Id HTTP header."
    ),
    # Stateless: every call is self-contained, so the service scales across uvicorn workers/replicas
    stateless_http=True,
    streamable_http_path="/mcp",
    transport_security=_transport_security(),
)

_expose_internal = os.getenv("expose_internal_tools", "false").strip().lower() in ("true", "1", "yes")


def _resolve_org_id(ctx: Optional[Context], org_id: Optional[str]) -> str:
    """
    Tenant resolution mirroring the REST API: the X-Organization-Id header set by the calling
    platform wins over the tool argument (which an orchestrator LLM fills in), then DEFAULT_ORG_ID.
    """
    try:
        request = ctx.request_context.request if ctx else None
        header = request.headers.get("x-organization-id") if request is not None else None
    except (AttributeError, LookupError, ValueError):
        header = None
    for candidate in (header, org_id):
        if candidate and candidate.strip():
            return candidate.strip()
    return os.getenv("DEFAULT_ORG_ID", "DGA").strip()


# ==============================================================================
# TIER 1 & 2: ATOMIC RETRIEVAL & GRAPH TRAVERSAL TOOLS
# ==============================================================================

@mcp.tool()
def get_statute_section(
    section: str,
    doc_title: Optional[str] = None,
    org_id: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Verbatim text of one statutory section (มาตรา) or regulation clause (ข้อ). No LLM, ~10 ms.

    Args:
        section: e.g. "มาตรา 56", "ข้อ 79", "มาตรา ๕๖ (๒) (ข)".
        doc_title: Document name or keyword, e.g. "ระเบียบกระทรวงการคลัง". Strongly recommended for
            "ข้อ N": the same clause number exists in many documents. If omitted and the number is
            ambiguous, the result has ambiguous=true and other_documents_with_same_number.
        org_id: Tenant fallback when the X-Organization-Id header is not sent.
    """
    try:
        service = get_service()
        return service.lookup_section(section=section, doc_title=doc_title, org_id=_resolve_org_id(ctx, org_id))
    except Exception as e:
        return {"found": False, "error": f"{type(e).__name__}: {e}"}


@mcp.tool()
def search_procurement_clauses(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
    org_id: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Hybrid retrieval (dense BGE-M3 + Thai BM25 + cross-encoder rerank) over statutes and
    regulations, returning ranked clauses without answer synthesis. Use when the section number
    is unknown or to gather evidence for your own reasoning.

    Args:
        query: Thai description of the topic, e.g. "วิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000 บาท".
        top_k: Number of clauses to return (1-50).
        doc_filter: Optional keyword restricting results, e.g. "พระราชบัญญัติ", "กฎกระทรวง".
        org_id: Tenant fallback when the X-Organization-Id header is not sent.
    """
    resolved_org = _resolve_org_id(ctx, org_id)
    try:
        service = get_service()
        results = service.search_clauses(query=query, top_k=top_k, doc_filter=doc_filter, org_id=resolved_org)
        return {"query": query, "count": len(results), "organization_id": resolved_org, "results": results}
    except Exception as e:
        return {"query": query, "count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}


if _expose_internal:
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
    def get_related_regulations(
        article_name: str,
        max_hops: int = 1,
        org_id: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """Traverse the statutory knowledge graph to find implementing regulations and related circulars."""
        try:
            service = get_service()
            results = service.get_related_clauses(
                section_reference=article_name,
                org_id=_resolve_org_id(ctx, org_id),
                max_hops=max_hops,
            )
            return {"article": article_name, "count": len(results), "results": results}
        except Exception as e:
            return {"article": article_name, "count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}


# ==============================================================================
# TIER 3: REASONING & COMPLIANCE TOOLS
# ==============================================================================

@mcp.tool()
def ask_procurement_law(
    query: str,
    org_id: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Full legal answer from the agentic RAG workflow (issue decomposition, hybrid + graph retrieval,
    self-correcting retries, citation guardrail). Slow (~1-2 min); prefer the retrieval tools when
    only statutory text is needed.

    Args:
        query: The procurement question or case facts, in Thai.
        org_id: Tenant fallback when the X-Organization-Id header is not sent.

    Returns the same contract as REST /api/v1/qa: status, answer, conditions, citations
    [{law, quote}], unresolved_issues (only sub-questions still needing another agent),
    grounded and org_id.
    """
    resolved_org = _resolve_org_id(ctx, org_id)
    try:
        service = get_service()
        result = service.ask_procurement_law(question=query, org_id=resolved_org)
        return QAResponse.from_service(result, resolved_org).model_dump()
    except Exception as e:
        return {
            "status": "ERROR",
            "answer": "",
            "citations": [],
            "unresolved_issues": [],
            "grounded": None,
            "org_id": resolved_org,
            "error": f"{type(e).__name__}: {e}"
        }


@mcp.tool()
def check_procurement_threshold(
    item: str,
    budget: float,
    method: str,
    justification: str = ""
) -> Dict[str, Any]:
    """
    Deterministic check of a procurement method against statutory budget thresholds (~2 ms, no LLM).

    Args:
        item: What is being procured, e.g. "เครื่องคอมพิวเตอร์".
        budget: Estimated budget in THB.
        method: Proposed method, e.g. "เฉพาะเจาะจง", "คัดเลือก", "e-bidding".
        justification: Optional legal ground, e.g. "จำเป็นเร่งด่วน".
    """
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


if _expose_internal:
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
