#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mcp_server.py

Production Dual-Protocol (MCP + REST API) Server for Thai Government Procurement Law.

Exposes LegalGraphRAG across 4 specialized tiers:
  Tier 1: Atomic Retrieval & Lookup (get_statute_section, search_procurement_clauses, search_procurement_faqs)
  Tier 2: Knowledge Graph Traversal (get_related_regulations)
  Tier 3: Reasoning & Compliance (ask_procurement_law with fast/deep modes, verify_procurement_compliance)
  Tier 4: Native MCP Resources & Prompts (thresholds, catalog, auditor prompts)

Also provides native REST endpoints on the same port:
  - GET  /healthz
  - GET  /ready
  - POST /api/v1/qa
  - POST /api/v1/search
  - POST /api/v1/verify
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, HTMLResponse

from core.mcp_service import ProcurementService

mcp = FastMCP("legalgraphrag-procurement")

_service: Optional[ProcurementService] = None


def _get_service() -> ProcurementService:
    """Lazily initialize ProcurementService once per process."""
    global _service
    if _service is None:
        dotenv_path = os.getenv("DOTENV_PATH", ".env")
        auto_build_env = os.getenv("AUTO_BUILD")
        auto_build = auto_build_env.lower() in ("true", "1", "yes") if auto_build_env is not None else None
        print(f"[mcp_server] Initializing ProcurementService with config: {dotenv_path}", file=sys.stderr)
        _service = ProcurementService.get_instance(dotenv_path=dotenv_path, auto_build=auto_build)
        print("[mcp_server] ProcurementService ready.", file=sys.stderr)
    return _service


# ==============================================================================
# TIER 1 & 2: ATOMIC RETRIEVAL & LOOKUP TOOLS (INTERNAL / LOW-LEVEL)
# ==============================================================================

_expose_internal = os.getenv("expose_internal_tools", "false").strip().lower() in ("true", "1", "yes")


def get_statute_section(section: str, doc_title: Optional[str] = None) -> Dict[str, Any]:
    """
    Exact verbatim statutory section lookup without generative overhead.

    Args:
        section: Section identifier, e.g. "มาตรา 56", "56", "ข้อ 79", "มาตรา ๕๖ (๒) (ข)".
        doc_title: Optional document name filter (e.g. "พระราชบัญญัติ", "ระเบียบ").

    Returns:
        Structured dictionary containing verbatim section text, topics, and source chunk.
    """
    try:
        service = _get_service()
        return service.lookup_section(section=section, doc_title=doc_title)
    except Exception as e:
        return {"found": False, "error": f"{type(e).__name__}: {e}"}


def search_procurement_clauses(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Direct hybrid search (Dense Vector + Thai BM25) over statutory clauses without LLM synthesis.

    Args:
        query: Search query in Thai or English.
        top_k: Number of ranked clauses to return (default: 5).
        doc_filter: Optional filter keyword (e.g. "พระราชบัญญัติ", "กฎกระทรวง").

    Returns:
        Dictionary with list of ranked statutory clauses and relevance scores.
    """
    try:
        service = _get_service()
        results = service.search_clauses(query=query, top_k=top_k, doc_filter=doc_filter)
        return {"query": query, "count": len(results), "results": results}
    except Exception as e:
        return {"query": query, "count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}


def search_procurement_faqs(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search historical consultation rulings and Comptroller General FAQs.

    Args:
        query: Inquiring question or issue terms.
        top_k: Number of relevant FAQ cases to return (default: 3).

    Returns:
        List of matching official Q&A precedents and cited legal references.
    """
    try:
        service = _get_service()
        results = service.search_faqs(query=query, top_k=top_k)
        return {"query": query, "count": len(results), "results": results}
    except Exception as e:
        return {"query": query, "count": 0, "results": [], "error": f"{type(e).__name__}: {e}"}


def get_related_regulations(section_reference: str) -> Dict[str, Any]:
    """
    Traverse the NetworkX knowledge graph to locate subordinate rules, ministerial
    regulations, or circular letters linked to a parent statutory section.

    Args:
        section_reference: Parent section identifier (e.g. "มาตรา 56", "ข้อ 79").

    Returns:
        Linked nodes, relationship types, and related topics.
    """
    try:
        service = _get_service()
        return service.traverse_regulations(section_reference=section_reference)
    except Exception as e:
        return {"target": section_reference, "error": f"{type(e).__name__}: {e}"}


# Expose low-level tools only if explicitly enabled via expose_internal_tools=true
if _expose_internal:
    mcp.tool()(get_statute_section)
    mcp.tool()(search_procurement_clauses)
    mcp.tool()(search_procurement_faqs)
    mcp.tool()(get_related_regulations)


# ==============================================================================
# TIER 3: HIGH-LEVEL REASONING & COMPLIANCE TOOLS (PRIMARY AGENT INTERFACE)
# ==============================================================================

@mcp.tool()
def procurement_qa(question: str, mode: str = "deep") -> Dict[str, Any]:
    """
    Answer a Thai government procurement law question using the Multi-Agent CRAG pipeline.
    Performs issue decomposition, statutory retrieval, legal synthesis, completeness auditing, and guardrails.

    Args:
        question: Question about Thai procurement law and regulations.
        mode: "deep" (default, full CRAG with completeness audit & refiner retry)
              or "fast" (single-pass hybrid retrieval + synthesis, ~3s).

    Returns:
        Structured answer including status, direct_answer, decisive_quotes,
        applicable_laws, exceptions, and citations.
    """
    try:
        service = _get_service()
        return service.ask_procurement_law(question=question, mode=mode)
    except Exception as e:
        return {
            "status": "ERROR",
            "direct_answer": "",
            "decisive_quotes": [],
            "applicable_laws": [],
            "exceptions_or_conditions": "",
            "citations": [],
            "crag_meta": {},
            "error": f"{type(e).__name__}: {e}",
        }


# Alias for backward compatibility
ask_procurement_law = procurement_qa
if _expose_internal:
    mcp.tool()(ask_procurement_law)


@mcp.tool()
def verify_procurement_compliance(
    procurement_item: str,
    estimated_budget: float,
    proposed_method: str,
    justification_reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate structured procurement project parameters against statutory thresholds
    under the Thai Public Procurement Act B.E. 2560 and Ministerial Regulations.

    Args:
        procurement_item: Description of item or service to procure.
        estimated_budget: Estimated budget in Thai Baht (THB).
        proposed_method: Proposed method (e.g. "เฉพาะเจาะจง", "e-bidding", "คัดเลือก").
        justification_reason: Justification (e.g. "จำเป็นเร่งด่วน", "วงเงินไม่เกิน 500,000").

    Returns:
        Audit verdict: is_compliant, compliance_status (PASSED|FLAGGED|VIOLATION),
        statutory_threshold, required_approvals, and potential_risks.
    """
    try:
        service = _get_service()
        return service.verify_compliance(
            procurement_item=procurement_item,
            estimated_budget=estimated_budget,
            proposed_method=proposed_method,
            justification_reason=justification_reason
        )
    except Exception as e:
        return {
            "procurement_item": procurement_item,
            "estimated_budget": estimated_budget,
            "proposed_method": proposed_method,
            "is_compliant": False,
            "compliance_status": "ERROR",
            "error": f"{type(e).__name__}: {e}"
        }


@mcp.tool()
def healthcheck() -> Dict[str, Any]:
    """Report whether the LegalGraphRAG pipeline and services are loaded and ready."""
    try:
        service = _get_service()
        rag = service.rag
        return {
            "ready": True,
            "graph_db_path": rag.config.graph.graph_db_path,
            "model": rag.config.model.model_name,
            "crag_enabled": rag.config.crag.enabled,
            "crag_max_retry": rag.config.crag.max_retry,
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
    service = _get_service()
    return service.get_thresholds_resource()


@mcp.resource("procurement://catalog/documents", name="Procurement Law Catalog")
def resource_catalog() -> str:
    """Summary of indexed statutes, regulations, and case database statistics."""
    service = _get_service()
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
        f"1. เรียกใช้เครื่องมือ `verify_procurement_compliance` เพื่อตรวจสอบเกณฑ์วงเงินและข้อห้าม\n"
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


# ==============================================================================
# DUAL PROTOCOL: REST API ENDPOINTS (STARLETTE CUSTOM ROUTES)
# ==============================================================================
OPENAPI_SCHEMA = {
    "openapi": "3.0.0",
    "info": {
        "title": "LegalGraphRAG Procurement QA & MCP Server",
        "version": "1.0.0",
        "description": "Production Dual-Protocol (MCP + REST API) Server for Thai Government Procurement Law & Regulations."
    },
    "paths": {
        "/ready": {
            "get": {
                "summary": "Readiness Probe",
                "description": "Reports whether model, embeddings, and Knowledge Graph DB are initialized.",
                "responses": {"200": {"description": "Service Ready"}}
            }
        },
        "/healthz": {
            "get": {
                "summary": "Liveness Probe",
                "description": "Simple health probe for Docker / Kubernetes.",
                "responses": {"200": {"description": "Service Healthy"}}
            }
        },
        "/api/v1/verify": {
            "post": {
                "summary": "Verify Procurement Compliance",
                "description": "Rule-based statutory compliance verification against procurement methods and budget thresholds.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "procurement_item": {"type": "string", "example": "จัดซื้อคอมพิวเตอร์และอุปกรณ์ต่อพ่วง"},
                                    "estimated_budget": {"type": "number", "example": 450000.0},
                                    "proposed_method": {"type": "string", "example": "เฉพาะเจาะจง"},
                                    "justification_reason": {"type": "string", "example": "วงเงินไม่เกิน 500,000 บาท"}
                                },
                                "required": ["procurement_item", "proposed_method"]
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "Compliance evaluation report"}}
            }
        },
        "/api/v1/search": {
            "post": {
                "summary": "Hybrid Statutory Search",
                "description": "Direct hybrid retrieval (Dense Vector + Thai BM25 + Cross-Encoder reranking) over statutory clauses.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "example": "การจัดซื้อจัดจ้างวิธีเฉพาะเจาะจง วงเงินไม่เกินเท่าใด"},
                                    "top_k": {"type": "integer", "example": 5},
                                    "doc_filter": {"type": "string", "example": "พระราชบัญญัติ"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "List of ranked statutory clauses with relevance scores"}}
            }
        },
        "/api/v1/qa": {
            "post": {
                "summary": "Multi-Agent Procurement QA",
                "description": "Streamlined Legal QA returning direct answer and decisive statutory quotes for clean Master Agent integration.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string", "example": "การจัดซื้อจัดจ้างวิธีเฉพาะเจาะจง วงเงินไม่เกินเท่าใด และต้องขออนุมัติใครบ้าง"},
                                    "mode": {"type": "string", "enum": ["fast", "deep"], "example": "fast"},
                                    "full": {"type": "boolean", "example": False, "description": "Optional: return full internal debug metadata"}
                                },
                                "required": ["question"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Streamlined legal response with mode, direct_answer, and decisive_quote"
                    }
                }
            }
        },
    }
}

SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>LegalGraphRAG Procurement QA - Swagger UI</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <link rel="icon" type="image/png" href="https://fastapi.tiangolo.com/img/favicon.png">
    <style>
        body { margin: 0; background: #fafafa; font-family: sans-serif; }
        .topbar { display: none; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout",
                deepLinking: true,
                displayRequestDuration: true
            });
        };
    </script>
</body>
</html>
"""


@mcp.custom_route("/openapi.json", methods=["GET"])
async def route_openapi(request: Request) -> Response:
    """OpenAPI 3.0 specification for Swagger UI."""
    return JSONResponse(OPENAPI_SCHEMA)


@mcp.custom_route("/docs", methods=["GET"])
async def route_docs(request: Request) -> Response:
    """Interactive Swagger UI Documentation."""
    return HTMLResponse(SWAGGER_HTML)


@mcp.custom_route("/healthz", methods=["GET"])
async def route_liveness(request: Request) -> Response:
    """Liveness probe for Docker / Kubernetes."""
    return JSONResponse({"status": "alive", "service": "legalgraphrag-procurement"})


@mcp.custom_route("/ready", methods=["GET"])
async def route_readiness(request: Request) -> Response:
    """Readiness probe: reports whether model and graph DB are initialized."""
    try:
        service = _get_service()
        return JSONResponse({
            "ready": True,
            "status": "ready",
            "model": service.rag.config.model.model_name,
            "sections_indexed": len(service._section_index)
        })
    except Exception as e:
        return JSONResponse({"ready": False, "status": "not_ready", "error": str(e)}, status_code=503)


@mcp.custom_route("/api/v1/qa", methods=["POST"])
@mcp.custom_route("/api/v1/ask", methods=["POST"])
async def route_api_qa(request: Request) -> Response:
    """REST API endpoint for procurement law Q&A returning streamlined response."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    question = body.get("question", "")
    mode = body.get("mode", "deep")
    full_response = bool(body.get("full", False) or request.query_params.get("full", "false").lower() == "true")
    if not question:
        return JSONResponse({"error": "Missing 'question' in request body"}, status_code=400)

    try:
        service = _get_service()
        result = service.ask_procurement_law(question=question, mode=mode)
        if full_response:
            return JSONResponse(result)

        quotes = result.get("decisive_quotes") or []
        clean_result = {
            "mode": result.get("mode", mode),
            "direct_answer": result.get("direct_answer", ""),
            "decisive_quote": quotes,
            "decisive_quotes": quotes,
        }
        return JSONResponse(clean_result)
    except Exception as e:
        return JSONResponse({
            "mode": mode,
            "direct_answer": "",
            "decisive_quote": [],
            "decisive_quotes": [],
            "error": str(e)
        }, status_code=500)


@mcp.custom_route("/api/v1/search", methods=["POST"])
async def route_api_search(request: Request) -> Response:
    """REST API endpoint for hybrid statutory search without synthesis."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    query = body.get("query", "")
    top_k = int(body.get("top_k", 5))
    doc_filter = body.get("doc_filter")

    if not query:
        return JSONResponse({"error": "Missing 'query' in request body"}, status_code=400)

    try:
        service = _get_service()
        results = service.search_clauses(query=query, top_k=top_k, doc_filter=doc_filter)
        return JSONResponse({"query": query, "count": len(results), "results": results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/v1/verify", methods=["POST"])
async def route_api_verify(request: Request) -> Response:
    """REST API endpoint for statutory compliance check."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    item = body.get("procurement_item", "")
    budget = float(body.get("estimated_budget", 0.0))
    method = body.get("proposed_method", "")
    reason = body.get("justification_reason")

    if not item or not method:
        return JSONResponse({"error": "procurement_item and proposed_method are required"}, status_code=400)

    try:
        service = _get_service()
        result = service.verify_compliance(
            procurement_item=item,
            estimated_budget=budget,
            proposed_method=method,
            justification_reason=reason
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LegalGraphRAG Procurement MCP & REST Server")
    parser.add_argument("--transport", default=os.getenv("MCP_TRANSPORT", "stdio"), choices=["stdio", "streamable-http", "sse"], help="MCP transport mode")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"), help="Host for HTTP transport")
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8000")), help="Port for HTTP transport")
    parser.add_argument("--no-auto-build", action="store_true", help="Disable auto-building graph database on startup")
    args = parser.parse_args()

    if args.no_auto_build:
        os.environ["AUTO_BUILD"] = "False"

    if args.transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"[mcp_server] Starting FastMCP on http://{args.host}:{args.port}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http")
    elif args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"[mcp_server] Starting FastMCP SSE on http://{args.host}:{args.port}/sse", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
