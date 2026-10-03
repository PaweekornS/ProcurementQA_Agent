# -*- coding: utf-8 -*-
"""
api/schemas.py

Pydantic Request and Response schemas for the ProcurementQA REST & Multi-Agent API.
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# QA ENDPOINT SCHEMAS (SUPER-ORCHESTRATOR CONTRACT, shared by REST /api/v1/qa and MCP)
# ==============================================================================

# Placeholders the synthesizer emits when an issue has nothing missing
_EMPTY_ASPECT = {"", "-", "ไม่มี", "none", "null", "n/a"}


class QARequest(BaseModel):
    """Procurement legal question for the agentic RAG workflow."""
    query: str = Field(
        ...,
        min_length=1,
        description="Legal question or case facts in Thai",
        examples=["หน่วยงานของรัฐสามารถจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจงในวงเงินไม่เกินเท่าใด และต้องขออนุมัติใครบ้าง"],
    )
    org_id: Optional[str] = Field(
        None,
        description="Tenant ID. Falls back to the X-Organization-Id header, then DEFAULT_ORG_ID",
        examples=["DGA"],
    )


class QACitation(BaseModel):
    """A law the answer relies on, with its verbatim supporting text and source location."""
    law: str = Field(..., description="Document and section, e.g. 'พระราชบัญญัติ... พ.ศ. 2560 มาตรา 56'")
    quote: Optional[str] = Field(None, description="Verbatim statutory text supporting the answer")
    filename: Optional[str] = Field(
        None, description="Source OCR document path, e.g. 'พรบ/พระราชบัญญัติ... พ.ศ. 2560.md'; null if not in the corpus"
    )
    page: Optional[str] = Field(None, description="Page range in that document, e.g. '19-20/42'; null if unknown")


class UnresolvedIssue(BaseModel):
    """A sub-question the workflow could not answer, for the orchestrator to delegate elsewhere."""
    issue_id: str = Field(..., description="Sub-question id, e.g. Q2")
    topic: str = Field(..., description="What the sub-question is about")
    status: str = Field(..., description="NO_LAW_FOUND or OUT_OF_LEGAL_SCOPE")
    missing_aspect: Optional[str] = Field(None, description="What information or domain is missing")


class QAResponse(BaseModel):
    """Minimal answer payload the Super-Orchestrator routes on."""
    status: str = Field(..., description="COMPLIANT, PARTIALLY_RESOLVED, NO_LAW_FOUND or OUT_OF_LEGAL_SCOPE")
    answer: str = Field(..., description="Direct legal answer covering all resolved sub-questions")
    conditions: Optional[str] = Field(None, description="Exceptions, thresholds or prerequisites that qualify the answer")
    citations: List[QACitation] = Field(default_factory=list)
    unresolved_issues: List[UnresolvedIssue] = Field(
        default_factory=list, description="Only sub-questions that were NOT resolved; empty when fully answered"
    )
    grounded: Optional[bool] = Field(
        None, description="True when every cited section appears in the retrieved evidence; null if not evaluated"
    )
    org_id: str = Field(..., description="Tenant the answer was scoped to")
    query_id: Optional[str] = Field(None, description="Audit log id (query_audit_logs) for tracing this answer")

    @classmethod
    def from_service(cls, raw: Dict[str, Any], org_id: str) -> "QAResponse":
        """Reduce ProcurementService.ask_procurement_law() output to the orchestrator contract."""
        def norm(law: str) -> str:
            return re.sub(r"\s+", " ", str(law or "")).strip()

        def clean(value: Any) -> Optional[str]:
            value = str(value or "").strip()
            return None if value.lower() in _EMPTY_ASPECT or value == "ไม่ระบุ" else value

        # applicable_laws and decisive_quotes describe the same laws: merge into one list keyed by law
        quotes: Dict[str, Dict[str, Any]] = {}
        for q in raw.get("decisive_quotes") or []:
            if isinstance(q, dict) and norm(q.get("law")):
                quotes.setdefault(norm(q.get("law")), q)
        citations: Dict[str, QACitation] = {}
        for law in list(raw.get("applicable_laws") or []) + list(quotes):
            key = norm(law)
            if key and key not in citations:
                q = quotes.get(key, {})
                citations[key] = QACitation(
                    law=key,
                    quote=clean(q.get("quote")),
                    filename=clean(q.get("filename")),
                    page=clean(q.get("page")),
                )

        unresolved = []
        for idx, item in enumerate(raw.get("issues_breakdown") or [], start=1):
            if not isinstance(item, dict) or item.get("status", "RESOLVED") == "RESOLVED":
                continue
            aspect = (item.get("missing_aspect") or "").strip()
            unresolved.append(UnresolvedIssue(
                issue_id=item.get("issue_id") or f"Q{idx}",
                topic=item.get("topic", ""),
                status=item["status"],
                missing_aspect=None if aspect.lower() in _EMPTY_ASPECT else aspect,
            ))

        verdict = raw.get("guardrail_verdict") or {}
        conditions = (raw.get("exceptions_or_conditions") or "").strip()
        return cls(
            status=raw.get("status") or "NO_LAW_FOUND",
            answer=raw.get("direct_answer", ""),
            conditions=None if conditions.lower() in _EMPTY_ASPECT else conditions,
            citations=list(citations.values()),
            unresolved_issues=unresolved,
            grounded=verdict.get("passed") if "passed" in verdict else None,
            org_id=org_id,
            query_id=raw.get("query_id"),
        )


# ==============================================================================
# SEARCH & VERIFY SCHEMAS
# ==============================================================================

class StatutorySearchRequest(BaseModel):
    """Payload for direct hybrid statutory search."""
    query: str = Field(..., description="Search query in Thai or English", example="วิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000")
    top_k: int = Field(5, ge=1, le=50, description="Number of ranked statutory clauses to return")
    doc_filter: Optional[str] = Field(None, description="Optional filter keyword (e.g. 'พระราชบัญญัติ', 'กฎกระทรวง')")
    org_id: Optional[str] = Field(None, description="Tenant organization ID")


class StatutorySearchResponse(BaseModel):
    """Response payload for statutory clause hybrid search."""
    query: str
    count: int
    org_id: str
    results: List[Dict[str, Any]] = Field(default_factory=list)


# ==============================================================================
# PROBES & HEALTH STATUS
# ==============================================================================

class ReadinessStatus(BaseModel):
    """Readiness probe schema for Kubernetes / Docker deployment."""
    ready: bool
    status: str
    model: Optional[str] = None
    sections_indexed: Optional[int] = None
    tri_store: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
