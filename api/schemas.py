# -*- coding: utf-8 -*-
"""
api/schemas.py

Pydantic Request and Response schemas for the ProcurementQA REST & Multi-Agent API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# SUB-ISSUE & QUOTE SCHEMAS (SUPER-ORCHESTRATOR CONTRACT)
# ==============================================================================

class IssueBreakdownItem(BaseModel):
    """Atomic issue evaluation item returned for Super-Orchestrator integration."""
    issue_id: str = Field(..., description="Unique sub-query identifier (e.g. Q1, Q2)")
    topic: str = Field(..., description="High-level topic/aspect of this issue")
    status: str = Field(
        ...,
        description="Resolution status: RESOLVED, OUT_OF_LEGAL_SCOPE, or NO_LAW_FOUND"
    )
    answer: str = Field(..., description="Synthesized legal answer or resolution note for this issue")
    missing_aspect: Optional[str] = Field(
        None,
        description="Explanation of what information/domain is missing if OUT_OF_LEGAL_SCOPE or NO_LAW_FOUND"
    )


class DecisiveQuote(BaseModel):
    """Verbatim legal citation extracted from official gazettes and statutory articles."""
    filename: str = Field(..., description="Source legal markdown / PDF filename")
    page: str = Field(..., description="Section or page locator (e.g. 'ข้อ 25', '4-6/42')")
    law: str = Field(..., description="Statute name and section number (e.g. 'ระเบียบฯ ข้อ 25')")
    quote: str = Field(..., description="Verbatim statutory quote supporting the legal conclusion")


# ==============================================================================
# QA ENDPOINT SCHEMAS
# ==============================================================================

class LegalQARequest(BaseModel):
    """Payload for submitting a procurement legal query."""
    question: str = Field(
        ...,
        description="Legal inquiry or case fact in Thai",
        example="หน่วยงานของรัฐสามารถจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจงในวงเงินไม่เกินเท่าใด และต้องขออนุมัติใครบ้าง"
    )
    mode: str = Field(
        "deep",
        description="Reasoning mode: 'fast' (0 retries, direct answer) or 'deep' (iterative multi-turn validation)",
        example="deep"
    )
    full: bool = Field(
        False,
        description="If True, return all internal LangGraph debug state and candidate chunks"
    )
    org_id: Optional[str] = Field(
        None,
        description="Optional tenant organization ID for data isolation (defaults to X-Organization-Id header or 'DGA')",
        example="DGA"
    )


class LegalQAResponse(BaseModel):
    """Streamlined response schema designed for Super-Orchestrator and frontend clients."""
    status: str = Field(..., description="Overall compliance status (e.g. COMPLIANT, OUT_OF_SCOPE, NO_LAW_FOUND)")
    mode: str = Field(..., description="Executed workflow mode ('fast' or 'deep')")
    direct_answer: str = Field(..., description="Concise, direct answer synthesizing the legal conclusion")
    applicable_laws: List[str] = Field(default_factory=list, description="List of primary laws cited (e.g. ['มาตรา 56', 'ข้อ 25'])")
    decisive_quotes: List[DecisiveQuote] = Field(default_factory=list, description="Decisive verbatim citations supporting the conclusion")
    issues_breakdown: List[IssueBreakdownItem] = Field(default_factory=list, description="Per-issue breakdown for multi-agent delegation")
    exceptions_or_conditions: str = Field("", description="Exceptions, financial thresholds, or prerequisites")
    organization_id: str = Field(..., description="Resolved tenant organization identifier")


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
    organization_id: str
    results: List[Dict[str, Any]] = Field(default_factory=list)


class VerifyComplianceRequest(BaseModel):
    """Payload for rule-based procurement threshold and method verification."""
    procurement_item: str = Field(..., example="จัดซื้อคอมพิวเตอร์และอุปกรณ์ต่อพ่วง")
    estimated_budget: float = Field(..., example=450000.0)
    proposed_method: str = Field(..., example="เฉพาะเจาะจง")
    justification_reason: Optional[str] = Field(None, example="วงเงินไม่เกิน 500,000 บาท")


# ==============================================================================
# PROBES & HEALTH STATUS
# ==============================================================================

class ReadinessStatus(BaseModel):
    """Readiness probe schema for Kubernetes / Docker deployment."""
    ready: bool
    status: str
    model: Optional[str] = None
    sections_indexed: Optional[int] = None
    error: Optional[str] = None
