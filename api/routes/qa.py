# -*- coding: utf-8 -*-
"""
api/routes/qa.py

Main Q&A endpoint exposing the Pure Agentic RAG pipeline to the Super-Orchestrator.
Returns direct answers, decisive verbatim statutory quotes, and per-sub-issue breakdowns.
"""

from typing import Any, Union
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_service, get_tenant_org_id
from api.schemas import LegalQARequest, LegalQAResponse, IssueBreakdownItem, DecisiveQuote
from core.service import ProcurementService

router = APIRouter(prefix="/api/v1", tags=["Agentic Legal Q&A"])


@router.post(
    "/qa",
    summary="Agentic Procurement Law Inquiry",
    response_model=Union[LegalQAResponse, dict],
    status_code=status.HTTP_200_OK
)
def ask_procurement_qa(
    payload: LegalQARequest,
    service: ProcurementService = Depends(get_service),
    default_tenant_id: str = Depends(get_tenant_org_id)
) -> Any:
    """
    Submits a procurement legal inquiry to the Pure LangGraph Agentic RAG workflow.
    
    Features:
    - **Dual-mode reasoning**: 'fast' (zero-retry direct answer) or 'deep' (iterative multi-turn validation).
    - **Multi-Tenant Isolation**: Scopes search and custom clauses to `org_id`.
    - **Super-Orchestrator Interface**: Returns `issues_breakdown` with atomic issue statuses
      (`RESOLVED`, `OUT_OF_LEGAL_SCOPE`, `NO_LAW_FOUND`) and `missing_aspect` for upstream routing.
    - **Verbatim Evidence**: Returns `decisive_quotes` extracted directly from Thai statutory text.
    """
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field cannot be empty."
        )

    resolved_org_id = payload.org_id.strip() if payload.org_id and payload.org_id.strip() else default_tenant_id

    try:
        raw_result = service.ask_procurement_law(
            question=payload.question,
            mode=payload.mode,
            org_id=resolved_org_id
        )

        if payload.full:
            return raw_result

        # Convert raw decisive_quotes and issues_breakdown into Pydantic models
        raw_quotes = raw_result.get("decisive_quotes") or []
        quotes = [
            DecisiveQuote(
                filename=q.get("filename", ""),
                page=q.get("page", ""),
                law=q.get("law", ""),
                quote=q.get("quote", "")
            )
            for q in raw_quotes if isinstance(q, dict)
        ]

        raw_issues = raw_result.get("issues_breakdown") or []
        issues = [
            IssueBreakdownItem(
                issue_id=item.get("issue_id", f"Q{idx}"),
                topic=item.get("topic", "ประเด็นข้อกฎหมาย"),
                status=item.get("status", "RESOLVED"),
                answer=item.get("answer", ""),
                missing_aspect=item.get("missing_aspect")
            )
            for idx, item in enumerate(raw_issues, start=1) if isinstance(item, dict)
        ]

        return LegalQAResponse(
            status=raw_result.get("status", "COMPLIANT"),
            mode=raw_result.get("mode", payload.mode),
            direct_answer=raw_result.get("direct_answer", ""),
            applicable_laws=raw_result.get("applicable_laws", []),
            decisive_quotes=quotes,
            issues_breakdown=issues,
            exceptions_or_conditions=raw_result.get("exceptions_or_conditions", ""),
            organization_id=resolved_org_id
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic RAG execution failed: {type(exc).__name__}: {str(exc)}"
        )
