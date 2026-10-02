# -*- coding: utf-8 -*-
"""
api/routes/qa.py

Main Q&A endpoint exposing the agentic RAG workflow to the Super-Orchestrator.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_service, get_tenant_org_id
from api.schemas import QARequest, QAResponse
from core.service import ProcurementService

router = APIRouter(prefix="/api/v1", tags=["Agentic Legal Q&A"])


@router.post(
    "/qa",
    summary="Agentic Procurement Law Inquiry",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK
)
def ask_procurement_qa(
    payload: QARequest,
    service: ProcurementService = Depends(get_service),
    default_tenant_id: str = Depends(get_tenant_org_id)
) -> QAResponse:
    """
    Answers a procurement legal question with the agentic RAG workflow (~1-2 min).

    - `status` and `unresolved_issues` tell the orchestrator what still needs another agent.
    - `grounded` is false when the answer cites a section missing from the retrieved evidence.
    - Tenant: `org_id` in the body, else the `X-Organization-Id` header, else `DEFAULT_ORG_ID`.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty."
        )

    resolved_org_id = payload.org_id.strip() if payload.org_id and payload.org_id.strip() else default_tenant_id

    try:
        raw_result = service.ask_procurement_law(question=payload.query, org_id=resolved_org_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic RAG execution failed: {type(exc).__name__}: {str(exc)}"
        )
    return QAResponse.from_service(raw_result, resolved_org_id)
