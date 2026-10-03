# -*- coding: utf-8 -*-
"""
api/routes/qa.py

Main Q&A endpoint exposing the agentic RAG workflow to the Super-Orchestrator.
"""

from fastapi import Request, APIRouter, Depends, HTTPException, status

from api.dependencies import get_service, get_tenant_org_id
from api.errors import internal_error, service_busy
from api.schemas import QARequest, QAResponse
from core.service import ProcurementService, ServiceBusyError

router = APIRouter(prefix="/api/v1", tags=["Agentic Legal Q&A"])


@router.post(
    "/qa",
    summary="Agentic Procurement Law Inquiry",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK
)
def ask_procurement_qa(
    request: Request,
    payload: QARequest,
    service: ProcurementService = Depends(get_service),
    default_tenant_id: str = Depends(get_tenant_org_id)
) -> QAResponse:
    """
    Answers a procurement legal question with the agentic RAG workflow (~1-2 min).

    - `status` and `unresolved_issues` tell the orchestrator what still needs another agent.
    - `grounded` is false when the answer cites a section missing from the retrieved evidence.
    - Tenant: the `X-Organization-Id` header, else `org_id` in the body, else `DEFAULT_ORG_ID`.
    - 503 + Retry-After when all QA slots (QA_MAX_CONCURRENCY) are busy.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty."
        )

    # X-Organization-Id (set by the calling platform) wins over the body, matching the MCP adapter
    header_org = request.headers.get("x-organization-id", "").strip()
    body_org = payload.org_id.strip() if payload.org_id else ""
    resolved_org_id = header_org or body_org or default_tenant_id

    try:
        raw_result = service.ask_procurement_law(question=payload.query, org_id=resolved_org_id)
    except ServiceBusyError as exc:
        raise service_busy(exc)
    except Exception as exc:
        raise internal_error(exc, "Agentic RAG execution")
    return QAResponse.from_service(raw_result, resolved_org_id)
