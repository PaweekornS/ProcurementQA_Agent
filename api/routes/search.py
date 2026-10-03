# -*- coding: utf-8 -*-
"""
api/routes/search.py

Direct statutory retrieval endpoint using Dense Vector + Thai BM25 + Cross-Encoder reranking.
"""

from fastapi import Request, APIRouter, Depends, HTTPException, status
from api.errors import internal_error
from api.dependencies import get_service, get_tenant_org_id
from api.schemas import StatutorySearchRequest, StatutorySearchResponse
from core.service import ProcurementService

router = APIRouter(prefix="/api/v1", tags=["Statutory Hybrid Search"])


@router.post(
    "/search",
    summary="Hybrid Statutory Search",
    response_model=StatutorySearchResponse,
    status_code=status.HTTP_200_OK
)
def search_statutory_clauses(
    request: Request,
    payload: StatutorySearchRequest,
    service: ProcurementService = Depends(get_service),
    default_tenant_id: str = Depends(get_tenant_org_id)
) -> StatutorySearchResponse:
    """
    Direct hybrid search over statutory clauses without generative LLM overhead.
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
        results = service.search_clauses(
            query=payload.query,
            top_k=payload.top_k,
            doc_filter=payload.doc_filter,
            org_id=resolved_org_id
        )
        return StatutorySearchResponse(
            query=payload.query,
            count=len(results),
            org_id=resolved_org_id,
            results=results
        )
    except Exception as exc:
        raise internal_error(exc, "Hybrid search")
