# -*- coding: utf-8 -*-
"""
api/routes/documents.py

Tenant-private document store (OCR markdown -> chunks -> Tri-Store). Documents are visible only
to the tenant that owns them and are searched alongside the statutes by /qa, /search and MCP.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from api.errors import internal_error
from core import tenant_documents

router = APIRouter(prefix="/api/v1/documents", tags=["Tenant Documents"])


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Document title shown in citations")
    markdown: str = Field(..., min_length=1, description="Page-aware markdown from the OCR service")
    source_id: Optional[str] = Field(None, description="Stable id from the caller; re-sending it replaces the document")
    source_file: Optional[str] = Field(None, description="Source path reported in citations")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    org_id: Optional[str] = Field(None, description="Tenant; the X-Organization-Id header takes precedence")


class DocumentIngestResponse(BaseModel):
    doc_id: str
    org_id: str
    title: str
    chunks: int
    total_pages: Optional[int] = None
    graph_status: Optional[str] = None
    graph_citations: int = 0


def _require_tri_store() -> None:
    if os.getenv("USE_TRI_STORE", "false").lower() not in ("true", "1", "yes"):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Tenant documents require USE_TRI_STORE=true")


def _tenant(header_org: Optional[str], body_org: Optional[str] = None) -> str:
    """Writes never fall back to DEFAULT_ORG_ID: an unscoped call must not land in another tenant."""
    org = (header_org or "").strip() or (body_org or "").strip()
    if not org:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "X-Organization-Id header is required")
    if not tenant_documents.ORG_ID_PATTERN.match(org):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid organization id")
    return org


@router.post("", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED,
             summary="Ingest or replace a tenant document")
def ingest_document(
    payload: DocumentIngestRequest,
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-Id"),
) -> DocumentIngestResponse:
    _require_tri_store()
    org_id = _tenant(x_organization_id, payload.org_id)
    try:
        result = tenant_documents.ingest_document(
            org_id=org_id,
            title=payload.title.strip(),
            markdown=payload.markdown,
            source_id=payload.source_id,
            source_file=payload.source_file,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except Exception as exc:
        raise internal_error(exc, "Document ingestion")
    return DocumentIngestResponse(**result)


@router.get("", summary="List the tenant's documents")
def list_documents(
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-Id"),
) -> Dict[str, Any]:
    _require_tri_store()
    org_id = _tenant(x_organization_id)
    try:
        docs: List[Dict[str, Any]] = tenant_documents.list_documents(org_id)
    except Exception as exc:
        raise internal_error(exc, "Document listing")
    return {"org_id": org_id, "count": len(docs), "documents": docs}


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a tenant document")
def delete_document(
    doc_id: str,
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-Id"),
) -> None:
    _require_tri_store()
    org_id = _tenant(x_organization_id)
    try:
        deleted = tenant_documents.delete_document(org_id, doc_id)
    except Exception as exc:
        raise internal_error(exc, "Document deletion")
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
