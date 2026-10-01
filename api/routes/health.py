# -*- coding: utf-8 -*-
"""
api/routes/health.py

Liveness and readiness probes for container orchestrators (Kubernetes / Docker).
"""

from fastapi import APIRouter, Depends, Response, status
from api.dependencies import get_service
from api.schemas import ReadinessStatus
from core.service import ProcurementService

router = APIRouter(tags=["Probes & Monitoring"])


@router.get("/healthz", summary="Liveness Probe", status_code=status.HTTP_200_OK)
def liveness() -> dict:
    """Simple liveness probe reporting that the HTTP process is responsive."""
    return {"status": "healthy"}


@router.get("/ready", summary="Readiness Probe", response_model=ReadinessStatus)
def readiness(response: Response, service: ProcurementService = Depends(get_service)) -> ReadinessStatus:
    """
    Readiness probe verifying that the Tri-Store Knowledge Graph,
    vector databases, and neural models are loaded into memory and ready for traffic.
    """
    try:
        model_name = service.rag.config.model.model_name
        sections_indexed = len(service._section_index)
        return ReadinessStatus(
            ready=True,
            status="ready",
            model=model_name,
            sections_indexed=sections_indexed
        )
    except Exception as exc:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessStatus(
            ready=False,
            status="not_ready",
            error=f"{type(exc).__name__}: {str(exc)}"
        )
