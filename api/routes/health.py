# -*- coding: utf-8 -*-
"""
api/routes/health.py

Liveness and readiness probes for container orchestrators (Kubernetes / Docker).
"""

import os

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

        from core.graph_construct.hybrid_reranker import reranker_status
        reranker = reranker_status()
        if not reranker["loaded"]:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadinessStatus(
                ready=False,
                status="not_ready",
                model=model_name,
                error="Reranker is enabled but failed to load; retrieval would run unreranked. Check the server log."
            )

        tri_store_stats = None
        if os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes"):
            from core.database import StorageManager
            # Raises if any of PostgreSQL / Qdrant / Neo4j is unreachable
            tri_store_stats = StorageManager.get_instance().get_stats()
            if tri_store_stats["postgres"]["statute_clauses"] == 0 or tri_store_stats["qdrant"]["statutes_points"] == 0:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                return ReadinessStatus(
                    ready=False,
                    status="not_ready",
                    model=model_name,
                    tri_store=tri_store_stats,
                    error="Tri-Store is reachable but not seeded; run scripts/migrate_to_tri_store.py"
                )

        return ReadinessStatus(
            ready=True,
            status="ready",
            model=model_name,
            sections_indexed=sections_indexed,
            tri_store=tri_store_stats
        )
    except Exception as exc:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessStatus(
            ready=False,
            status="not_ready",
            error=f"{type(exc).__name__}: {str(exc)}"
        )
