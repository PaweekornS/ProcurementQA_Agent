# -*- coding: utf-8 -*-
"""
api/routes/

FastAPI APIRouter modules for ProcurementQA_Agent.
"""

from .health import router as health_router
from .qa import router as qa_router
from .search import router as search_router
from .documents import router as documents_router

__all__ = [
    "health_router",
    "qa_router",
    "search_router",
    "documents_router",
]
