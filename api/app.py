# -*- coding: utf-8 -*-
"""
api/app.py

Main FastAPI Application for LegalGraphRAG Thai Procurement Law.
Provides dual-protocol serving:
  1. Standard REST API (/api/v1/qa, /api/v1/search, /api/v1/verify, /healthz, /ready)
  2. Mounted FastMCP Protocol (/mcp for streamable-http, /sse for Server-Sent Events)
"""

import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.dependencies import get_service
from api.routes import health_router, qa_router, search_router, verify_router
from api.mcp.server import mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan:
    Pre-warms the Tri-Store Knowledge Graph, Vector DBs, and Cross-Encoder model on server startup.
    """
    print("[api.app] Starting ProcurementQA_Agent Backend Service...", file=sys.stderr)
    try:
        service = get_service()
        print(f"[api.app] Service successfully pre-warmed! (Indexed sections: {len(service._section_index)})", file=sys.stderr)
    except Exception as exc:
        print(f"[api.app] WARNING: Startup pre-warming failed: {exc}", file=sys.stderr)

    # The streamable-HTTP transport needs its session manager task group running for the app's
    # lifetime; without it every /mcp request fails with "Task group is not initialized".
    async with mcp.session_manager.run():
        yield

    print("[api.app] Shutting down ProcurementQA_Agent Backend Service...", file=sys.stderr)


app = FastAPI(
    title="LegalGraphRAG Thai Procurement Law API",
    description=(
        "Production-grade Multi-Agent RAG Backend for Thai Government Procurement Law & Regulations.\n\n"
        "Exposes:\n"
        "- **Agentic Q&A (/api/v1/qa)**: Pure LangGraph workflow returning direct answers, decisive quotes, and sub-issue breakdown.\n"
        "- **Hybrid Search (/api/v1/search)**: Dense vector + Thai BM25 statutory retrieval.\n"
        "- **Compliance Check (/api/v1/verify)**: Rule-based monetary thresholds & procurement method verification.\n"
        "- **MCP Protocol (/mcp, /sse)**: Model Context Protocol endpoints for client agents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware for web orchestrator clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API Routers
app.include_router(health_router)
app.include_router(qa_router)
app.include_router(search_router)
app.include_router(verify_router)

# FastMCP sub-apps already carry their own paths (/mcp, /sse, /messages/). Register their routes on
# the root app instead of mounting, which would nest them under /mcp/mcp and /sse/sse.
# streamable_http_app() must be built before the lifespan starts: it creates mcp.session_manager.
app.router.routes.extend(mcp.streamable_http_app().routes)
app.router.routes.extend(mcp.sse_app().routes)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")
