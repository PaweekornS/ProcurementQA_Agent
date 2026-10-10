# -*- coding: utf-8 -*-
"""
api/app.py

Main FastAPI Application for ProcurementQA Agent.
Provides dual-protocol serving:
  1. Standard REST API (/api/v1/qa, /api/v1/search, /api/v1/documents, /healthz, /ready)
  2. FastMCP Protocol over Streamable HTTP (/mcp)
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.dependencies import get_service
from api.routes import documents_router, health_router, qa_router, search_router
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
    title="ProcurementQA Agent API",
    description=(
        "Production-grade Multi-Agent RAG Backend for Thai Government Procurement Law & Regulations.\n\n"
        "Exposes:\n"
        "- **Agentic Q&A (/api/v1/qa)**: Pure LangGraph workflow returning direct answers, decisive quotes, and sub-issue breakdown.\n"
        "- **Hybrid Search (/api/v1/search)**: Dense vector + Thai BM25 statutory retrieval.\n"
        "- **MCP Protocol (/mcp)**: Model Context Protocol endpoint (Streamable HTTP) for client agents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS: list the frontend origins in CORS_ALLOW_ORIGINS (comma-separated). Without it any origin
# may call the API, but without credentials: browsers reject wildcard origins with credentials,
# and pairing them would let any website make cookie-authenticated calls.
_cors_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=bool(_cors_origins),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include REST API Routers
app.include_router(health_router)
app.include_router(qa_router)
app.include_router(search_router)
app.include_router(documents_router)

# The FastMCP sub-app already carries its own path (/mcp). Register its routes on the root app
# instead of mounting, which would nest it under /mcp/mcp. The legacy SSE transport is not served:
# it is deprecated in the MCP spec and Streamable HTTP covers every client.
# streamable_http_app() must be built before the lifespan starts: it creates mcp.session_manager.
app.router.routes.extend(mcp.streamable_http_app().routes)


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")
