#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
server.py

Production entrypoint for the LegalGraphRAG Thai Procurement Law Service.
Supports:
  1. Default ASGI HTTP Mode: Runs FastAPI + mounted MCP on Uvicorn
  2. Legacy MCP Stdio Mode: Runs FastMCP over standard input/output for local desktop clients
"""

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="ProcurementQA_Agent Production Backend Server")
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "streamable-http"),
        choices=["streamable-http", "sse", "stdio"],
        help="Transport mode (streamable-http runs FastAPI+MCP on Uvicorn; stdio runs MCP over stdin/out)"
    )
    parser.add_argument("--host", default=os.getenv("HOST", os.getenv("MCP_HOST", "0.0.0.0")), help="Bind host")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", os.getenv("MCP_PORT", "8000"))), help="Bind port")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WEB_CONCURRENCY", "1")), help="Number of uvicorn workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-auto-build", action="store_true", help="Disable auto-building graph database on startup")

    args = parser.parse_args()

    if args.no_auto_build:
        os.environ["AUTO_BUILD"] = "False"

    if args.transport == "stdio":
        from api.mcp.server import mcp
        print("[server] Running in MCP stdio transport mode...", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        import uvicorn
        print(f"[server] Launching FastAPI Backend on http://{args.host}:{args.port} (Docs: http://{args.host}:{args.port}/docs)", file=sys.stderr)
        uvicorn.run(
            "api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=os.getenv("LOG_LEVEL", "info").lower()
        )


if __name__ == "__main__":
    main()
