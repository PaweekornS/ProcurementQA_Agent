# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System dependencies: build tools for compiled wheels (torch/sentence-transformers),
# curl for health probes, and locale libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create application user and runtime directories
RUN useradd -u 10001 -m -d /app appuser \
    && mkdir -p /app/outputs /app/datasets /app/logs /app/datas \
    && chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser core/ ./core/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser evaluation/ ./evaluation/
COPY --chown=appuser:appuser datas/ ./datas/
COPY --chown=appuser:appuser run.py mcp_server.py __init__.py ./

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOTENV_PATH=.env \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

EXPOSE 8000

# Production Readiness Probe: checks whether models, graph DB, and services are loaded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -s http://localhost:${MCP_PORT}/ready || exit 1

CMD ["python", "mcp_server.py"]
