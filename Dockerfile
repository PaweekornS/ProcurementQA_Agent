# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System dependencies: build tools for compiled wheels (torch/sentence-transformers),
# curl for health probes, and locale libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

# Install Python dependencies first so this layer is cached across code changes.
# Use BuildKit cache mount and high timeout to prevent network drop failures.
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url ${TORCH_INDEX_URL} -r requirements.txt

# Pre-download default Cross-Encoder reranker into image cache during build time
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    m = 'BAAI/bge-reranker-v2-m3'; \
    print(f'Pre-caching {m}...'); \
    AutoTokenizer.from_pretrained(m); \
    AutoModelForSequenceClassification.from_pretrained(m)"

# Create application user and runtime directories (including HuggingFace model cache)
RUN useradd -u 10001 -m -d /app appuser \
    && mkdir -p /app/outputs /app/datasets /app/logs /app/datas /app/.cache/huggingface \
    && chown -R appuser:appuser /app

# Copy application source code
COPY core/ ./core/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY datas/ ./datas/
COPY run.py mcp_server.py ./

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOTENV_PATH=.env \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    HF_HOME=/app/.cache/huggingface

EXPOSE 8000

# Production Readiness Probe: checks whether models, graph DB, and services are loaded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -s http://localhost:${MCP_PORT}/ready || exit 1

CMD ["python", "mcp_server.py"]
