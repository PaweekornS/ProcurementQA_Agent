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

COPY requirements.txt requirements.lock ./
# Install torch from TORCH_INDEX_URL first: with only --extra-index-url, pip may resolve the
# much larger CUDA build from PyPI instead of the CPU wheel.
# requirements.lock pins every (transitive) package to the versions the test suite passed with.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX_URL} -c requirements.lock torch \
    && pip install -r requirements.txt -c requirements.lock

# Pre-download default Cross-Encoder reranker into image cache during build time
# RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
#     m = 'BAAI/bge-reranker-v2-m3'; \
#     print(f'Pre-caching {m}...'); \
#     AutoTokenizer.from_pretrained(m); \
#     AutoModelForSequenceClassification.from_pretrained(m)"

# Create application user and runtime directories (including HuggingFace model cache)
RUN useradd -u 10001 -m -d /app appuser \
    && mkdir -p /app/outputs /app/datasets /app/logs /app/datas /app/.cache/huggingface \
    && chown -R appuser:appuser /app

# Copy application source code
COPY core/ ./core/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY datas/ ./datas/
COPY tests/ ./tests/
COPY run.py server.py ./

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOTENV_PATH=.env \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    HF_HOME=/app/.cache/huggingface

EXPOSE 8000

# Readiness probe: models loaded and Tri-Store reachable + seeded. Long start period covers
# the first-boot download of the local cross-encoder reranker into the HF cache volume.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f -s http://localhost:${MCP_PORT}/ready || exit 1

CMD ["python", "server.py"]
