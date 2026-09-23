FROM python:3.11-slim

WORKDIR /app

# System deps: build tools for a few compiled wheels (torch/sentence-transformers
# extras), plus locales for Thai text handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code. `datas/` (bundled small knowledge base) is copied in;
# `outputs/`, `configs/`, and `procurement_data/` are supplied at runtime via
# volumes/env, see docker-compose.yml.
COPY core/ ./core/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY datas/ ./datas/
COPY run.py mcp_server.py __init__.py ./

ENV PYTHONUNBUFFERED=1 \
    DOTENV_PATH=.env \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

EXPOSE 8000

# Basic liveness check: the server process must be accepting connections on
# MCP_PORT once the model/graph DB have loaded. (Not -f: any HTTP response,
# including the MCP endpoint's expected 4xx on a bare GET, means it's alive.)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -s -o /dev/null http://localhost:${MCP_PORT}/ || exit 1

CMD ["python", "mcp_server.py"]
