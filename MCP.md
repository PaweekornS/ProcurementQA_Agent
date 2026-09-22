# Running LegalGraphRAG as an MCP Sub-Agent

This document covers `mcp_server.py`: it exposes the CRAG pipeline as an [MCP](https://modelcontextprotocol.io) tool so a larger orchestrator (or any MCP-compatible client) can call it directly, plus how to build, run, and test it via Docker.

For what the pipeline itself does, see [REPORT.md](REPORT.md). For the general project layout, see [README.md](README.md).

---

## 1. What was added


| File                         | Purpose                                                                                                                                            |
| :----------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mcp_server.py`              | MCP server. Loads`LegalGraphRAG` once, exposes it as two tools: `ask_procurement_law(question)` and `healthcheck()`.                               |
| `Dockerfile`                 | Builds a container with all runtime deps (torch, sentence-transformers, pythainlp, rank_bm25, mcp) and the app code.                               |
| `docker-compose.yml`         | Runs the server over the`streamable-http` transport on port 8000, mounting `outputs/`, `datasets/`, `procurement_data/`, `configs/` from the host. |
| `scripts/test_mcp_client.py` | A minimal Python MCP client that lists tools and calls`ask_procurement_law` against a running server — the automated way to smoke-test it.        |

`mcp_server.py` does not change any CRAG logic — it's a thin adapter around `LegalGraphRAG.analyze_case()` (see `core/LegalGraphRAG.py`), which already returns the same `judge_result` / `crag_meta` structure `run.py` produces for the benchmark.

### Tool contract

**`ask_procurement_law(question: str) -> dict`**

```json
{
  "status": "OK",
  "direct_answer": "...",
  "legal_reasoning": "...",
  "applicable_laws": ["มาตรา ๕๖ (๒) (ข)", "ข้อ ๗๙"],
  "exceptions_or_conditions": "...",
  "citations": [{"entry": "...", "topics": ["..."]}],
  "crag_meta": {"issues": [...], "retries": 0, "audited_complete": true}
}
```

`status` is `"NO_LAW_FOUND"` when the 3-tier guardrail determines the question is out of scope, or `"ERROR"` if the call itself failed (bad input, exception during inference) — check `error` in that case.

**`healthcheck() -> dict`** — reports whether the model/graph DB loaded successfully, without running inference. Use this first when debugging a new deployment.

---

## 2. Prerequisites

1. `.env` (project root) — copy from `env.example` and fill in `OPENROUTER_API_KEY`:
   ```bash
   cp env.example .env
   ```
   `mcp_server.py` and `docker-compose.yml` both read `.env` by default (`DOTENV_PATH=.env`). `configs/thai_procurement.env` is not used by the MCP server — that path is only relevant if you're invoking `run.py`/`index_knowledge_base.py` directly with their own `--dotenv_path`/`--config` defaults.
2. `procurement_data/` — statute markdown, FAQ Excel, and `qa_*.csv` benchmark files (gitignored, supply locally).
3. A built graph DB at `outputs/openrouter_graph_db.pkl` — build it once with:
   ```bash
   python scripts/index_knowledge_base.py --config .env --force
   ```

   (Or let the container build it on first run — `auto_build=True` by default in `env.example` — but that adds startup latency and needs `procurement_data/` mounted, which `docker-compose.yml` already does.)

---

## 3. Build & run with Docker

```bash
docker compose build
docker compose up
```

This starts the MCP server on `http://localhost:8000/mcp` using the `streamable-http` transport. Watch the logs for:

```
[mcp_server] Loading LegalGraphRAG config from: .env
[mcp_server] LegalGraphRAG ready.
```

To run without Docker (e.g. local development):

```bash
pip install -r requirements.txt
python mcp_server.py                       # stdio transport (default)
MCP_TRANSPORT=streamable-http python mcp_server.py   # HTTP transport on :8000
```

---

## 4. How other people can test it

### Option A — MCP Inspector (fastest, no code)

The `mcp` Python package ships a dev inspector with a web UI for calling tools by hand:

```bash
pip install "mcp[cli]"
mcp dev mcp_server.py
```

This opens a browser UI where you can call `ask_procurement_law` and `healthcheck` directly, see raw request/response JSON, and inspect the tool schemas — the easiest way for someone unfamiliar with the codebase to poke at it.

### Option B — the included test client (automated / CI)

Against a running `docker compose up` server:

```bash
python scripts/test_mcp_client.py
python scripts/test_mcp_client.py --question "ผู้มีสิทธิอุทธรณ์ผลการจัดซื้อจัดจ้างต้องยื่นภายในกี่วัน"
```

It lists available tools, calls `healthcheck`, then calls `ask_procurement_law` and prints the structured JSON response. Point `--url` at a remote deployment to test that instead of localhost.

### Option C — plug it into an orchestrator / Claude Desktop / Claude Code

For **stdio** clients (Claude Desktop, Claude Code, most local agent frameworks), add an MCP server entry pointing at the script directly, e.g. for Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "legalgraphrag-procurement": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "env": { "DOTENV_PATH": "/absolute/path/to/.env" }
    }
  }
}
```

For an orchestrator that talks HTTP (the more likely case for "a big orchestrator working with sub-agents"), point its MCP client config at the running container's endpoint instead:

```json
{
  "mcpServers": {
    "legalgraphrag-procurement": {
      "url": "http://<host>:8000/mcp",
      "transport": "streamable-http"
    }
  }
}
```

The exact config keys depend on the orchestrator framework — the important part is it needs the URL above and the `streamable-http` transport.

---

## 5. Troubleshooting

- **Server hangs on startup / first call is very slow**: expected on a cold start if `auto_build=True` and no `outputs/openrouter_graph_db.pkl` exists yet — it's building the graph from `procurement_data/`. Check `healthcheck()` — `ready: false` plus a `FileNotFoundError` means `datas/law_to_crime.json` or `datas/cases_with_feature.json` is missing (run `scripts/prepare_thai_corpus.py` first).
- **`ready: false` with an API-key-shaped error**: `OPENROUTER_API_KEY` (or equivalent) isn't set in `.env` / the container's environment.
- **GPU reranker not used / falls back to CPU**: `reranker_device=cuda:0` in the env file requires a GPU visible to the container — uncomment the `deploy.resources` block in `docker-compose.yml` and ensure the NVIDIA Container Toolkit is installed on the host; otherwise set `reranker_device=cpu`.
- **`docker compose up` can't find `.env`**: it's gitignored and not created automatically — `cp env.example .env` first (see §2).
- **Docker Desktop isn't running**: `docker compose build`/`up` need the Docker daemon started first — open Docker Desktop (or start the `docker` service) before running these commands. Until then, test locally with `python mcp_server.py` instead (see §3, "run without Docker").
