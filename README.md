# LegalGraphRAG: Pure Agentic RAG for Thai Government Procurement Law

> **State-of-the-Art Multi-Agent Legal Reasoning & Statutory Retrieval for Thai Public Procurement Laws.**  
> Powered by **Pure Agentic RAG with LangGraph**, combining **Multi-Aspect Hybrid Retrieval (Dense Vector + Thai BM25 + GPU Cross-Encoder Reranking)**, **Knowledge Graph Traversal (Neo4j)**, and **Dual-Protocol Serving (FastAPI REST + FastMCP)**.

---

## 🚀 Key Highlights & Capabilities

- ✅ **Pure LangGraph Agentic Workflow**:
  1. **Issue Decomposer (`core/agent/decomposer.py`)**: Decomposes complex, multi-faceted inquiries into atomic sub-questions (`Q1`, `Q2`), preventing dominant topics from masking secondary issues during search.
  2. **Multi-Aspect Hybrid Retrieval (`core/graph_construct/feature_graph.py`)**: Concurrently queries Dense Vector Embeddings (BGE-M3), Tokenized Thai BM25 (PyThaiNLP), and Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`) with Reciprocal Rank Fusion.
  3. **Knowledge Graph Traversal (`core/database/neo4j_repository.py`)**: Traverses statutory hierarchies, cross-citations (`CITES_CLAUSE`), sequential sections (`ADJACENT_SECTION`), and FAQ precedents (`RELATES_TO_LAW`).
  4. **Iterative Self-Reflection & Query Refiner (`core/agent/refiner.py`)**: Automatically detects retrieval gaps across sub-issues and triggers sharpened follow-up queries before synthesizing the final answer.
  5. **Grounding Guardrail (`core/agent/guardrail.py`)**: Extracts verbatim statutory quotes (`decisive_quotes`) and validates conclusions against hallucination.
- ✅ **Super-Orchestrator Interface (`/api/v1/qa`)**:
  - Designed as 1 of ~10 specialized AI feature agents in a larger enterprise system.
  - Takes `query` + `org_id`; returns `answer`, `citations` (law, verbatim quote, source file and page), `unresolved_issues` for delegation and a `grounded` flag.
- ✅ **Dual-Protocol Serving (`api/` + `server.py`)**:
  - **FastAPI REST API**: Interactive Swagger UI at `http://localhost:8000/docs`
  - **FastMCP Protocol**: Streamable HTTP on `/mcp`
- ✅ **Validated Benchmark Metrics (40 Held-Out Thai QA Cases)**:
  - **Strict Hit Rate:** **97.5% (39/40)**
  - **Section Recall@k (k=20):** **88.33%**
  - **Faithfulness / Grounding:** **91.89% – 94.44%**
  - **Answer Relevancy:** **95.95% – 98.61%**
  - **Hallucination Rate:** **0.0%**

---

## 🧩 Project Structure

```text
ProcurementQA_Agent/
├── api/                            # Production Backend Serving Layer (FastAPI & FastMCP)
│   ├── app.py                      # FastAPI App, Lifespan Pre-warming, CORS, OpenAPI Docs
│   ├── schemas.py                  # Pydantic Schemas (QARequest/QAResponse orchestrator contract)
│   ├── dependencies.py             # Singleton ProcurementService & Multi-Tenant Resolvers
│   ├── routes/
│   │   ├── qa.py                   # POST /api/v1/qa (Agentic Inquiry for Super-Orchestrator)
│   │   ├── search.py               # POST /api/v1/search (Statutory Hybrid Search)
│   │   └── health.py               # GET /healthz (Liveness) & GET /ready (Readiness)
│   └── mcp/                        # FastMCP Tools & Prompts Adapter (Streamable HTTP on /mcp)
│
├── core/                           # Pure Domain Engine
│   ├── agent/                      # LangGraph Pure Agentic RAG Workflow
│   │   ├── workflow.py             # Compiled LangGraph Workflow & State Management
│   │   ├── state.py                # Typed Dict State Schema (AgenticRAGState)
│   │   ├── decomposer.py           # Sub-issue Decomposition Agent
│   │   ├── refiner.py              # Query Refiner & Targeted Search Agent
│   │   ├── synthesizer.py          # Legal Synthesizer & Adjudicator
│   │   └── guardrail.py            # Grounding Guardrail & Decisive Quotes Extraction
│   ├── database/                   # Tri-Store Multi-Tenant Engine (PostgreSQL, Qdrant, Neo4j)
│   ├── judge/                      # LLM-as-a-Judge & Legal Synthesizer Implementation
│   ├── models/                     # Embedding & Cross-Encoder Reranker Singletons
│   ├── preprocess/                 # Legal Document Chunking & Text Splitters
│   ├── prompt/                     # Centralized Prompt Templates
│   ├── service.py                  # Business Service Facade (ProcurementService)
│   └── utils/                      # Agent Trace Logger, Config, Thai Text Normalization
│
├── docs/                           # Documentation & Architecture Specifications
│   ├── MCP.md                      # Dual-Protocol REST & MCP Guide
│   ├── REPORT.md                   # Technical Architecture & Evolution Report
│   ├── MULTI_TENANCY_CHECKLIST.md  # Stakeholder Interview & Decision Matrix
│   └── NEO4J_GUIDE.md              # Neo4j Graph DB Setup & Cypher Reference
│
├── evaluation/                     # Offline Benchmark & Evaluation Suite (RAG Triad)
├── scripts/                        # Ingestion & Database Indexing Scripts
├── tests/                          # Integration & Unit Tests
├── run.py                          # Offline Evaluation CLI & Experiments
├── server.py                       # Production ASGI Server Entrypoint
├── Dockerfile                      # Production Containerfile
└── docker-compose.yml              # Tri-Store Container Stack
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/PaweekornS/ProcurementQA_Agent.git
cd ProcurementQA_Agent
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `env.example` to `.env` and configure your API keys:
```bash
cp env.example .env
```
Ensure your LLM API key (e.g. `OPENROUTER_API_KEY`) and Embedding API key (`TOKENMIND_API_KEY`) are set.

### 3. Launch the Backend Server
```bash
python server.py --host 0.0.0.0 --port 8000
```
- **Interactive Swagger UI:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
- **Readiness Probe:** `curl http://localhost:8000/ready`
- **FastMCP Protocol:** Available at `http://localhost:8000/mcp`

### 4. Run Offline Benchmark Evaluation
To benchmark the Agentic RAG pipeline across the 40 test cases:
```bash
python run.py --rag-mode agentic --limit 40
```

---

## 🔌 Super-Orchestrator API Integration (`POST /api/v1/qa`)

### Request
```json
{
  "query": "หน่วยงานของรัฐจัดซื้อโดยวิธีเฉพาะเจาะจงได้ในวงเงินไม่เกินเท่าใด และต้องขออนุมัติใครบ้าง",
  "org_id": "DGA"
}
```
`org_id` is optional: falls back to the `X-Organization-Id` header, then `DEFAULT_ORG_ID`.
The same contract is returned by the MCP tool `ask_procurement_law(query, org_id)`.

### Response
```json
{
  "status": "COMPLIANT",
  "answer": "หน่วยงานของรัฐสามารถสั่งซื้อหรือสั่งจ้างโดยวิธีเฉพาะเจาะจงได้ภายในวงเงินที่กำหนดตามตำแหน่งผู้สั่งซื้อ ...",
  "conditions": "กรณีที่มีความจำเป็นเร่งด่วน ... ให้ดำเนินการไปก่อนแล้วรีบรายงานขอความเห็นชอบต่อหัวหน้าหน่วยงานของรัฐ",
  "citations": [
    {
      "law": "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 มาตรา 56",
      "quote": "การจัดซื้อจัดจ้างพัสดุ ... โดยวิธีเฉพาะเจาะจง ...",
      "filename": "พรบ/พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560.md",
      "page": "19-20/42"
    },
    {
      "law": "ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 ข้อ 86",
      "quote": "การสั่งซื้อหรือสั่งจ้างโดยวิธีเฉพาะเจาะจงครั้งหนึ่ง ให้เป็นอำนาจของผู้ดำรงตำแหน่งและภายในวงเงิน ...",
      "filename": "ระเบียบกระทรวงการคลัง/ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560.md",
      "page": "28/72"
    }
  ],
  "unresolved_issues": [],
  "grounded": true,
  "org_id": "DGA"
}
```

| Field | Meaning |
|---|---|
| `status` | `COMPLIANT`, `PARTIALLY_RESOLVED`, `NO_LAW_FOUND` or `OUT_OF_LEGAL_SCOPE` |
| `answer` | Direct legal answer covering every resolved sub-question |
| `conditions` | Exceptions, thresholds or prerequisites qualifying the answer; `null` if none |
| `citations[]` | Laws relied on. `quote` is verbatim statutory text. `filename` is the OCR document path under `datas/typhoon_ocr/` and `page` its page range (`start-end/total`); both are `null` when the cited law is not in the corpus (e.g. a repealed regulation) or its page could not be recovered |
| `unresolved_issues[]` | Only sub-questions that were **not** answered (`NO_LAW_FOUND` / `OUT_OF_LEGAL_SCOPE`), with `missing_aspect`, so the orchestrator can delegate them to another agent |
| `grounded` | `true` when every cited section appears in the retrieved evidence (guardrail); `false` means treat the answer with caution; `null` if not evaluated |
| `org_id` | Tenant the answer was scoped to |

---

## 🏢 Multi-Tenancy & Tri-Store Architecture

For production environments requiring database isolation:
- **PostgreSQL**: Row-Level Security (RLS) scoping queries by `org_id`
- **Qdrant**: Metadata payload filtering ensuring vector tenant isolation
- **Neo4j**: Property-level isolation with B-Tree indexes on `org_id`

See [docs/MULTI_TENANCY_CHECKLIST.md](docs/MULTI_TENANCY_CHECKLIST.md) for detailed stakeholder requirement questions and architectural decision matrix.

### Run the full stack locally (Docker) with ProcurementOCR

```bash
# QA Agent: Postgres + Qdrant + Neo4j + API on :8000 (first run seeds the corpus via the migrate container)
git clone https://github.com/PaweekornS/ProcurementQA_Agent.git && cd ProcurementQA_Agent
cp env.example .env        # set TOKENMIND_API_KEY, OPPER_API_KEY (or RERANKER_ENABLED=false), LLM key
mkdir -p outputs logs && sudo chown -R 10001:10001 outputs logs   # Linux only; not needed on Docker Desktop
docker compose up -d --build
docker logs -f procurement-migrate     # wait for exit 0

# OCR on :8001; finished jobs are pushed to the QA Agent automatically
git clone https://github.com/PaweekornS/ProcurementOCR.git && cd ProcurementOCR
cp env.example .env        # set TYPHOON_API_KEY
docker compose up -d --build

# End-to-end regression (needs both stacks running)
bash ../ProcurementQA_Agent/tests/regression_deploy.sh
```

**Tenants:** the `X-Organization-Id` header wins over `org_id` in the body, then `DEFAULT_ORG_ID`.
`/api/v1/documents` always requires a tenant. Tenant documents (`POST /api/v1/documents`, or pushed by OCR)
are visible only to their owner; they are indexed in Postgres (RLS), Qdrant and Neo4j, where chunks link to
the statutes they cite (`มาตรา N` → the Act, `ระเบียบฯ ข้อ N` → the MoF regulation).
The migrate container seeds the statute corpus as tenant `DGA` (`DEFAULT_ORG_ID`; override with
`python scripts/migrate_to_tri_store.py --org-id <ORG>`), so other tenants see only their own documents.

---

## 📄 License
Internal proprietary research & development for Thai Government Procurement Automation.
