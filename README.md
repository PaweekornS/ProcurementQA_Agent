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
  - Returns `issues_breakdown` with atomic issue statuses (`RESOLVED`, `OUT_OF_LEGAL_SCOPE`, `NO_LAW_FOUND`) and `missing_aspect` for upstream routing.
- ✅ **Dual-Protocol Serving (`api/` + `server.py`)**:
  - **FastAPI REST API**: Interactive Swagger UI at `http://localhost:8000/docs`
  - **FastMCP Protocol**: Mounted on `/mcp` (Streamable-HTTP) and `/sse` (Server-Sent Events)
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
│   ├── schemas.py                  # Pydantic Schemas (Q&A, IssuesBreakdown, DecisiveQuotes)
│   ├── dependencies.py             # Singleton ProcurementService & Multi-Tenant Resolvers
│   ├── routes/
│   │   ├── qa.py                   # POST /api/v1/qa (Agentic Inquiry for Super-Orchestrator)
│   │   ├── search.py               # POST /api/v1/search (Statutory Hybrid Search)
│   │   ├── verify.py               # POST /api/v1/verify (Compliance Threshold Check)
│   │   └── health.py               # GET /healthz (Liveness) & GET /ready (Readiness)
│   └── mcp/                        # FastMCP Tools & Prompts Adapter (Mounted on /mcp, /sse)
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

Submit a query via cURL:
```bash
curl -X POST http://localhost:8000/api/v1/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงในวงเงินไม่เกินเท่าใด และต้องขออนุมัติใครบ้าง",
    "mode": "fast",
    "org_id": "DGA"
  }'
```

**Response Format:**
```json
{
  "status": "COMPLIANT",
  "mode": "fast",
  "direct_answer": "หน่วยงานของรัฐสามารถจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงได้ในวงเงินไม่เกิน 500,000 บาท และต้องขอความเห็นชอบรายงานขอซื้อขอจ้างจากหัวหน้าหน่วยงานของรัฐก่อนเริ่มดำเนินการ",
  "applicable_laws": [
    "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 มาตรา 56 (2) (ข)",
    "ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 ข้อ 25"
  ],
  "decisive_quotes": [
    {
      "filename": "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560.md",
      "page": "มาตรา 56 (2) (ข)",
      "law": "พ.ร.บ. จัดซื้อจัดจ้างฯ มาตรา 56 (2) (ข)",
      "quote": "การจัดซื้อจัดจ้างพัสดุที่มีการผลิต จำหน่าย... หรือวงเงินไม่เกินที่กำหนดในกฎกระทรวง"
    }
  ],
  "issues_breakdown": [
    {
      "issue_id": "Q1",
      "topic": "วงเงินวิธีเฉพาะเจาะจง",
      "status": "RESOLVED",
      "answer": "วงเงินไม่เกิน 500,000 บาท ตามที่กำหนดในกฎกระทรวง",
      "missing_aspect": null
    },
    {
      "issue_id": "Q2",
      "topic": "ผู้อนุมัติรายงานขอซื้อขอจ้าง",
      "status": "RESOLVED",
      "answer": "ต้องได้รับความเห็นชอบจากหัวหน้าหน่วยงานของรัฐก่อนดำเนินการ",
      "missing_aspect": null
    }
  ],
  "exceptions_or_conditions": "ห้ามมิให้แบ่งซื้อหรือแบ่งจ้างพัสดุเพื่อลดวงเงินให้เข้าเกณฑ์วิธีเฉพาะเจาะจง",
  "organization_id": "DGA"
}
```

---

## 🏢 Multi-Tenancy & Tri-Store Architecture

For production environments requiring database isolation:
- **PostgreSQL**: Row-Level Security (RLS) scoping queries by `org_id`
- **Qdrant**: Metadata payload filtering ensuring vector tenant isolation
- **Neo4j**: Property-level isolation with B-Tree indexes on `org_id`

See [docs/MULTI_TENANCY_CHECKLIST.md](docs/MULTI_TENANCY_CHECKLIST.md) for detailed stakeholder requirement questions and architectural decision matrix.

---

## 📄 License
Internal proprietary research & development for Thai Government Procurement Automation.
