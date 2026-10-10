# Database Architecture & Multi-Tenancy Specification

## Thai Government Procurement ProcurementQAPipeline System

---

## 📌 1. Executive Summary & Design Principles

The **Procurement ProcurementQAPipeline** backend transitions from a monolithic in-memory graph representation (`.pkl`) to a production-grade **Decoupled Tri-Store Architecture** (PostgreSQL 16, Qdrant 1.13, and Neo4j 5.26 Community).

This design delivers:

- **Strict Multi-Tenant Isolation:** Zero cross-tenant data leakage ($0.00\%$ leak rate across 10 distinct government agencies).
- **Sub-35ms Retrieval Latency:** Replaces naive $O(N)$ linear scans with indexed vector graphs and topological constraints.
- **Ultra-Lightweight API Footprint:** Decouples heavy vector matrices and text payloads from the Python runtime, keeping the FastAPI process at **~87 MB RAM**.
- **Zero Software Licensing Cost ($0):** Utilizes 100% Free and Open-Source Software (FOSS) and Community Editions without requiring costly Enterprise licenses.

```mermaid
flowchart TD
    Client(["Client / Agentic RAG / MCP Service"]) --> API["FastAPI / MCP Server<br/>(Memory: ~87 MB RSS)"]
  
    subgraph TriStore ["Tri-Store Database Tier (100% FOSS / Budget: 0 USD)"]
        PG[("PostgreSQL 16 (SSOT)<br/>• Authoritative Statutory Text<br/>• Composite B-Tree & GIN Trigram<br/>• Row-Level Security (RLS)")]
        QD[("Qdrant 1.13 (Vector Engine)<br/>• 1024-dim BGE-M3 Dense Vectors<br/>• Native PyThaiNLP Sparse BM25<br/>• Payload Inverted Index on org_id")]
        NEO[("Neo4j 5.26 (Knowledge Graph)<br/>• Legal Hierarchies & Citations<br/>• Sequential Clause Adjacency<br/>• Range Property Index on org_id")]
    end
  
    API -->|1. Hybrid Vector + Lexical Search with Tenant Pre-Filter| QD
    API -->|2. Authoritative Full-Text Hydration by IDs| PG
    API -->|3. Topological Legal Traversal & Precedent Retrieval| NEO
```

---

## ⚠️ 2. Why Move Beyond Monolithic In-Memory Graphs?

The previous implementation bundled the entire knowledge base into a single Python pickle file (`outputs/openrouter_section_graph_db.pkl`, ~87 MB on disk). While suitable for single-user prototyping, it exhibited critical bottlenecks in multi-tenant environments:

| Engineering Constraint              | Monolithic In-Memory (`.pkl`)                                                                                                                                                                                                                                                                           | Decoupled Tri-Store Architecture                                                                                                                                          |
| :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Memory Allocation**         | Deserializes all text strings, NumPy vector arrays, and NetworkX node objects directly into the Python Heap. Scaling to 10 tenants caused RAM to balloon to**1,451.8 MB (+1.05 GB)**.                                                                                                               | Uses**Memory-Mapped I/O (mmap)** in dedicated engines. Heavy text is paged to disk in PostgreSQL; vectors reside in Qdrant. Python process uses **87.04 MB**. |
| **Horizontal Worker Scaling** | Running multi-worker Uvicorn (`--workers 4`) duplicates the in-memory graph per worker process ($1.45\text{ GB} \times 4 \approx 5.8\text{ GB}$), causing Out-of-Memory (`OOM Kill`) crashes.                                                                                                       | API workers are**stateless**. Multiple workers share persistent socket connections to central databases without replicating memory.                                 |
| **Search Complexity**         | **$O(N \cdot D)$ Linear Scan:** Iterates over every vector row sequentially in Python CPU loops. Latency was **~77.3 ms** at 26,000 clauses. | **$O(\log N)$ Indexed Graph:** Qdrant HNSW and inverted payload indexes execute vector searches in **~30.2 ms** (2.5x faster). |                                                                                                                                                                           |
| **Crash Resilience**          | An uncaught exception or heavy PDF OCR ingestion in FastAPI crashes the database state.                                                                                                                                                                                                                   | **Full Process Isolation:** Database daemons run independently in Docker containers with automatic restart policies.                                                |

---

## 🏗️ 3. Tri-Store Component Deep Dive

### 3.1 PostgreSQL 16 (Single Source of Truth & Relational Metadata)

PostgreSQL serves as the immutable system of record for all legal statutes, executive decrees, and precedent consultations.

- **Key Tables:**
  - `documents`: Master statutory acts, ministerial regulations, and administrative orders.
  - `chunks`: Section/clause-level legal articles with `org_id` scoping and chapter hierarchies.
  - `faq_cases`: Historical procurement consultation rulings from the Comptroller General's Department (CGD).
  - `query_audit_logs`: Immutable compliance audit trail logging timestamp, client IP, querying `org_id`, and retrieved clause IDs.
- **Relational Optimization & Indexing:**
  - `idx_chunk_org_id`: B-Tree index on `org_id` to guarantee tenant isolation during hydration.
  - `idx_chunk_doc_sec`: Composite index on `(doc_id, section_num)` providing exact statutory lookups (e.g., *Section 56*) in **$< 5$ ms**.
  - `idx_chunk_content_trgm`: GIN Trigram index (`gin_trgm_ops`) on `content` for typo-tolerant lexical searching.
  - **Row-Level Security (RLS):** Enforces database-level isolation policies:
    ```sql
    CREATE POLICY tenant_isolation_policy ON chunks
    FOR SELECT USING (org_id IN ('PUBLIC', current_setting('app.current_org_id', true)));
    ```

---

### 3.2 Qdrant 1.13 (Hybrid Dense & Sparse Vector Engine)

Qdrant coordinates dense semantic representations and native lexical BM25 token weights.

- **Dual Vector Representations:**
  1. **Dense Vector (`dense_bge_m3`):** 1024-dimensional normalized embeddings computed via BAAI/bge-m3. HNSW graph configured with $M=16$ and $ef\_construct=128$, distance metric `Cosine`.
  2. **Native Sparse Vector (`sparse_bm25`):** Generated via deterministic PyThaiNLP dictionary tokenization (`newmm` engine) with sub-linear term frequency weighting ($1 + \ln(\text{tf})$).
- **Server-Side Fusion (RRF):**
  Executes server-side Reciprocal Rank Fusion directly inside Qdrant:
  $$
  \text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + r_m(d)}
  $$

  eliminating client-side network roundtrips for intermediate candidate sets.
- **Memory-Mapped Vectors (`on_disk=True`):**
  Cold vectors remain paged out on disk/SSD, while active HNSW edges are cached in RAM.

---

### 3.3 Neo4j 5.26 Community Edition (Legal Knowledge Graph)

Neo4j models topological inter-statute dependencies and judicial precedents.

- **Node Classifications:**
  - `:Document`: Master legal enactments (e.g., พ.ร.บ. จัดซื้อจัดจ้างฯ พ.ศ. 2560).
  - `:Chunk`: Granular statutory sections and sub-clauses.
  - `:FAQCase`: Precedent advisory cases.
  - `:LegalTopic`: Procurement domain ontology clusters.
- **Directed Semantic Relationships:**
  - `[:CONTAINS]`: Document-to-clause structural hierarchy.
  - `[:ADJACENT_SECTION]`: Sequential section flow (`PREV_SECTION` / `NEXT_SECTION`) for context expansion.
  - `[:CITES_CLAUSE]`: Cross-statutory citations (e.g., internal agency rules citing Public Act Section 56).
  - `[:RELATES_TO_LAW]`: Precedent rulings linked to governed sections.
- **Enterprise-Free Multi-Tenancy Strategy ($0 Cost):**
  Instead of Neo4j Enterprise's proprietary multi-database feature, we implement **Logical Graph Partitioning** via node property tagging (`c.org_id`) coupled with Neo4j Range Property Indexes.

---

## 🛡️ 4. Multi-Tenancy & Data Isolation Architecture

Data security is designed to guarantee **Zero Cross-Tenant Leakage** across multi-agency deployments.

### 4.1 Data Visibility Matrix

| Asset Type                                                                   | Scoping Tag (`org_id`) | Own Agency (`Tenant A`) | Other Agencies (`Tenant B`) |    Public / External    |
| :--------------------------------------------------------------------------- | :----------------------: | :-----------------------: | :---------------------------: | :----------------------: |
| **National Statutes (พ.ร.บ. แม่บท)**                           |        `PUBLIC`        |        ✅ Visible        |          ✅ Visible          |        ✅ Visible        |
| **MOF Ministerial Rules (ระเบียบกระทรวงการคลัง)** |        `PUBLIC`        |        ✅ Visible        |          ✅ Visible          |        ✅ Visible        |
| **DGA Internal Cloud Regulations**                                     |      `ORG_01_DGA`      |        ✅ Visible        |   🛑**Blocked (0%)**   | 🛑**Blocked (0%)** |
| **MOPH Emergency Medical Rules**                                       |     `ORG_05_MOPH`     | 🛑**Blocked (0%)** |   🛑**Blocked (0%)**   | 🛑**Blocked (0%)** |

### 4.2 Layered Enforcement Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (Tenant: ORG_05_MOPH)
    participant API as FastAPI Gateway
    participant QD as Qdrant (Payload Index)
    participant PG as PostgreSQL (SSOT)
    participant NEO as Neo4j (Property Guard)

    User->>API: POST /api/v1/search (Header: X-Tenant-ID: ORG_05_MOPH)
    API->>API: Validate & Sanitize Tenant Scope (Allowed: ['PUBLIC', 'ORG_05_MOPH'])
  
    API->>QD: Vector Hybrid Search with Pre-Filter [org_id IN ('PUBLIC', 'ORG_05_MOPH')]
    Note over QD: Inverted Index skips all other tenant vertices<br/>Computes Cosine only on valid points
    QD-->>API: Return Top-K Candidate IDs & RRF Scores
  
    API->>PG: Batch Hydration SELECT WHERE chunk_id IN (...) AND org_id IN (...)
    PG-->>API: Authoritative Legal Texts
  
    API->>NEO: Cypher Traversal WHERE org_id IN [PUBLIC, ORG_05_MOPH]
    Note over NEO: Traversals terminate immediately at boundaries of unauthorized nodes
    NEO-->>API: Neighboring Context & Citations
  
    API-->>User: Synthesized Multi-Tenant Legal Context (0% Leakage)
```

---

## ⚡ 5. Indexing & Optimization (Beyond Linear Search)

### 5.1 Qdrant Payload Inverted Indexing

- **Mechanism:** Qdrant maintains an inverted index on payload keywords.
  ```python
  client.create_payload_index(
      collection_name="procurement_chunks",
      field_name="org_id",
      field_schema=models.PayloadSchemaType.KEYWORD
  )
  ```
- **Performance Impact:** During search, Qdrant applies **pre-filtering** before HNSW graph traversal. Vectors belonging to other organizations are pruned at the index layer. Latency dropped from **77.3 ms** (in flat linear scan) to **30.18 ms** (a **2.5x speedup**).

### 5.2 Neo4j Range Property Indexing

- **Mechanism:** B-Tree/Range indexes on node labels and properties:
  ```cypher
  CREATE INDEX chunk_org_id_lookup IF NOT EXISTS FOR (c:Chunk) ON (c.org_id);
  CREATE INDEX chunk_section_lookup IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id, c.section_num);
  ```
- **Performance Impact:** Node lookup by `chunk_id` or `(doc_id, section_num)` operates in $O(1)$ to $O(\log N)$ time ($< 0.05\text{ ms}$). Graph traversals enforce boundary checks without full database scans, running 1-to-2 hop traversals in **$< 46\text{ ms}$**.

### 5.3 Memory Decoupling via Memory-Mapped I/O (mmap)

- By offloading string storage to PostgreSQL TOAST pages and float matrices to Qdrant memory-mapped segment files, the main Python runtime avoids memory bloating.
- **RAM Footprint:** FastAPI maintains an RSS memory of **87.04 MB**, allowing horizontal scaling across 8 to 16 workers on modest hardware without memory pressure.

---

## 📊 6. Empirical 10-Tenant Benchmark Audit

Rigorous multi-tenant benchmark executed across **10 distinct Thai Government Agencies and State Enterprises** (DGA, depa, ETDA, BMA, MOPH, MOF, CHULA, EGAT, PTT, AOT) simulating public law and confidential internal regulation queries:

| Metric                                 |  Monolithic Graph (`.pkl`)  |   Tri-Store Architecture (Postgres + Qdrant + Neo4j)   |        Evaluation Status        |
| :------------------------------------- | :---------------------------: | :----------------------------------------------------: | :------------------------------: |
| **Cross-Tenant Data Leak Rate**  |   0.00% (Python loop check)   |     **0.00% (Database Engine Enforcement)**     | **PASSED (Zero Leakage)** |
| **Dense/Hybrid Vector Latency**  |           77.30 ms           |                   **30.18 ms**                   |      **2.5x Faster**      |
| **Exact Section Lookup Latency** |      0.004 ms (RAM dict)      |         **2.40 ms (Relational B-Tree)**         |    **Production Grade**    |
| **Graph Traversal Latency**      | 0.001 ms (In-Memory NetworkX) |         **45.99 ms (Distributed Neo4j)**         | **High Topology Fidelity** |
| **Python Process RSS Memory**    |   1,451.80 MB (High Bloat)   |       **87.04 MB (Extremely Lightweight)**       | **16.6x Memory Reduction** |
| **Docker Containers Memory**     |              N/A              | **Neo4j: 636 MB \| Qdrant: 412 MB \| PG: 42 MB** |    **Stable & Bounded**    |
| **Worker Scalability**           |   Max 1 worker (due to RAM)   |      **Multi-Worker Ready (Stateless API)**      |        **Scalable**        |
| **Enterprise Software Cost**     |              $0              |     **$0 (100% Free / Community Editions)**     |   **Zero Budget Impact**   |

Detailed per-query audit logs and retrieved chunks are cataloged in [outputs/tri_store_10_tenants_retrieved_chunks.json](file:///c:/Users/ACER/Downloads/ProcurementQA_Agent/outputs/tri_store_10_tenants_retrieved_chunks.json) and [outputs/TRI_STORE_10_TENANTS_REPORT.md](file:///c:/Users/ACER/Downloads/ProcurementQA_Agent/outputs/TRI_STORE_10_TENANTS_REPORT.md).

---

## 🛠️ 7. Operational Deployment & Configuration

### 7.1 Docker Compose Services

To launch the Tri-Store database daemons in the background:

```bash
docker compose up -d postgres qdrant neo4j
```

Verify service health:

```bash
docker compose ps
```

### 7.2 Configuration (.env)

Enable the Tri-Store backend by updating your `.env` configuration:

```env
# ==============================================================================
# Tri-Store Production Database Activation
# ==============================================================================
USE_TRI_STORE=true
DEFAULT_ORG_ID="DGA"
ENABLE_TENANT_ISOLATION=true

# PostgreSQL (SSOT)
POSTGRES_HOST="localhost"
POSTGRES_PORT=5435
POSTGRES_DB="procurement_rag"
POSTGRES_USER="procurement_user"
POSTGRES_PASSWORD="procurement_secret123"

# Qdrant (Vector Engine)
QDRANT_HOST="localhost"
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_DENSE_DIM=1024

# Neo4j (Knowledge Graph)
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="procurement_secret123"
```

### 7.3 Data Migration Utility

To ingest or re-synchronize the statutory corpus into Tri-Store:

```bash
python scripts/migrate_to_tri_store.py   # chunks data_ocr/ into outputs/corpus/, then ingests
```

---

*Document Version: 2.0 (Production Multi-Tenant Specification)*
*Maintained by: Antigravity AI Engineering Team*
