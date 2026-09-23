# Neo4j Knowledge Graph Visualization & Exploration Guide

This guide explains how to spin up a **Neo4j** graph database, migrate your Thai Government Procurement Knowledge Graph from the in-memory pickle (`outputs/openrouter_graph_db.pkl`), and visualize it using **Neo4j Browser** and **Neo4j Bloom**.

---

## 🏗️ 1. Architecture Overview

Your knowledge graph contains **3,471 nodes** and **12,841 relationships**:

### Node Labels
- `:Laws` (3,192 nodes): Statutory articles, sections, and clauses from Thai procurement law and regulations.
- `:Crimes` (129 nodes): Legal breach concepts, procedural violations, and procurement issue classifications.
- `:Cluster` (121 nodes): High-level thematic community clusters derived from graph modularity/Louvain.
- `:Cases` (29 nodes): Consultation/FAQ precedent cases from Comptroller General's Dept. (กรมบัญชีกลาง).

### Relationship Types
- `[:RELATED_CRIME]`: Links statutory clauses (`:Laws`) to specific legal violation issues (`:Crimes`).
- `[:CONTAINS_LAW]` / `[:BELONGS_TO]`: Connects thematic community clusters (`:Cluster`) and statutes (`:Laws`).
- `[:RELATES_TO_LAW]`: Links precedent consultation cases (`:Cases`) to governing statutes (`:Laws`).
- `[:SIMILAR_TO]`: Semantic similarity edges between related cases.
- `[:ADJACENT_SECTION]` / `[:CITES_CLAUSE]`: Intra-document statutory structure and cross-references.

---

## 🚀 2. Quick Start

### Step 1: Install Neo4j Python Driver
If running locally on your host environment:
```bash
pip install neo4j>=5.15.0
```

### Step 2: Start Neo4j Container
A pre-configured Neo4j service with APOC plugins is included in `docker-compose.yml`:
```bash
docker compose up -d neo4j
```

Wait ~15 seconds for the container to initialize.
Verify that Neo4j is running:
```bash
docker compose ps neo4j
```

### Step 3: Run the Migration Utility
To export the knowledge graph into Neo4j:
```bash
python scripts/export_to_neo4j.py
```

Optional parameters:
```bash
# Test sanitization without connecting to Neo4j
python scripts/export_to_neo4j.py --dry-run

# Clear existing graph database before importing
python scripts/export_to_neo4j.py --clear

# Specify custom connection credentials
python scripts/export_to_neo4j.py --uri bolt://localhost:7687 --user neo4j --password procurement_secret123
```

---

## 🎨 3. Visualizing in Neo4j Browser

Open your browser at:
👉 **[http://localhost:7474](http://localhost:7474)**

### Login Credentials
- **Connect URL**: `bolt://localhost:7687`
- **Username**: `neo4j`
- **Password**: `procurement_secret123` *(defined in `.env` / `docker-compose.yml`)*

### Recommended Visual Styling (Grass Style)
Click on the node labels in the top-left sidebar to configure colors and captions:
- **`Laws`**: Emerald Green (`#2ecc71`), Caption: `entry` or `id`
- **`Crimes`**: Coral Red (`#e74c3c`), Caption: `description`
- **`Cases`**: Dodger Blue (`#3498db`), Caption: `caseId` or `id`
- **`Cluster`**: Purple (`#9b59b6`), Caption: `id`

---

## 🔍 4. Curated Cypher Queries for Legal Analysis

### 1. High-Level Schema Visualization
Displays the meta-graph showing how all node labels and relationships connect:
```cypher
CALL db.schema.visualization()
```

### 2. Landmark Statutes (Highest Degree Centrality)
Find the most referenced and central procurement laws in the knowledge base:
```cypher
MATCH (l:Laws)-[r]-()
RETURN l.id AS law_id, l.entry AS statute_name, count(r) AS connections
ORDER BY connections DESC
LIMIT 15
```

### 3. Explore a Specific Statute and Its Ecosystem
Visualize a specific law (e.g., procurement methods or committee responsibilities) and all related cases and violation topics:
```cypher
MATCH (l:Laws)
WHERE l.entry CONTAINS "มาตรา 60" OR l.id CONTAINS "60"
MATCH path = (l)-[r]-(target)
RETURN path
LIMIT 50
```

### 4. Consultations Linked to Statutes (`:Cases` -> `:Laws`)
Inspect how FAQ consultation cases connect to governing statutory clauses:
```cypher
MATCH path = (c:Cases)-[r:RELATES_TO_LAW]->(l:Laws)
RETURN path
LIMIT 30
```

### 5. Legal Crime / Issue Neighborhood
Inspect which statutes govern a specific procurement issue (e.g., collusive bidding or conflict of interest):
```cypher
MATCH path = (cr:Crimes)<-[r:RELATED_CRIME]-(l:Laws)
RETURN path
LIMIT 40
```

### 6. Thematic Clusters (`:Cluster` -> `:Laws`)
Explore community clusters discovered by the graph modularity algorithm:
```cypher
MATCH path = (cl:Cluster)-[r:CONTAINS_LAW]->(l:Laws)
RETURN path
LIMIT 50
```

### 7. Intra-Document Citation Hierarchy
Visualize cross-references between sections (such as when Section A cites Section B):
```cypher
MATCH path = (l1:Laws)-[r:CITES_CLAUSE|ADJACENT_SECTION]->(l2:Laws)
RETURN path
LIMIT 50
```

---

## 🌸 5. Exploring with Neo4j Bloom (Business & Legal Stakeholders)

If using Neo4j Desktop or Enterprise, **Neo4j Bloom** offers a codeless, natural language visual graph explorer:
1. Open Neo4j Bloom.
2. Create a perspective for the `procurement` database.
3. Search in natural language: e.g. `"Laws connected to Crimes"` or `"Cases relating to มาตรา"`.
4. Use Bloom's path-finding tool to discover multi-hop paths between distinct procurement concepts.
