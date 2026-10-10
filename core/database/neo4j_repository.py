# -*- coding: utf-8 -*-
"""
core/database/neo4j_repository.py

Neo4j Graph Database Repository for ProcurementQA Agent.
Manages topological relationships: document hierarchies, sequential clause adjacency
(ADJACENT_SECTION), cross-statutory citations (CITES_CLAUSE), and precedent links (RELATES_TO_LAW).
"""

import os
import logging
from typing import Tuple, List, Dict, Any, Optional

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv(override=False)
logger = logging.getLogger("neo4j_repository")


class Neo4jRepository:
    """Manages Neo4j knowledge graph schema, batched UNWIND ingestion, and Cypher traversals."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver is not installed. Run 'pip install neo4j>=5.15.0'.")

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "procurement_secret123")
        self.driver: Optional[Driver] = None

    def connect(self):
        """Establish connection pool to Neo4j."""
        if self.driver is None:
            driver_kwargs = {
                "auth": (self.user, self.password),
                "max_connection_lifetime": 3600,
                "max_connection_pool_size": 50,
                "connection_acquisition_timeout": 30.0,
            }
            try:
                from neo4j import NotificationMinimumSeverity
                driver_kwargs["notifications_min_severity"] = NotificationMinimumSeverity.OFF
            except (ImportError, AttributeError):
                pass

            self.driver = GraphDatabase.driver(
                self.uri,
                **driver_kwargs
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j instance.")

    def close(self):
        """Close driver connections."""
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def init_schema(self):
        """Idempotently create uniqueness constraints, indexes, and register relationship types."""
        self.connect()
        constraints = [
            "CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT unique_case_id IF NOT EXISTS FOR (f:FAQCase) REQUIRE f.case_id IS UNIQUE",
            "CREATE CONSTRAINT unique_topic_name IF NOT EXISTS FOR (t:LegalTopic) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX chunk_section_lookup IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id, c.section_num)",
            "CREATE INDEX chunk_number_lookup IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id, c.clause_num)",
            "CREATE INDEX doc_org_id_lookup IF NOT EXISTS FOR (d:Document) ON (d.org_id)",
            "CREATE INDEX chunk_org_id_lookup IF NOT EXISTS FOR (c:Chunk) ON (c.org_id)",
            "CREATE INDEX case_org_id_lookup IF NOT EXISTS FOR (f:FAQCase) ON (f.org_id)",
            "CREATE CONSTRAINT unique_tenant_doc_id IF NOT EXISTS FOR (t:TenantDocument) REQUIRE t.doc_id IS UNIQUE",
            "CREATE CONSTRAINT unique_tenant_chunk_id IF NOT EXISTS FOR (t:TenantChunk) REQUIRE t.chunk_id IS UNIQUE",
            "CREATE INDEX tenant_doc_org_id_lookup IF NOT EXISTS FOR (t:TenantDocument) ON (t.org_id)",
            "CREATE INDEX tenant_chunk_org_id_lookup IF NOT EXISTS FOR (t:TenantChunk) ON (t.org_id)",
        ]
        with self.driver.session() as session:
            for stmt in constraints:
                session.run(stmt)

            # Register relationship types to prevent Neo4j 5.x 01N51 schema warnings
            session.run("""
            MERGE (a:_SchemaInit {id: 1})
            MERGE (b:_SchemaInit {id: 2})
            MERGE (a)-[:RELATES_TO_LAW]->(b)
            MERGE (a)-[:REFERENCES_DOCUMENT]->(b)
            MERGE (a)-[:ADJACENT_SECTION]->(b)
            MERGE (a)-[:CITES_CLAUSE]->(b)
            MERGE (a)-[:EMPOWERED_BY]->(b)
            WITH a, b
            MATCH (a)-[r]-(b)
            DELETE r, a, b
            """)
        logger.info("Neo4j constraints and indexes successfully initialized.")

    def sync_documents(self, documents: List[Dict[str, Any]], org_id: str = "PUBLIC"):
        """Batch upsert :Document nodes with tenant org_id."""
        self.connect()
        query = """
        UNWIND $batch AS doc
        MERGE (d:Document {doc_id: doc.doc_id})
        SET d.title = doc.title,
            d.doc_type = doc.doc_type,
            d.year_be = doc.year_be,
            d.source_file = doc.source_file,
            d.org_id = coalesce(doc.org_id, $default_org_id, 'PUBLIC')
        """
        with self.driver.session() as session:
            session.run(query, batch=documents, default_org_id=org_id)

    def sync_chunks(self, chunks: List[Dict[str, Any]], org_id: str = "PUBLIC", batch_size: int = 500):
        """Batch upsert :Chunk nodes and connect them to their parent :Document."""
        self.connect()
        query = """
        UNWIND $batch AS item
        MERGE (c:Chunk {chunk_id: item.chunk_id})
        SET c.doc_id = item.doc_id,
            c.entry = item.entry,
            c.chapter_num = item.chapter_num,
            c.section_num = item.section_num,
            c.clause_num = item.clause_num,
            c.page_start = item.page_start,
            c.page_end = item.page_end,
            c.kind = item.kind,
            c.label = item.label,
            c.org_id = coalesce(item.org_id, $default_org_id, 'PUBLIC')
        WITH c, item
        MATCH (d:Document {doc_id: item.doc_id})
        MERGE (d)-[:CONTAINS]->(c)
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            with self.driver.session() as session:
                session.run(query, batch=batch, default_org_id=org_id)
            logger.info(f"Synced {len(batch)} Chunk nodes to Neo4j.")

    def sync_faq_cases(self, cases: List[Dict[str, Any]], org_id: str = "PUBLIC", batch_size: int = 200):
        """Batch upsert :FAQCase nodes with tenant org_id."""
        self.connect()
        query = """
        UNWIND $batch AS cs
        MERGE (f:FAQCase {case_id: cs.case_id})
        SET f.question_preview = substring(cs.question, 0, 150),
            f.source = cs.source,
            f.org_id = coalesce(cs.org_id, $default_org_id, 'PUBLIC')
        """
        for i in range(0, len(cases), batch_size):
            chunk = cases[i:i + batch_size]
            with self.driver.session() as session:
                session.run(query, batch=chunk, default_org_id=org_id)

    def sync_relationships(self, edges: List[Dict[str, Any]], batch_size: int = 1000):
        """
        Batch creates relationships between nodes:
        Edge dict keys: 'source_id', 'target_id', 'rel_type', 'props'
        """
        self.connect()
        # Group by relation type for optimal parameterized Cypher execution
        rel_groups: Dict[str, List[Dict[str, Any]]] = {}
        for e in edges:
            rel_type = e.get("rel_type", "RELATED_TO")
            rel_groups.setdefault(rel_type, []).append(e)

        for rel_type, edge_list in rel_groups.items():
            query = f"""
            UNWIND $batch AS e
            MATCH (src {{chunk_id: e.source_id}})
            MATCH (tgt {{chunk_id: e.target_id}})
            MERGE (src)-[r:{rel_type}]->(tgt)
            SET r += coalesce(e.props, {{}})
            """
            for i in range(0, len(edge_list), batch_size):
                chunk = edge_list[i:i + batch_size]
                with self.driver.session() as session:
                    session.run(query, batch=chunk)
            logger.info(f"Synced {len(edge_list)} relationships of type ':{rel_type}' to Neo4j.")

    def delete_corpus(self, org_id: str, batch_size: int = 5000) -> int:
        """Detach-delete one tenant's seeded corpus nodes (documents, chunks, FAQ cases) before a
        re-ingest, including nodes under the pre-rename labels (LegalDocument / StatuteClause).
        Tenant-uploaded TenantDocument/TenantChunk nodes are not touched."""
        self.connect()
        query = """
        MATCH (n) WHERE (n:Document OR n:Chunk OR n:FAQCase OR n:LegalDocument OR n:StatuteClause)
          AND n.org_id = $org_id
        WITH n LIMIT $limit
        DETACH DELETE n
        RETURN count(*) AS n
        """
        removed = 0
        with self.driver.session() as session:
            while True:
                n = session.run(query, org_id=org_id, limit=batch_size).single()["n"]
                removed += n
                if n < batch_size:
                    break
        return removed

    def clear_derived_relationships(self, rel_types: List[str], org_id: str = "PUBLIC", batch_size: int = 10000) -> int:
        """
        Delete corpus-derived relationships of the given types between corpus nodes (PUBLIC or the
        corpus tenant `org_id`) so a re-link starts clean. Tenant document edges are left untouched.
        """
        self.connect()
        removed = 0
        for rel_type in rel_types:
            query = f"""
            MATCH (a)-[r:{rel_type}]->(b)
            WHERE NOT a:TenantChunk AND NOT a:TenantDocument
              AND coalesce(a.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
              AND coalesce(b.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
            WITH r LIMIT $limit
            DELETE r
            RETURN count(*) AS n
            """
            with self.driver.session() as session:
                while True:
                    n = session.run(query, limit=batch_size, org_id=org_id).single()["n"]
                    removed += n
                    if n < batch_size:
                        break
        return removed

    def get_graph_meta(self, key: str) -> Any:
        """Read a build-metadata value (e.g. linker_version) stored on the :_GraphMeta node."""
        self.connect()
        with self.driver.session() as session:
            rec = session.run("MATCH (m:_GraphMeta {id: 'corpus'}) RETURN m[$key] AS v", key=key).single()
            return rec["v"] if rec else None

    def set_graph_meta(self, key: str, value: Any):
        """Persist a build-metadata value on the :_GraphMeta node."""
        self.connect()
        with self.driver.session() as session:
            session.run("MERGE (m:_GraphMeta {id: 'corpus'}) SET m[$key] = $value", key=key, value=value)

    def link_case_to_documents(self, links: List[Dict[str, str]]):
        """Links :FAQCase to the :Document it is answered under via :REFERENCES_DOCUMENT."""
        self.connect()
        query = """
        UNWIND $batch AS link
        MATCH (f:FAQCase {case_id: link.case_id})
        MATCH (d:Document {doc_id: link.doc_id})
        MERGE (f)-[:REFERENCES_DOCUMENT]->(d)
        """
        with self.driver.session() as session:
            session.run(query, batch=links)

    def link_case_to_laws(self, links: List[Dict[str, str]]):
        """Links :FAQCase to :Chunk via :RELATES_TO_LAW."""
        self.connect()
        query = """
        UNWIND $batch AS link
        MATCH (f:FAQCase {case_id: link.case_id})
        MATCH (c:Chunk {chunk_id: link.chunk_id})
        MERGE (f)-[:RELATES_TO_LAW]->(c)
        """
        with self.driver.session() as session:
            session.run(query, batch=links)

    # ==========================================
    # Graph Traversal Queries (For CRAG Engine)
    # ==========================================

    def get_adjacent_sections(self, chunk_id: str, direction: str = "both", org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Traverse sequential adjacent sections (ADJACENT_SECTION) with tenant filtering."""
        self.connect()
        if direction == "next":
            rel_pattern = "-[:ADJACENT_SECTION {direction: 'next'}]->"
        elif direction == "prev":
            rel_pattern = "-[:ADJACENT_SECTION {direction: 'prev'}]->"
        else:
            rel_pattern = "-[:ADJACENT_SECTION]-"

        query = f"""
        MATCH (c:Chunk {{chunk_id: $cid}}){rel_pattern}(adj:Chunk)
        WHERE coalesce(adj.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN DISTINCT adj.chunk_id AS chunk_id, adj.entry AS entry, adj.section_num AS section_num
        """
        with self.driver.session() as session:
            result = session.run(query, cid=chunk_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_cited_clauses(self, chunk_id: str, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fetch all clauses cited by, or empowering, this clause (CITES_CLAUSE | EMPOWERED_BY) with tenant filtering."""
        self.connect()
        query = """
        MATCH (c:Chunk {chunk_id: $cid})-[r:CITES_CLAUSE|EMPOWERED_BY]->(cited:Chunk)
        WHERE coalesce(cited.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN cited.chunk_id AS chunk_id, cited.entry AS entry, r.quote AS quote
        """
        with self.driver.session() as session:
            result = session.run(query, cid=chunk_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_cited_clauses_batch(self, chunk_ids: List[str], org_id: str = "DGA") -> List[Dict[str, Any]]:
        """get_cited_clauses for many clauses in one round trip: [{source_id, target_id, rel_type}]."""
        self.connect()
        query = """
        UNWIND $cids AS cid
        MATCH (c:Chunk {chunk_id: cid})-[r:CITES_CLAUSE|EMPOWERED_BY]->(cited:Chunk)
        WHERE coalesce(cited.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN cid AS source_id, cited.chunk_id AS target_id, type(r) AS rel_type
        ORDER BY source_id, CASE type(r) WHEN 'EMPOWERED_BY' THEN 0 ELSE 1 END
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, cids=list(chunk_ids), org_id=org_id)]

    def get_tenant_chunk_citations_batch(self, chunk_ids: List[str], org_id: str) -> List[Dict[str, Any]]:
        """Statute clauses cited by several tenant chunks: [{source_id, target_id, rel_type}]."""
        self.connect()
        query = """
        UNWIND $cids AS cid
        MATCH (t:TenantChunk {chunk_id: cid})-[r:CITES_CLAUSE]->(c:Chunk)
        WHERE t.org_id = $org_id AND coalesce(c.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN cid AS source_id, c.chunk_id AS target_id, 'CITES_CLAUSE' AS rel_type
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(query, cids=list(chunk_ids), org_id=org_id)]

    def get_subordinate_laws(self, chunk_id: str, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """
        Traverses from an Act section to subordinate Ministerial Regulations/Rules
        that cite or derive from this section with tenant filtering.
        """
        self.connect()
        # Subordinate = a clause in *another* document that derives authority from (EMPOWERED_BY)
        # or cites (CITES_CLAUSE) this section; same-document citations are not subordinates.
        query = """
        MATCH (act:Chunk {chunk_id: $cid})<-[r:EMPOWERED_BY|CITES_CLAUSE]-(reg:Chunk)
        WHERE reg.doc_id <> act.doc_id
          AND coalesce(reg.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        MATCH (reg)<-[:CONTAINS]-(doc:Document)
        WHERE coalesce(doc.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        WITH reg, doc, collect(DISTINCT type(r)) AS rels
        RETURN reg.chunk_id AS chunk_id, reg.entry AS entry, doc.title AS document_title,
               CASE WHEN 'EMPOWERED_BY' IN rels THEN 'EMPOWERED_BY' ELSE 'CITES_CLAUSE' END AS relation
        ORDER BY relation DESC, document_title
        """
        with self.driver.session() as session:
            result = session.run(query, cid=chunk_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_related_cases(self, chunk_id: str, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fetch FAQ cases interpreting or relating to this statutory clause with tenant filtering."""
        self.connect()
        query = """
        MATCH (f:FAQCase)-[:RELATES_TO_LAW]->(c:Chunk {chunk_id: $cid})
        WHERE coalesce(f.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN f.case_id AS case_id, f.question_preview AS question
        """
        with self.driver.session() as session:
            result = session.run(query, cid=chunk_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_document_clusters(self, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fetch high-level document clusters with clause counts with tenant filtering."""
        self.connect()
        query = """
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
        WHERE coalesce(d.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
          AND coalesce(c.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN d.doc_id AS cluster_id, d.title AS title, count(c) AS clause_count
        ORDER BY clause_count DESC
        """
        with self.driver.session() as session:
            result = session.run(query, org_id=org_id)
            return [dict(r) for r in result]

    # ==========================================
    # Tenant-private documents (OCR'd TOR / BOQ / contracts)
    # ==========================================
    # Tenant nodes always carry the owner's org_id and every read filters on it exactly
    # (never PUBLIC), so a CITES_CLAUSE edge into a shared statute leaks nothing across tenants.

    def resolve_statute_refs(self, refs: List[Dict[str, Any]]) -> Dict[Tuple[str, str, int], List[str]]:
        """{(doc_title, kind, num): [chunk_id, ...]} for ('section'|'clause') references by document title."""
        if not refs:
            return {}
        self.connect()
        query = """
        UNWIND $refs AS ref
        MATCH (d:Document {title: ref.title})-[:CONTAINS]->(c:Chunk)
        WHERE (ref.kind = 'section' AND c.section_num = ref.num)
           OR (ref.kind = 'clause' AND c.clause_num = ref.num)
        RETURN ref.title AS title, ref.kind AS kind, ref.num AS num, collect(c.chunk_id) AS ids
        """
        with self.driver.session() as session:
            result = session.run(query, refs=refs)
            return {(r["title"], r["kind"], r["num"]): r["ids"] for r in result}

    def replace_tenant_document(
        self,
        org_id: str,
        doc: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> int:
        """Drop any previous version, then write TenantDocument -CONTAINS-> TenantChunk -NEXT_CHUNK->
        and TenantChunk -CITES_CLAUSE-> Chunk. Returns the number of citation edges."""
        self.connect()
        with self.driver.session() as session:
            return session.execute_write(self._replace_tenant_document_tx, org_id, doc, chunks, citations)

    @staticmethod
    def _replace_tenant_document_tx(tx, org_id, doc, chunks, citations) -> int:
        tx.run(
            """
            MATCH (d:TenantDocument {doc_id: $doc_id}) WHERE d.org_id = $org_id
            OPTIONAL MATCH (d)-[:CONTAINS]->(t:TenantChunk)
            DETACH DELETE d, t
            """,
            doc_id=doc["doc_id"], org_id=org_id,
        )
        tx.run(
            """
            CREATE (d:TenantDocument {doc_id: $doc.doc_id, org_id: $org_id, title: $doc.title,
                                      source_file: $doc.source_file, total_pages: $doc.total_pages})
            WITH d
            UNWIND $chunks AS ch
            CREATE (d)-[:CONTAINS]->(:TenantChunk {chunk_id: ch.chunk_id, org_id: $org_id, doc_id: $doc.doc_id,
                                                   chunk_index: ch.chunk_index, page_start: ch.page_start})
            """,
            doc=doc, org_id=org_id,
            chunks=[{k: c.get(k) for k in ("chunk_id", "chunk_index", "page_start")} for c in chunks],
        )
        tx.run(
            """
            MATCH (d:TenantDocument {doc_id: $doc_id})-[:CONTAINS]->(a:TenantChunk)
            WHERE d.org_id = $org_id
            WITH a ORDER BY a.chunk_index
            WITH collect(a) AS seq
            UNWIND range(0, size(seq) - 2) AS i
            WITH seq[i] AS a, seq[i + 1] AS b
            CREATE (a)-[:NEXT_CHUNK]->(b)
            """,
            doc_id=doc["doc_id"], org_id=org_id,
        )
        if not citations:
            return 0
        rec = tx.run(
            """
            UNWIND $cites AS ct
            MATCH (t:TenantChunk {chunk_id: ct.chunk_id}) WHERE t.org_id = $org_id
            MATCH (c:Chunk {chunk_id: ct.chunk_id})
            MERGE (t)-[r:CITES_CLAUSE]->(c)
            SET r.quote = ct.quote
            RETURN count(r) AS n
            """,
            cites=citations, org_id=org_id,
        ).single()
        return rec["n"] if rec else 0

    def delete_tenant_document(self, org_id: str, doc_id: str) -> None:
        self.connect()
        with self.driver.session() as session:
            session.run(
                """
                MATCH (d:TenantDocument {doc_id: $doc_id}) WHERE d.org_id = $org_id
                OPTIONAL MATCH (d)-[:CONTAINS]->(t:TenantChunk)
                DETACH DELETE d, t
                """,
                doc_id=doc_id, org_id=org_id,
            )

    def get_tenant_chunk_citations(self, chunk_id: str, org_id: str) -> List[Dict[str, Any]]:
        """Statute clauses cited by a tenant chunk, limited to clauses this tenant may see."""
        self.connect()
        query = """
        MATCH (t:TenantChunk {chunk_id: $cid})-[r:CITES_CLAUSE]->(c:Chunk)
        WHERE t.org_id = $org_id AND coalesce(c.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN c.chunk_id AS chunk_id, c.entry AS entry, r.quote AS quote
        """
        with self.driver.session() as session:
            result = session.run(query, cid=chunk_id, org_id=org_id)
            return [dict(r) for r in result]

    def count_stats(self) -> Dict[str, int]:
        """Return counts of nodes and relationships in Neo4j."""
        self.connect()
        query = """
        CALL () {
            MATCH (d:Document) RETURN count(d) AS docs
        }
        CALL () {
            MATCH (c:Chunk) RETURN count(c) AS chunks
        }
        CALL () {
            MATCH (f:FAQCase) RETURN count(f) AS cases
        }
        CALL () {
            MATCH ()-[r]->() RETURN count(r) AS rels
        }
        RETURN docs, chunks, cases, rels
        """
        with self.driver.session() as session:
            rec = session.run(query).single()
            if rec:
                return {
                    "documents": rec["docs"],
                    "chunks": rec["chunks"],
                    "cases": rec["cases"],
                    "relationships": rec["rels"],
                }
            return {}
