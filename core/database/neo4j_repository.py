# -*- coding: utf-8 -*-
"""
core/database/neo4j_repository.py

Neo4j Graph Database Repository for Thai Procurement LegalGraphRAG.
Manages topological relationships: document hierarchies, sequential clause adjacency
(ADJACENT_SECTION), cross-statutory citations (CITES_CLAUSE), and precedent links (RELATES_TO_LAW).
"""

import os
import logging
from typing import List, Dict, Any, Optional

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
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=30.0,
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
        """Idempotently create uniqueness constraints and search indexes, including tenant indexes."""
        self.connect()
        constraints = [
            "CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (d:LegalDocument) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT unique_clause_id IF NOT EXISTS FOR (c:StatuteClause) REQUIRE c.clause_id IS UNIQUE",
            "CREATE CONSTRAINT unique_case_id IF NOT EXISTS FOR (f:FAQCase) REQUIRE f.case_id IS UNIQUE",
            "CREATE CONSTRAINT unique_topic_name IF NOT EXISTS FOR (t:LegalTopic) REQUIRE t.name IS UNIQUE",
            "CREATE INDEX clause_section_lookup IF NOT EXISTS FOR (c:StatuteClause) ON (c.doc_id, c.section_num)",
            "CREATE INDEX clause_number_lookup IF NOT EXISTS FOR (c:StatuteClause) ON (c.doc_id, c.clause_num)",
            "CREATE INDEX doc_org_id_lookup IF NOT EXISTS FOR (d:LegalDocument) ON (d.org_id)",
            "CREATE INDEX clause_org_id_lookup IF NOT EXISTS FOR (c:StatuteClause) ON (c.org_id)",
            "CREATE INDEX case_org_id_lookup IF NOT EXISTS FOR (f:FAQCase) ON (f.org_id)",
        ]
        with self.driver.session() as session:
            for stmt in constraints:
                session.run(stmt)
        logger.info("Neo4j constraints and indexes successfully initialized.")

    def sync_documents(self, documents: List[Dict[str, Any]], org_id: str = "PUBLIC"):
        """Batch upsert :LegalDocument nodes with tenant org_id."""
        self.connect()
        query = """
        UNWIND $batch AS doc
        MERGE (d:LegalDocument {doc_id: doc.doc_id})
        SET d.title = doc.title,
            d.doc_type = doc.doc_type,
            d.year_be = doc.year_be,
            d.source_file = doc.source_file,
            d.org_id = coalesce(doc.org_id, $default_org_id, 'PUBLIC')
        """
        with self.driver.session() as session:
            session.run(query, batch=documents, default_org_id=org_id)

    def sync_statute_clauses(self, clauses: List[Dict[str, Any]], org_id: str = "PUBLIC", batch_size: int = 500):
        """Batch upsert :StatuteClause nodes and connect them to their parent :LegalDocument."""
        self.connect()
        query = """
        UNWIND $batch AS item
        MERGE (c:StatuteClause {clause_id: item.clause_id})
        SET c.doc_id = item.doc_id,
            c.entry = item.entry,
            c.chapter_num = item.chapter_num,
            c.section_num = item.section_num,
            c.clause_num = item.clause_num,
            c.page_start = item.page_start,
            c.page_end = item.page_end,
            c.org_id = coalesce(item.org_id, $default_org_id, 'PUBLIC')
        WITH c, item
        MATCH (d:LegalDocument {doc_id: item.doc_id})
        MERGE (d)-[:CONTAINS]->(c)
        """
        for i in range(0, len(clauses), batch_size):
            chunk = clauses[i:i + batch_size]
            with self.driver.session() as session:
                session.run(query, batch=chunk, default_org_id=org_id)
            logger.info(f"Synced {len(chunk)} StatuteClause nodes to Neo4j.")

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
            MATCH (src {{clause_id: e.source_id}})
            MATCH (tgt {{clause_id: e.target_id}})
            MERGE (src)-[r:{rel_type}]->(tgt)
            SET r += coalesce(e.props, {{}})
            """
            for i in range(0, len(edge_list), batch_size):
                chunk = edge_list[i:i + batch_size]
                with self.driver.session() as session:
                    session.run(query, batch=chunk)
            logger.info(f"Synced {len(edge_list)} relationships of type ':{rel_type}' to Neo4j.")

    def link_case_to_laws(self, links: List[Dict[str, str]]):
        """Links :FAQCase to :StatuteClause via :RELATES_TO_LAW."""
        self.connect()
        query = """
        UNWIND $batch AS link
        MATCH (f:FAQCase {case_id: link.case_id})
        MATCH (c:StatuteClause {clause_id: link.clause_id})
        MERGE (f)-[:RELATES_TO_LAW]->(c)
        """
        with self.driver.session() as session:
            session.run(query, batch=links)

    # ==========================================
    # Graph Traversal Queries (For CRAG Engine)
    # ==========================================

    def get_adjacent_sections(self, clause_id: str, direction: str = "both", org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Traverse sequential adjacent sections (ADJACENT_SECTION) with tenant filtering."""
        self.connect()
        if direction == "next":
            rel_pattern = "-[:ADJACENT_SECTION {direction: 'next'}]->"
        elif direction == "prev":
            rel_pattern = "-[:ADJACENT_SECTION {direction: 'prev'}]->"
        else:
            rel_pattern = "-[:ADJACENT_SECTION]-"

        query = f"""
        MATCH (c:StatuteClause {{clause_id: $cid}}){rel_pattern}(adj:StatuteClause)
        WHERE coalesce(adj.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN adj.clause_id AS clause_id, adj.entry AS entry, adj.section_num AS section_num
        """
        with self.driver.session() as session:
            result = session.run(query, cid=clause_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_cited_clauses(self, clause_id: str, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fetch all clauses cited by this clause (CITES_CLAUSE) with tenant filtering."""
        self.connect()
        query = """
        MATCH (c:StatuteClause {clause_id: $cid})-[r:CITES_CLAUSE]->(cited:StatuteClause)
        WHERE coalesce(cited.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN cited.clause_id AS clause_id, cited.entry AS entry, r.quote AS quote
        """
        with self.driver.session() as session:
            result = session.run(query, cid=clause_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_subordinate_laws(self, clause_id: str, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """
        Traverses from an Act section to subordinate Ministerial Regulations/Rules
        that cite or derive from this section with tenant filtering.
        """
        self.connect()
        query = """
        MATCH (act:StatuteClause {clause_id: $cid})<-[:CITES_CLAUSE]-(reg:StatuteClause)
        WHERE coalesce(reg.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        MATCH (reg)<-[:CONTAINS]-(doc:LegalDocument)
        WHERE coalesce(doc.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN reg.clause_id AS clause_id, reg.entry AS entry, doc.title AS document_title
        """
        with self.driver.session() as session:
            result = session.run(query, cid=clause_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_related_cases(self, clause_id: str, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fetch FAQ cases interpreting or relating to this statutory clause with tenant filtering."""
        self.connect()
        query = """
        MATCH (f:FAQCase)-[:RELATES_TO_LAW]->(c:StatuteClause {clause_id: $cid})
        WHERE coalesce(f.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN f.case_id AS case_id, f.question_preview AS question
        """
        with self.driver.session() as session:
            result = session.run(query, cid=clause_id, org_id=org_id)
            return [dict(r) for r in result]

    def get_document_clusters(self, org_id: str = "DGA") -> List[Dict[str, Any]]:
        """Fetch high-level document clusters with clause counts with tenant filtering."""
        self.connect()
        query = """
        MATCH (d:LegalDocument)-[:CONTAINS]->(c:StatuteClause)
        WHERE coalesce(d.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
          AND coalesce(c.org_id, 'PUBLIC') IN ['PUBLIC', $org_id]
        RETURN d.doc_id AS cluster_id, d.title AS title, count(c) AS clause_count
        ORDER BY clause_count DESC
        """
        with self.driver.session() as session:
            result = session.run(query, org_id=org_id)
            return [dict(r) for r in result]

    def count_stats(self) -> Dict[str, int]:
        """Return counts of nodes and relationships in Neo4j."""
        self.connect()
        query = """
        CALL () {
            MATCH (d:LegalDocument) RETURN count(d) AS docs
        }
        CALL () {
            MATCH (c:StatuteClause) RETURN count(c) AS clauses
        }
        CALL () {
            MATCH (f:FAQCase) RETURN count(f) AS cases
        }
        CALL () {
            MATCH ()-[r]->() RETURN count(r) AS rels
        }
        RETURN docs, clauses, cases, rels
        """
        with self.driver.session() as session:
            rec = session.run(query).single()
            if rec:
                return {
                    "documents": rec["docs"],
                    "clauses": rec["clauses"],
                    "cases": rec["cases"],
                    "relationships": rec["rels"],
                }
            return {}
