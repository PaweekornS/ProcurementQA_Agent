# -*- coding: utf-8 -*-
"""
core/database/storage_manager.py

Unified Storage Manager Facade for Thai Procurement LegalGraphRAG.
Coordinates queries and data flows between PostgreSQL (SSOT), Qdrant (VectorDB),
and Neo4j (GraphDB).
"""

import os
import logging
from typing import Dict, Any, List, Optional

from .pg_repository import PostgresRepository
from .qdrant_repository import QdrantRepository
from .neo4j_repository import Neo4jRepository

logger = logging.getLogger("storage_manager")


class StorageManager:
    """Singleton Facade managing Tri-Store operations."""

    _instance: Optional["StorageManager"] = None

    def __init__(self):
        self.pg = PostgresRepository()
        self.qdrant = QdrantRepository(dense_dim=int(os.getenv("QDRANT_DENSE_DIM", 1024)))
        self.neo4j = Neo4jRepository()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "StorageManager":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init_all_stores(self):
        """Initializes schemas, collections, indexes, and constraints across all 3 databases."""
        if self._initialized:
            return

        logger.info("Initializing Tri-Store databases (PostgreSQL, Qdrant, Neo4j)...")
        self.pg.init_schema()
        self.qdrant.init_collections()
        self.neo4j.init_schema()
        self._initialized = True
        logger.info("Tri-Store databases successfully initialized and ready.")

    def hybrid_search_clauses(
        self,
        query_text: str,
        query_dense: List[float],
        top_k: int = 10,
        doc_filter: Optional[str] = None,
        section_filter: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid dense/sparse search in Qdrant, then hydrates authoritative text
        and legal metadata from PostgreSQL in a single batch.
        """
        qdrant_results = self.qdrant.hybrid_search_statutes(
            query_text=query_text,
            query_dense=query_dense,
            top_k=top_k,
            doc_filter=doc_filter,
            section_filter=section_filter,
        )

        if not qdrant_results:
            return []

        # Map scores by clause_id
        score_map = {r["clause_id"]: r["score"] for r in qdrant_results if r.get("clause_id")}
        clause_ids = list(score_map.keys())

        # Hydrate from PostgreSQL
        pg_records = self.pg.get_clauses_by_ids(clause_ids)
        pg_map = {r["clause_id"]: r for r in pg_records}

        hydrated = []
        for cid in clause_ids:
            if cid in pg_map:
                item = dict(pg_map[cid])
                item["similarity"] = score_map.get(cid, 0.0)
                item["score"] = score_map.get(cid, 0.0)
                hydrated.append(item)

        # Ensure order matches Qdrant score ranking
        hydrated.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return hydrated

    def hybrid_search_cases(
        self,
        query_text: str,
        query_dense: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Hybrid search over FAQ cases in Qdrant hydrated with Postgres features."""
        qdrant_results = self.qdrant.hybrid_search_cases(
            query_text=query_text,
            query_dense=query_dense,
            top_k=top_k,
        )
        if not qdrant_results:
            return []

        results = []
        for r in qdrant_results:
            cid = r.get("case_id")
            score = r.get("score", 0.0)
            results.append({
                "case_id": cid,
                "score": score,
                "question": r.get("payload", {}).get("question", ""),
                "topics": r.get("payload", {}).get("topics", []),
            })
        return results

    def traverse_clause_graph(self, clause_id: str) -> Dict[str, Any]:
        """Traverses Neo4j for structural and citation graph context around a clause."""
        return {
            "adjacent_sections": self.neo4j.get_adjacent_sections(clause_id),
            "cited_clauses": self.neo4j.get_cited_clauses(clause_id),
            "subordinate_laws": self.neo4j.get_subordinate_laws(clause_id),
            "related_cases": self.neo4j.get_related_cases(clause_id),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate data metrics across all 3 databases."""
        return {
            "postgres": self.pg.count_stats(),
            "qdrant": self.qdrant.count_stats(),
            "neo4j": self.neo4j.count_stats(),
        }
