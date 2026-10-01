# -*- coding: utf-8 -*-
"""
tests/test_tri_store_integration.py

Integration verification for Tri-Store Database Architecture:
1. Validates PostgreSQL connectivity, schemas, and counts.
2. Validates Qdrant collection vectors (1024-dim dense + native sparse BM25).
3. Validates Neo4j knowledge graph traversal (CITES_CLAUSE, ADJACENT_SECTION).
4. Validates ProcurementService MCP tool calls.
"""

import os
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(override=True)
os.environ["USE_TRI_STORE"] = "true"

from core.database import StorageManager
from core.service import ProcurementService


class TestTriStoreIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.storage = StorageManager.get_instance()
        cls.storage.init_all_stores()
        cls.service = ProcurementService.get_instance()

    def test_01_store_counts(self):
        """Verify data counts across PostgreSQL, Qdrant, and Neo4j."""
        stats = self.storage.get_stats()
        print("\n[Test 1] DB Counts:", stats)
        
        # PostgreSQL assertions
        self.assertGreaterEqual(stats["postgres"]["legal_documents"], 100)
        self.assertGreaterEqual(stats["postgres"]["statute_clauses"], 3000)
        self.assertGreaterEqual(stats["postgres"]["faq_cases"], 20)

        # Qdrant assertions
        self.assertEqual(stats["qdrant"]["statutes_points"], stats["postgres"]["statute_clauses"])
        self.assertEqual(stats["qdrant"]["cases_points"], stats["postgres"]["faq_cases"])

        # Neo4j assertions
        self.assertEqual(stats["neo4j"]["documents"], stats["postgres"]["legal_documents"])
        self.assertEqual(stats["neo4j"]["clauses"], stats["postgres"]["statute_clauses"])
        self.assertGreater(stats["neo4j"]["relationships"], 5000)

    def test_02_hybrid_search_clauses(self):
        """Verify hybrid vector + sparse search in Qdrant with Postgres hydration."""
        query = "การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000 บาท"
        from core.graph_construct.feature_graph import get_embedding
        q_emb = get_embedding(query)

        results = self.storage.hybrid_search_clauses(
            query_text=query,
            query_dense=q_emb,
            top_k=5
        )
        print(f"\n[Test 2] Hybrid search returned {len(results)} clauses.")
        self.assertGreater(len(results), 0)
        top_clause = results[0]
        self.assertIn("clause_id", top_clause)
        self.assertIn("content_thai", top_clause)
        self.assertGreater(len(top_clause["content_thai"]), 20)
        print(f"Top result: {top_clause.get('entry')} (score: {top_clause.get('score'):.4f})")

    def test_03_graph_traversal(self):
        """Verify Neo4j graph traversal for adjacent sections and citations."""
        # Query section 56
        records = self.storage.pg.lookup_section("พระราชบัญญัติ", 56)
        self.assertGreater(len(records), 0)
        clause_id = records[0]["clause_id"]

        graph_ctx = self.storage.traverse_clause_graph(clause_id)
        print(f"\n[Test 3] Graph traversal around '{records[0]['entry']}':")
        print(f"  - Adjacent sections: {len(graph_ctx['adjacent_sections'])}")
        print(f"  - Cited clauses: {len(graph_ctx['cited_clauses'])}")

        self.assertIn("adjacent_sections", graph_ctx)
        self.assertIn("cited_clauses", graph_ctx)

    def test_04_mcp_service_tools(self):
        """Verify domain service MCP endpoints backed by Tri-Store."""
        # 1. Search clauses
        clauses = self.service.search_clauses("วิธีคัดเลือก", top_k=3)
        print(f"\n[Test 4] Service search_clauses returned {len(clauses)} results.")
        self.assertGreater(len(clauses), 0)
        self.assertIn("content", clauses[0])

        # 2. Traverse regulations
        graph_res = self.service.traverse_regulations("มาตรา 56")
        print(f"[Test 4] Service traverse_regulations returned {graph_res['graph_neighbors_count']} neighbors.")
        self.assertGreaterEqual(graph_res["graph_neighbors_count"], 1)


if __name__ == "__main__":
    unittest.main()
