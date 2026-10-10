# -*- coding: utf-8 -*-
"""
tests/test_tri_store_integration.py

Integration verification for Tri-Store Database Architecture:
1. Validates PostgreSQL connectivity, schemas, and counts.
2. Validates Qdrant collection vectors (1024-dim dense + native sparse BM25).
3. Validates Neo4j knowledge graph traversal (CITES_CLAUSE, EMPOWERED_BY, ADJACENT_SECTION).
4. Validates ProcurementService MCP tool calls.
"""

import os
import sys
import unittest
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# override=False: keep container/CI endpoints (POSTGRES_HOST=postgres, ...) over .env localhost values
load_dotenv(override=False)
os.environ["USE_TRI_STORE"] = "true"

import json

from core.database import StorageManager
from core.service import ProcurementService
from scripts.migrate_to_tri_store import GRAPH_LINKER_VERSION, normalize_doc_name


def _corpus_expectations() -> dict:
    """
    Minimum counts derived from the chunked corpus in outputs/corpus/ (the PUBLIC tenant), parsed exactly as
    scripts/migrate_to_tri_store.py does. Stores may hold more when tenant benchmark data is
    loaded on top, hence the >= assertions.
    """
    laws_path = os.getenv("law_to_crime_path", str(PROJECT_ROOT / "outputs" / "corpus" / "law_to_crime.json"))
    cases_path = os.getenv("case_db_path", str(PROJECT_ROOT / "outputs" / "corpus" / "cases_with_feature.json"))
    with open(laws_path, encoding="utf-8") as f:
        laws = [r for r in json.load(f) if r.get("items")]
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    return {
        "legal_documents": len({normalize_doc_name(str(r.get("id", "")).split("|")[0]) for r in laws}),
        "statute_clauses": len(laws),
        "faq_cases": len(cases),
    }


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
        
        # PostgreSQL assertions: at least the full real corpus is loaded
        expected = _corpus_expectations()
        for key, minimum in expected.items():
            self.assertGreaterEqual(stats["postgres"][key], minimum, f"{key} below corpus size")

        # Qdrant assertions
        self.assertEqual(stats["qdrant"]["statutes_points"], stats["postgres"]["statute_clauses"])
        self.assertEqual(stats["qdrant"]["cases_points"], stats["postgres"]["faq_cases"])

        # Neo4j assertions
        self.assertEqual(stats["neo4j"]["documents"], stats["postgres"]["legal_documents"])
        self.assertEqual(stats["neo4j"]["clauses"], stats["postgres"]["statute_clauses"])
        # Every clause hangs off its document via CONTAINS, so citations/adjacency must add more
        self.assertGreater(stats["neo4j"]["relationships"], stats["neo4j"]["clauses"])
        self.assertEqual(self.storage.neo4j.get_graph_meta("linker_version"), GRAPH_LINKER_VERSION)

    def test_02_hybrid_search_clauses(self):
        """Verify hybrid vector + sparse search in Qdrant with Postgres hydration."""
        query = "การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000 บาท"
        from core.retrieval.embedding import get_embedding
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

    def test_05_cross_document_subordinate_links(self):
        """Act มาตรา 56 must reach subordinate instruments in *other* documents (EMPOWERED_BY / CITES_CLAUSE)."""
        records = self.storage.pg.lookup_section("", 56)
        self.assertGreater(len(records), 0)
        self.assertTrue(records[0]["doc_title"].startswith("พระราชบัญญัติ"), "bare section lookup must rank the Act first")

        subordinates = self.storage.neo4j.get_subordinate_laws(records[0]["clause_id"])
        print(f"\n[Test 5] มาตรา 56 has {len(subordinates)} subordinate clauses.")
        self.assertGreater(len(subordinates), 0)
        self.assertIn("EMPOWERED_BY", {s["relation"] for s in subordinates})
        self.assertTrue(all(s["document_title"] != records[0]["doc_title"] for s in subordinates))

    def test_06_ambiguous_clause_lookup_is_flagged(self):
        """A bare 'ข้อ N' that exists in several documents is flagged so the caller (MCP get_statute_section) can disambiguate."""
        res = self.service.lookup_section("ข้อ 2")
        print(f"\n[Test 6] 'ข้อ 2' resolved to: {res.get('doc_title')}")
        self.assertTrue(res.get("found"))
        self.assertTrue(res.get("ambiguous"))
        self.assertGreater(len(res.get("other_documents_with_same_number", [])), 0)

        scoped = self.service.lookup_section("ข้อ 2", doc_title=res["other_documents_with_same_number"][0])
        self.assertTrue(scoped.get("found"))
        self.assertNotIn("ambiguous", scoped)

    def test_07_reranker_loaded_and_scoring(self):
        """
        The cross-encoder must actually load and score candidates. A missing runtime dependency
        once made it fail silently: retrieval ran unreranked and Hit/MRR collapsed with no error.
        """
        from core.retrieval.reranker import reranker_status
        from core.retrieval.retriever import get_retriever

        status = reranker_status()
        self.assertTrue(status["loaded"], f"reranker not usable: {status}")
        if status["enabled"]:
            result = get_retriever().retrieve(["วิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000 บาท"], "DGA", top_k=5)
            self.assertTrue(result.stats.get("reranked"), "retrieval ran without the cross-encoder")


if __name__ == "__main__":
    unittest.main()
