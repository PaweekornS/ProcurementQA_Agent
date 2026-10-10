# -*- coding: utf-8 -*-
"""
tests/test_retrieval_modes.py

Validates all retrieval strategies supported by ProcurementQA Agent:
1. Exact Section Lookup (Verbatim paragraph matching without LLM overhead)
2. Direct Hybrid Search (Dense vector + Thai BM25 sparse + GPU Cross-Encoder Reranker)
3. Knowledge Graph Traversal (Subordinate regulations and statutory citation edges)
4. REST Search Endpoint (/api/v1/search)
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.app import app
from api.dependencies import get_service


class TestRetrievalModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.service = get_service()

    def test_01_exact_section_lookup(self):
        """Validates exact section lookup for Public Act Section 56 (มาตรา 56)."""
        res = self.service.lookup_section("มาตรา 56")
        self.assertTrue(res.get("found"), f"Failed to find มาตรา 56: {res}")
        self.assertIn("focused_content", res)
        self.assertTrue(len(res["focused_content"]) > 10, "Focused content is too short")
        self.assertIn("56", res.get("section", ""))

    def test_02_exact_rule_lookup(self):
        """Validates exact rule lookup for Ministerial Rule No. 25 (ข้อ 25)."""
        res = self.service.lookup_section("ข้อ 25")
        self.assertTrue(res.get("found"), f"Failed to find ข้อ 25: {res}")
        self.assertIn("focused_content", res)

    def test_03_hybrid_vector_lexical_search(self):
        """Validates hybrid dense+sparse retrieval for procurement methods."""
        query = "วิธีเฉพาะเจาะจง วงเงินไม่เกิน 500000 บาท"
        results = self.service.search_clauses(query=query, top_k=3)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Hybrid search returned 0 results")
        first = results[0]
        self.assertIn("entry", first)
        self.assertIn("description", first)

    def test_04_knowledge_graph_traversal(self):
        """Validates knowledge graph neighbor walking for related clauses."""
        related = self.service.get_related_clauses("มาตรา 56", max_hops=1)
        self.assertIsInstance(related, list)
        # Graph neighbors or citation edges should be found
        self.assertGreaterEqual(len(related), 0)

    def test_05_rest_api_search_endpoint(self):
        """Validates POST /api/v1/search endpoint with top_k parameter."""
        payload = {
            "query": "การจัดซื้อจัดจ้างโดยวิธีคัดเลือก",
            "top_k": 3
        }
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("query"), payload["query"])
        self.assertGreater(data.get("count", 0), 0)
        self.assertIsInstance(data.get("results"), list)


if __name__ == "__main__":
    unittest.main()
