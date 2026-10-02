# -*- coding: utf-8 -*-
"""
tests/test_main_qa_engine.py

Validates the Main Agentic QA Reasoning Engine:
1. /api/v1/qa returns exactly the Super-Orchestrator contract
   (status, answer, conditions, citations, unresolved_issues, grounded, org_id).
2. Citations carry law, quote, filename and page (null when unknown).
3. Tenant org_id propagation, empty-query 400 and legacy 'question' field 422.
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.app import app


class TestMainQAEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_main_qa_execution(self):
        """End-to-end QA returns the orchestrator contract scoped to the requested tenant."""
        payload = {
            "query": "หน่วยงานสามารถจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจงในวงเงินไม่เกินเท่าใด และต้องขออนุมัติใคร",
            "org_id": "ORG_01_DGA"
        }
        response = self.client.post("/api/v1/qa", json=payload)
        self.assertEqual(response.status_code, 200, f"QA failed: {response.text}")
        data = response.json()

        self.assertEqual(
            set(data),
            {"status", "answer", "conditions", "citations", "unresolved_issues", "grounded", "org_id"},
        )
        self.assertEqual(data["org_id"], "ORG_01_DGA")
        self.assertTrue(data["answer"].strip(), "answer is empty")
        for citation in data["citations"]:
            self.assertEqual(set(citation), {"law", "quote", "filename", "page"})
            self.assertNotIn(citation["page"], ("", "ไม่ระบุ"), "unknown page must be null")
        self.assertTrue(any(c["filename"] and c["page"] for c in data["citations"]),
                        "at least one citation should resolve to a source document page")
        for issue in data["unresolved_issues"]:
            self.assertNotEqual(issue["status"], "RESOLVED")

    def test_02_empty_query_validation(self):
        """Whitespace-only queries are rejected with HTTP 400."""
        response = self.client.post("/api/v1/qa", json={"query": "   ", "org_id": "DGA"})
        self.assertEqual(response.status_code, 400)

    def test_03_legacy_question_field_rejected(self):
        """The pre-refactor 'question' field is no longer accepted (422 makes the breaking change explicit)."""
        response = self.client.post("/api/v1/qa", json={"question": "มาตรา 56 คืออะไร"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
