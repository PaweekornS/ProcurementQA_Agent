# -*- coding: utf-8 -*-
"""
tests/test_main_qa_engine.py

Validates the Main Agentic QA Reasoning Engine:
1. Verifies that /api/v1/qa processes legal questions and returns compliant structured responses.
2. Asserts presence of decisive_quotes, issues_breakdown, and applicable_laws.
3. Asserts absence of deprecated 'mode' parameter.
4. Asserts correct propagation of tenant organization_id.
5. Verifies validation error (HTTP 400) on empty question.
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
        """Validates end-to-end legal QA inquiry returning structured answer and citations."""
        payload = {
            "question": "หน่วยงานสามารถจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจงในวงเงินไม่เกินเท่าใด และต้องขออนุมัติใคร",
            "org_id": "ORG_01_DGA"
        }
        response = self.client.post("/api/v1/qa", json=payload)
        self.assertEqual(response.status_code, 200, f"QA failed: {response.text}")
        data = response.json()

        # Contract assertions
        self.assertNotIn("mode", data, "Regression: 'mode' should not be present in response!")
        self.assertEqual(data.get("organization_id"), "ORG_01_DGA")
        self.assertIn("status", data)
        self.assertIn("direct_answer", data)
        self.assertTrue(len(data.get("direct_answer", "").strip()) > 0, "direct_answer is empty")
        self.assertIsInstance(data.get("applicable_laws"), list)
        self.assertIsInstance(data.get("decisive_quotes"), list)
        self.assertIsInstance(data.get("issues_breakdown"), list)

    def test_02_empty_question_validation(self):
        """Validates that empty question strings return HTTP 400 Bad Request."""
        payload = {"question": "   ", "org_id": "DGA"}
        response = self.client.post("/api/v1/qa", json=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
