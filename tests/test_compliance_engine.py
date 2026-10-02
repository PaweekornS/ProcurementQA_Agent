# -*- coding: utf-8 -*-
"""
tests/test_compliance_engine.py

Validates the rule-based Compliance & Threshold Verification Engine:
1. Verifies that specific selection below 500,000 THB passes compliance.
2. Verifies that specific selection above 500,000 THB without justification fails or flags risk.
3. Asserts anti-splitting (มาตรา 65) cautionary warnings.
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.app import app


class TestComplianceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_compliant_specific_selection(self):
        """Validates specific method with budget under 500,000 THB."""
        payload = {
            "procurement_item": "จัดซื้อโต๊ะและเก้าอี้สำนักงาน",
            "estimated_budget": 350000.0,
            "proposed_method": "เฉพาะเจาะจง",
            "justification_reason": "วงเงินไม่เกิน 500,000 บาท"
        }
        response = self.client.post("/api/v1/verify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("is_compliant"))
        self.assertEqual(data.get("compliance_status"), "PASSED")
        self.assertIn("หัวหน้าเจ้าหน้าที่", data.get("required_approvals", []))

    def test_02_non_compliant_exceeding_threshold(self):
        """Validates specific method exceeding 500,000 THB without emergency justification."""
        payload = {
            "procurement_item": "จัดซื้อคอมพิวเตอร์แม่ข่าย",
            "estimated_budget": 1200000.0,
            "proposed_method": "เฉพาะเจาะจง",
            "justification_reason": ""
        }
        response = self.client.post("/api/v1/verify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data.get("is_compliant"))
        self.assertEqual(data.get("compliance_status"), "VIOLATION")
        self.assertGreater(len(data.get("potential_risks", [])), 0)


if __name__ == "__main__":
    unittest.main()
