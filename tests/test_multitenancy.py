# -*- coding: utf-8 -*-
"""
tests/test_multitenancy.py

Validates Multi-Tenant Data Isolation and Tenant Propagation:
1. Verifies that querying with tenant X returns strictly nodes from ['PUBLIC', X].
2. Verifies zero cross-tenant leakage between ORG_01_DGA and ORG_05_MOPH.
3. Verifies that FastAPI resolves org_id from either request body or X-Organization-Id header.
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


class TestMultiTenancy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.service = get_service()

    def test_01_search_scoping_isolation(self):
        """Verifies that search_clauses filtered by org_id strictly prohibits foreign tenant data."""
        tenants_to_test = ["ORG_01_DGA", "ORG_05_MOPH"]
        for tenant_id in tenants_to_test:
            results = self.service.search_clauses(
                query="ข้อกำหนดและแนวปฏิบัติพัสดุเฉพาะ",
                top_k=5,
                org_id=tenant_id
            )
            for item in results:
                data = item.get("data") or item
                item_org = data.get("org_id", "PUBLIC")
                self.assertIn(
                    item_org,
                    {"PUBLIC", tenant_id},
                    f"Data Leakage Detected! Query for '{tenant_id}' returned item belonging to '{item_org}'"
                )

    def test_02_api_body_org_id_resolution(self):
        """Verifies that POST /api/v1/search correctly respects org_id in request body."""
        payload = {
            "query": "วิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000",
            "top_k": 3,
            "org_id": "ORG_05_MOPH"
        }
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("org_id"), "ORG_05_MOPH")

    def test_03_api_header_org_id_resolution(self):
        """Verifies that X-Organization-Id header resolves tenant when omitted from body."""
        payload = {
            "query": "วิธีเฉพาะเจาะจง",
            "top_k": 2
        }
        headers = {"X-Organization-Id": "ORG_02_DEPA"}
        response = self.client.post("/api/v1/search", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("org_id"), "ORG_02_DEPA")

    def test_04_default_tenant_fallback(self):
        """Verifies that missing org_id falls back cleanly to DEFAULT_ORG_ID (e.g. DGA)."""
        payload = {"query": "การจัดซื้อจัดจ้าง", "top_k": 2}
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        expected_default = os.getenv("DEFAULT_ORG_ID", "DGA")
        self.assertEqual(data.get("org_id"), expected_default)


if __name__ == "__main__":
    unittest.main()
