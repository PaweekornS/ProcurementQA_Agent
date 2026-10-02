# -*- coding: utf-8 -*-
"""
tests/test_healthcheck.py

Validates system readiness, liveness probes, and component pre-warming:
- GET /healthz (Liveness)
- GET /ready (Readiness & Tri-Store / Graph status)
- Direct ProcurementService instance readiness check
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


class TestHealthcheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.service = get_service()

    def test_01_liveness_healthz(self):
        """Validates that /healthz returns 200 OK and status 'healthy'."""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")

    def test_02_readiness_ready(self):
        """Validates that /ready returns 200 OK and reports service components."""
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("ready"), "Service reported ready=False")
        self.assertIn("sections_indexed", data)
        self.assertGreater(data.get("sections_indexed", 0), 0, "No sections indexed in service")

    def test_03_service_prewarmed(self):
        """Validates that the singleton service has an initialized graph and section index."""
        self.assertIsNotNone(self.service.rag, "LegalGraphRAG instance is None")
        self.assertGreater(len(self.service._section_index), 0, "Section index is empty")


if __name__ == "__main__":
    unittest.main()
