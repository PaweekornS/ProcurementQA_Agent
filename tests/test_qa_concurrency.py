# -*- coding: utf-8 -*-
"""
tests/test_qa_concurrency.py

QA admission control (no LLM calls): when every QA slot is taken, ask_procurement_law raises
ServiceBusyError after QA_QUEUE_TIMEOUT_SECONDS and REST /api/v1/qa answers 503 + Retry-After.
"""

import os
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["QA_QUEUE_TIMEOUT_SECONDS"] = "0.5"

from fastapi.testclient import TestClient

import core.service as service_module
from api.app import app
from api.dependencies import get_service


class TestQAConcurrency(unittest.TestCase):
    def setUp(self):
        # A real service object; its workflow is never reached because no slot is free
        self.service = service_module.ProcurementService.__new__(service_module.ProcurementService)
        self.held = 0
        while service_module._QA_SLOTS.acquire(blocking=False):
            self.held += 1

    def tearDown(self):
        for _ in range(self.held):
            service_module._QA_SLOTS.release()
        app.dependency_overrides.clear()

    def test_01_service_raises_busy_after_queue_timeout(self):
        started = time.monotonic()
        with self.assertRaises(service_module.ServiceBusyError):
            self.service.ask_procurement_law("มาตรา 56 คืออะไร", org_id="DGA")
        self.assertGreaterEqual(time.monotonic() - started, 0.4)

    def test_02_rest_returns_503_with_retry_after(self):
        app.dependency_overrides[get_service] = lambda: self.service
        response = TestClient(app).post("/api/v1/qa", json={"query": "มาตรา 56 คืออะไร"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Retry-After", response.headers)


if __name__ == "__main__":
    unittest.main()
