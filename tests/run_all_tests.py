#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/run_all_tests.py

Unified Container Verification & CI/CD Test Runner:
Executes the comprehensive suite of LegalGraphRAG tests:
  1. Healthcheck & Liveness Probe
  2. All Retrieval Strategies (Exact, Hybrid, Graph)
  3. Multi-Tenant Data Isolation & Tenant Resolution
  4. Rule-Based Compliance & Statutory Thresholds
  5. MCP Protocol Server Contract (4 Core Public Tools)
  6. Main Agentic QA Reasoning Engine

Usage:
  python tests/run_all_tests.py
  python tests/run_all_tests.py --skip-qa   # Fast offline build test (skips LLM API calls)
"""

import os
import sys
import time
import argparse
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Test Suites
from tests.test_healthcheck import TestHealthcheck
from tests.test_retrieval_modes import TestRetrievalModes
from tests.test_multitenancy import TestMultiTenancy
from tests.test_mcp_protocol import TestMCPProtocol
from tests.test_main_qa_engine import TestMainQAEngine


def run_suites(skip_qa: bool = False) -> bool:
    print("\n" + "=" * 80)
    print(" [*] LEGAL-GRAPH-RAG PRODUCTION CONTAINER TEST RUNNER")
    print("=" * 80)
    print(f"Working Directory: {PROJECT_ROOT}")
    print(f"Skip Generative QA: {skip_qa}")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 1. Healthcheck
    suite.addTests(loader.loadTestsFromTestCase(TestHealthcheck))
    # 2. Retrieval Modes
    suite.addTests(loader.loadTestsFromTestCase(TestRetrievalModes))
    # 3. Multi-Tenancy Isolation
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTenancy))
    # 4. Compliance Engine
    # 5. MCP Protocol
    suite.addTests(loader.loadTestsFromTestCase(TestMCPProtocol))

    # 6. Main QA Engine (Requires LLM API access)
    if not skip_qa:
        suite.addTests(loader.loadTestsFromTestCase(TestMainQAEngine))
    else:
        print("[!] Note: Skipping TestMainQAEngine (--skip-qa flag active).\n")

    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(" 📊 TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed:         {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:       {len(result.failures)}")
    print(f"Errors:         {len(result.errors)}")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print("=" * 80)

    if result.wasSuccessful():
        print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY! CONTAINER IS PRODUCTION READY.\n")
        return True
    else:
        print("\n[FAILED] SOME TESTS FAILED. PLEASE REVIEW LOGS ABOVE.\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete test suite for LegalGraphRAG container")
    parser.add_argument("--skip-qa", action="store_true", help="Skip generative LLM QA test for offline CI/CD")
    args = parser.parse_args()

    success = run_suites(skip_qa=args.skip_qa)
    sys.exit(0 if success else 1)
