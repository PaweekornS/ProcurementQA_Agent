# -*- coding: utf-8 -*-
"""
scratch/test_multi_tenant_qa_pipeline.py

Verification script for Multi-Tenant Pipeline Alignment:
1. Validates that FastAPI /api/v1/qa works without 'mode' parameter.
2. Validates that org_id correctly propagates through AgenticRAGState down to retrieval.
3. Asserts that returned response schema matches the clean contract with organization_id.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.app import app

def test_api_qa_multi_tenant():
    print("\n" + "=" * 70)
    print(" [TEST] Multi-Tenant QA Pipeline & API Verification")
    print("=" * 70)

    client = TestClient(app)

    # 1. Test /api/v1/qa without 'mode' parameter
    payload = {
        "question": "หน่วยงานสามารถจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจงในวงเงินไม่เกินเท่าใด",
        "org_id": "ORG_05_MOPH"
    }
    print(f"\n[*] Sending request to /api/v1/qa with org_id='{payload['org_id']}'...")
    response = client.post("/api/v1/qa", json=payload)

    print(f"[*] Response Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"[-] Response Error Body: {response.text}")
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()
    print(f"[+] Direct Answer Preview: {data.get('direct_answer')[:120]}...")
    print(f"[+] Applicable Laws: {data.get('applicable_laws')}")
    print(f"[+] Organization ID in Response: {data.get('organization_id')}")

    # Assertions
    assert "mode" not in data, "Regression: 'mode' should not be present in LegalQAResponse!"
    assert data.get("organization_id") == "ORG_05_MOPH", f"Expected org_id 'ORG_05_MOPH', got {data.get('organization_id')}"
    assert data.get("status") in ("COMPLIANT", "OUT_OF_SCOPE", "NO_LAW_FOUND", "NON_COMPLIANT"), f"Unexpected status: {data.get('status')}"
    assert len(data.get("applicable_laws", [])) > 0 or len(data.get("decisive_quotes", [])) > 0, "Expected at least one cited law or quote"

    print("\n[SUCCESS] Multi-Tenant QA Pipeline & Clean API Verified Successfully!")


if __name__ == "__main__":
    test_api_qa_multi_tenant()
