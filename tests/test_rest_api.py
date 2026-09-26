#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_rest_api.py

Smoke-tests the dual-protocol REST API endpoints of mcp_server.py:
  - GET  /healthz
  - GET  /ready
  - POST /api/v1/verify
  - POST /api/v1/search
  - POST /api/v1/qa

Usage:
    python scripts/test_rest_api.py
    python scripts/test_rest_api.py --host http://localhost:8000
    python scripts/test_rest_api.py --run-qa
"""

import argparse
import json
import requests


def main(base_url: str, run_qa: bool):
    print(f"=== Smoke-Testing REST API on {base_url} ===\n")

    # 1. Healthz
    print("[1] GET /healthz")
    try:
        r = requests.get(f"{base_url}/healthz", timeout=5)
        print(f"Status: {r.status_code} -> {r.json()}")
    except Exception as e:
        print(f"Failed: {e}")

    # 2. Ready
    print("\n[2] GET /ready")
    try:
        r = requests.get(f"{base_url}/ready", timeout=10)
        print(f"Status: {r.status_code} -> {r.json()}")
    except Exception as e:
        print(f"Failed: {e}")

    # 3. Verify Compliance
    print("\n[3] POST /api/v1/verify")
    payload_verify = {
        "procurement_item": "จัดจ้างพัฒนาซอฟต์แวร์สารสนเทศ",
        "estimated_budget": 350000.0,
        "proposed_method": "เฉพาะเจาะจง",
        "justification_reason": "วงเงินไม่เกิน 500,000 บาท"
    }
    try:
        r = requests.post(f"{base_url}/api/v1/verify", json=payload_verify, timeout=5)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Failed: {e}")

    # 4. Search Clauses
    print("\n[4] POST /api/v1/search")
    payload_search = {
        "query": "วิธีเฉพาะเจาะจง วงเงินไม่เกิน 500,000",
        "top_k": 2
    }
    try:
        r = requests.post(f"{base_url}/api/v1/search", json=payload_search, timeout=60)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Failed: {e}")

    # 5. Full Q&A
    if run_qa:
        print("\n[5] POST /api/v1/qa")
        payload_qa = {
            "question": "หน่วยงานของรัฐจะจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงได้ไม่เกินวงเงินเท่าใด",
            "mode": "fast"
        }
        try:
            r = requests.post(f"{base_url}/api/v1/qa", json=payload_qa, timeout=60)
            print(f"Status: {r.status_code}")
            print(json.dumps(r.json(), ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test REST API endpoints")
    parser.add_argument("--host", default="http://localhost:8000", help="Base URL of server")
    parser.add_argument("--run-qa", action="store_true", help="Include generative QA test")
    args = parser.parse_args()

    main(args.host, args.run_qa)
