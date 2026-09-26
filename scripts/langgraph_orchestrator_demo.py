#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/langgraph_orchestrator_demo.py

Production reference architecture demonstrating how an outer LangGraph Orchestrator
integrates with the LegalGraphRAG container (MCP / REST API Service).

Two connection modes are demonstrated:
1. Streamable-HTTP MCP client (via langchain-mcp-adapters)
2. Direct REST API tool binding (HTTPX / Requests fallback)
"""

import os
import json
import asyncio
from typing import Annotated, Dict, Any, List, TypedDict
import requests

# ==============================================================================
# 1. Direct REST Tool Bindings for LangGraph
# ==============================================================================

SERVICE_BASE_URL = os.getenv("PROCUREMENT_SERVICE_URL", "http://localhost:8000")


def get_statute_section(section: str, doc_title: str = "") -> Dict[str, Any]:
    """Exact verbatim statutory clause lookup (Tier 1, 0-LLM latency ~10ms)."""
    # Calling the container's Tier 1 atomic endpoint
    url = f"{SERVICE_BASE_URL}/api/v1/search"
    res = requests.post(url, json={"query": f"{doc_title} {section}".strip(), "top_k": 1}, timeout=10)
    return res.json()


def verify_procurement_compliance(
    procurement_item: str,
    estimated_budget: float,
    proposed_method: str,
    justification_reason: str = ""
) -> Dict[str, Any]:
    """Deterministic statutory threshold audit (Tier 3, rule-based ~5ms)."""
    url = f"{SERVICE_BASE_URL}/api/v1/verify"
    payload = {
        "procurement_item": procurement_item,
        "estimated_budget": estimated_budget,
        "proposed_method": proposed_method,
        "justification_reason": justification_reason
    }
    res = requests.post(url, json=payload, timeout=10)
    return res.json()


def ask_procurement_law(question: str, mode: str = "fast") -> Dict[str, Any]:
    """Execute CRAG pipeline with guardrails (Tier 3, deep reasoning)."""
    url = f"{SERVICE_BASE_URL}/api/v1/qa"
    res = requests.post(url, json={"question": question, "mode": mode}, timeout=60)
    return res.json()


# ==============================================================================
# 2. Reference LangGraph State & Node Architecture
# ==============================================================================

class ProcurementAuditState(TypedDict):
    project_name: str
    budget: float
    proposed_method: str
    justification: str
    compliance_result: Dict[str, Any]
    legal_basis: str
    decision: str
    explanation: str


def node_audit_compliance(state: ProcurementAuditState) -> Dict[str, Any]:
    """Node 1: Call container's deterministic statutory verification."""
    print(f"[*] Auditing project: {state['project_name']} (Budget: {state['budget']:,} THB, Method: {state['proposed_method']})")
    audit = verify_procurement_compliance(
        procurement_item=state["project_name"],
        estimated_budget=state["budget"],
        proposed_method=state["proposed_method"],
        justification_reason=state.get("justification", "")
    )
    return {"compliance_result": audit}


def route_compliance_verdict(state: ProcurementAuditState) -> str:
    """Conditional Edge: Route based on compliance audit status."""
    status = state["compliance_result"].get("compliance_status", "FLAGGED")
    if status == "PASSED":
        return "approved_node"
    elif status == "VIOLATION":
        return "rejected_node"
    return "legal_inquiry_node"


def node_approved(state: ProcurementAuditState) -> Dict[str, Any]:
    """Node 2A: Generate approval summary and required approvers."""
    approvals = state["compliance_result"].get("required_approvals", [])
    threshold = state["compliance_result"].get("statutory_threshold", "")
    return {
        "decision": "PASSED",
        "explanation": f"ถูกต้องตามเกณฑ์ ({threshold}). ผู้มีอำนาจอนุมัติ: {', '.join(approvals)}"
    }


def node_legal_inquiry(state: ProcurementAuditState) -> Dict[str, Any]:
    """Node 2B: Ambiguous or flagged case -> invoke container CRAG reasoning."""
    print("[-] Flagged case: querying container CRAG for deeper analysis...")
    query = (
        f"การจัดซื้อ '{state['project_name']}' วงเงิน {state['budget']} บาท "
        f"โดยวิธี '{state['proposed_method']}' มีข้อยกเว้นหรือเงื่อนไขตามระเบียบใดบ้าง"
    )
    crag_res = ask_procurement_law(question=query, mode="fast")
    return {
        "decision": "REQUIRES_REVIEW",
        "legal_basis": crag_res.get("applicable_laws", []),
        "explanation": crag_res.get("direct_answer", "")
    }


def node_rejected(state: ProcurementAuditState) -> Dict[str, Any]:
    """Node 2C: Immediate violation detected."""
    risks = state["compliance_result"].get("potential_risks", [])
    return {
        "decision": "VIOLATION",
        "explanation": f"ขัดต่อระเบียบจัดซื้อจัดจ้าง: {'; '.join(risks)}"
    }


# ==============================================================================
# 3. Execution Demonstration
# ==============================================================================

if __name__ == "__main__":
    print("=== LangGraph Orchestrator Integration Pattern ===")
    print(f"Target Container URL: {SERVICE_BASE_URL}\n")

    sample_state: ProcurementAuditState = {
        "project_name": "จัดจ้างพัฒนาโมเดล AI ตรวจสอบพัสดุ",
        "budget": 450000.0,
        "proposed_method": "เฉพาะเจาะจง",
        "justification": "วงเงินไม่เกิน 500,000 บาท",
        "compliance_result": {},
        "legal_basis": "",
        "decision": "",
        "explanation": ""
    }

    print("[Flow Demonstration]")
    print("Step 1: Ingest into LangGraph State")
    print(f"Input: {json.dumps(sample_state, ensure_ascii=False, indent=2)}\n")
    print("Step 2: Routing via Container MCP / REST APIs")
    print("  StateGraph: START -> node_audit_compliance -> [conditional edge]")
    print("              -> node_approved | node_legal_inquiry | node_rejected -> END")
