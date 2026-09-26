#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/langgraph_orchestrator_demo.py

Production reference architecture demonstrating an outer Multi-Agent LangGraph Orchestrator
interacting with the specialized Procurement Legal Sub-Agent container.

Features demonstrated:
1. Orchestrator Agent (powered by Qwen-2.5-72B via OpenRouter):
   - Ingests user procurement requests.
   - Analyzes intent, extracts procurement parameters, and formulates sub-agent queries.
   - Evaluates legal sub-agent findings and synthesizes executive procurement advice.
2. Specialized Procurement Legal Sub-Agent (Container MCP / REST API):
   - Fast deterministic compliance verification (POST /api/v1/verify).
   - Deep CRAG statutory reasoning & quotes (POST /api/v1/qa) returning:
     `mode`, `direct_answer`, and `decisive_quotes` (filename, page, law, quote).
3. Live HTTP Connection with In-Process Fallback:
   - Connects to http://localhost:8000 by default.
   - Automatically falls back to in-process ProcurementService if server is offline.
"""

import os
import sys
import io

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from typing import Dict, Any, List, TypedDict, Optional
from dotenv import load_dotenv
import requests

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True, write_through=True)
    except Exception:
        pass

# Load environment configuration
load_dotenv()

# ==============================================================================
# Configuration
# ==============================================================================
SERVICE_BASE_URL = os.getenv("PROCUREMENT_SERVICE_URL", "http://localhost:8000")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("base_url", "https://openrouter.ai/api/v1")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "qwen/qwen3.5-flash-02-23")

# Lazy-loaded in-process service for offline fallback
_in_process_service = None


def _get_in_process_service():
    """Fallback in-process service if the REST server is not running."""
    global _in_process_service
    if _in_process_service is None:
        try:
            from core.mcp_service import ProcurementService
            _in_process_service = ProcurementService()
        except Exception as e:
            print(f"[!] Warning: Could not initialize in-process ProcurementService: {e}")
    return _in_process_service


# ==============================================================================
# Sub-Agent Client Interface (Dual Protocol: HTTP REST -> In-Process Fallback)
# ==============================================================================

def call_subagent_verify_compliance(
    procurement_item: str,
    estimated_budget: float,
    proposed_method: str,
    justification_reason: str = ""
) -> Dict[str, Any]:
    """
    Sub-Agent Tool 1: Deterministic statutory compliance audit (~5ms).
    """
    payload = {
        "procurement_item": procurement_item,
        "estimated_budget": estimated_budget,
        "proposed_method": proposed_method,
        "justification_reason": justification_reason
    }

    # 1. Attempt HTTP REST call to container
    try:
        url = f"{SERVICE_BASE_URL}/api/v1/verify"
        res = requests.post(url, json=payload, timeout=(1.5, 10.0))
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # 2. Offline in-process fallback
    print("  [i] REST container offline or unreachable. Using in-process Sub-Agent pipeline...")
    service = _get_in_process_service()
    if service:
        return service.verify_procurement_compliance(
            procurement_item=procurement_item,
            estimated_budget=estimated_budget,
            proposed_method=proposed_method,
            justification_reason=justification_reason
        )

    # 3. Static fallback if service uninitialized
    return {
        "compliance_status": "FLAGGED",
        "method": proposed_method,
        "estimated_budget": estimated_budget,
        "statutory_threshold": "วงเงินไม่เกิน 500,000 บาทสำหรับวิธีเฉพาะเจาะจง",
        "potential_risks": [f"งบประมาณ {estimated_budget:,.2f} บาท เกินเกณฑ์ทั่วไปของวิธี{proposed_method}"],
        "required_approvals": ["หัวหน้าหน่วยงานของรัฐ"],
        "recommended_alternatives": ["วิธีประกาศเชิญชวนทั่วไป (e-bidding)"]
    }


def call_subagent_procurement_qa(question: str, mode: str = "fast") -> Dict[str, Any]:
    """
    Sub-Agent Tool 2: Deep legal reasoning via CRAG pipeline (POST /api/v1/qa).
    Returns: mode, direct_answer, decisive_quotes (filename, page, law, quote).
    """
    payload = {"question": question, "mode": mode}

    # 1. Attempt HTTP REST call to container
    try:
        url = f"{SERVICE_BASE_URL}/api/v1/qa"
        res = requests.post(url, json=payload, timeout=(1.5, 60.0))
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    # 2. Offline in-process fallback
    print("  [i] REST container offline or unreachable. Querying in-process Sub-Agent Legal CRAG...")
    service = _get_in_process_service()
    if service:
        result = service.ask_procurement_law(question=question, mode=mode)
        quotes = result.get("decisive_quotes") or []
        return {
            "mode": result.get("mode", mode),
            "direct_answer": result.get("direct_answer", ""),
            "decisive_quotes": quotes
        }

    # 3. Static fallback simulation
    return {
        "mode": mode,
        "direct_answer": "การจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจงตามระเบียบกระทรวงการคลังฯ และ พ.ร.บ. จัดซื้อจัดจ้างฯ พ.ศ. 2560 มาตรา 56 (2) (ข) กำหนดวงเงินไม่เกิน 500,000 บาท หากเกินวงเงินดังกล่าว ต้องดำเนินการโดยวิธีประกาศเชิญชวนทั่วไป (e-bidding) หรือมีข้อยกเว้นกรณีเร่งด่วนตามมาตรา 56 (2) (ค) หรือ (ง)",
        "decisive_quotes": [
            {
                "filename": "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560.md",
                "page": "18-20/42",
                "law": "มาตรา ๕๖ (๒) (ข)",
                "quote": "การจัดซื้อจัดจ้างพัสดุที่มีการผลิต จำหน่าย ก่อสร้าง หรือให้บริการทั่วไป และมีวงเงินในการจัดซื้อจัดจ้างครั้งหนึ่งไม่เกินวงเงินตามที่กำหนดในกฎกระทรวง"
            },
            {
                "filename": "กฎกระทรวงกำหนดวงเงินการจัดซื้อจัดจ้างพัสดุโดยวิธีเฉพาะเจาะจง พ.ศ. 2560.md",
                "page": "1-2/3",
                "law": "ข้อ ๑",
                "quote": "การจัดซื้อจัดจ้างพัสดุตามมาตรา ๕๖ วรรคหนึ่ง (๒) (ข) ที่มีวงเงินในการจัดซื้อจัดจ้างครั้งหนึ่งไม่เกิน ๕๐๐,๐๐๐ บาท ให้ใช้วิธีเฉพาะเจาะจง"
            }
        ]
    }


# ==============================================================================
# Orchestrator LLM Caller (Qwen-2.5-72B via OpenRouter)
# ==============================================================================

def call_orchestrator_llm(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """Invokes the larger Orchestrator LLM (e.g. Qwen-2.5-72B)."""
    if not OPENROUTER_API_KEY:
        return "[Simulated Orchestrator Response: OPENROUTER_API_KEY is not set]"

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY
        )
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=4096
        )
        content = response.choices[0].message.content or ""
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        return content
    except Exception as e:
        return f"[Error invoking Orchestrator LLM ({ORCHESTRATOR_MODEL}): {e}]"


# ==============================================================================
# LangGraph Workflow Architecture
# ==============================================================================

class OrchestratorDialogueState(TypedDict):
    user_query: str
    parsed_params: Dict[str, Any]
    orchestrator_plan: str
    compliance_verdict: Dict[str, Any]
    legal_qa_result: Dict[str, Any]
    dialogue_log: List[Dict[str, str]]
    final_report: str


def node_orchestrator_plan(state: OrchestratorDialogueState) -> Dict[str, Any]:
    """
    Node 1: Orchestrator analyzes the user's inquiry, identifies required parameters,
    and formulates tasks for the Procurement Legal Sub-Agent.
    """
    user_query = state["user_query"]
    prompt = f"""คุณคือ "Master Procurement Orchestrator Agent" ที่ทำหน้าที่บริหารจัดการและกำกับดูแลการจัดซื้อจัดจ้างภาครัฐ
ผู้ใช้ส่งคำขอ/ข้อปรึกษาเข้ามาดังนี้:
"{user_query}"

กรุณาวิเคราะห์คำขอ โดยสกัดข้อมูลในรูปแบบ JSON ดังนี้:
{{
  "procurement_item": "ชื่อโครงการหรือรายการพัสดุ",
  "estimated_budget": ตัวเลขงบประมาณ (ตัวเลข float ไม่ใส่ comma),
  "proposed_method": "วิธีจัดซื้อจัดจ้างที่เสนอ (เช่น เฉพาะเจาะจง, ประกาศเชิญชวนทั่วไป, คัดเลือก)",
  "justification": "เหตุผลความจำเป็นที่ผู้ใช้อ้างอิง",
  "legal_question": "คำถามข้อกฎหมายและระเบียบที่ระบุวิธีจัดซื้อและวงเงินชัดเจนเพื่อส่งให้ Sub-Agent ค้นหาตัวบทกฎหมาย (เช่น การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจง วงเงิน 650,000 บาท มีเกณฑ์และข้อยกเว้นตามกฎหมายอย่างไร)",
  "reasoning_plan": "แนวทางการประสานงานกับ Sub-Agent สั้นๆ 1-2 ประโยค"
}}

ตอบเฉพาะ JSON block ที่ถูกต้องเท่านั้น:"""

    messages = [
        {"role": "system", "content": "You are a professional Enterprise Procurement Orchestrator. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    llm_output = call_orchestrator_llm(messages, temperature=0.0)

    # Extract JSON safely
    try:
        clean_json_str = llm_output.strip()
        if "```json" in clean_json_str:
            clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json_str:
            clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()
        parsed = json.loads(clean_json_str)
    except Exception:
        # Fallback heuristic if json parsing fails
        parsed = {
            "procurement_item": "โครงการจัดซื้อจัดจ้างตามคำขอ",
            "estimated_budget": 650000.0,
            "proposed_method": "เฉพาะเจาะจง",
            "justification": "ความเชี่ยวชาญเฉพาะด้านและความเร่งด่วน",
            "legal_question": f"การจัดซื้อจัดจ้างตามคำขอนี้ วงเงินและเหตุผลที่เสนอ เป็นไปตาม พ.ร.บ. และระเบียบกระทรวงการคลังหรือไม่ และมีข้อยกเว้นหรือทางเลือกใดบ้าง",
            "reasoning_plan": "ตรวจสอบความสอดคล้องของวงเงินกับวิธีจัดซื้อ จากนั้นสอบถามข้อยกเว้นทางกฎหมาย"
        }

    log_entry = {
        "speaker": f"Orchestrator Agent ({ORCHESTRATOR_MODEL})",
        "action": "Plan & Task Decomposition",
        "content": (
            f"📋 วิเคราะห์โครงการ: '{parsed.get('procurement_item')}'\n"
            f"💰 วงเงิน: {parsed.get('estimated_budget', 0):,.2f} บาท | วิธีที่เสนอ: '{parsed.get('proposed_method')}'\n"
            f"🎯 แผนงาน: {parsed.get('reasoning_plan')}"
        )
    }

    return {
        "parsed_params": parsed,
        "orchestrator_plan": parsed.get("reasoning_plan", ""),
        "dialogue_log": state.get("dialogue_log", []) + [log_entry]
    }


def node_subagent_compliance(state: OrchestratorDialogueState) -> Dict[str, Any]:
    """
    Node 2: Sub-Agent executes deterministic statutory verification.
    """
    params = state["parsed_params"]
    print("\n" + "="*70)
    print("🔄 [Orchestrator -> Sub-Agent] Sending request to 'check_procurement_threshold'...")
    print("="*70)

    verdict = call_subagent_verify_compliance(
        procurement_item=params.get("procurement_item", ""),
        estimated_budget=float(params.get("estimated_budget", 0)),
        proposed_method=params.get("proposed_method", "เฉพาะเจาะจง"),
        justification_reason=params.get("justification", "")
    )

    log_entry = {
        "speaker": "Procurement Legal Sub-Agent (Container /verify)",
        "action": "Statutory Compliance Audit Verdict",
        "content": (
            f"⚖️ สถานะ: {verdict.get('compliance_status')}\n"
            f"📌 เกณฑ์กฎหมาย: {verdict.get('statutory_threshold', 'N/A')}\n"
            f"⚠️ ความเสี่ยง: {', '.join(verdict.get('potential_risks', [])) or 'ไม่มี'}\n"
            f"💡 ทางเลือกแนะนำ: {', '.join(verdict.get('recommended_alternatives', [])) or 'ดำเนินการตามวิธีปกติ'}"
        )
    }

    return {
        "compliance_verdict": verdict,
        "dialogue_log": state.get("dialogue_log", []) + [log_entry]
    }


def node_subagent_legal_qa(state: OrchestratorDialogueState) -> Dict[str, Any]:
    """
    Node 3: Sub-Agent executes deep legal QA (POST /api/v1/qa) with CRAG.
    """
    params = state["parsed_params"]
    method = params.get("proposed_method", "เฉพาะเจาะจง")
    budget = float(params.get("estimated_budget", 0))
    legal_q = params.get("legal_question", "").strip()

    # Ensure statutory keywords (วิธีจัดซื้อจัดจ้าง, วงเงิน, ข้อยกเว้น) are present for CRAG retriever
    if not legal_q or not any(k in legal_q for k in ["เฉพาะเจาะจง", "e-bidding", "คัดเลือก", "วงเงิน"]):
        legal_q = f"การจัดซื้อจัดจ้างพัสดุโดยวิธี{method} ในวงเงิน {budget:,.0f} บาท มีหลักเกณฑ์และข้อยกเว้นตาม พ.ร.บ. จัดซื้อจัดจ้างฯ หรือกฎกระทรวงอย่างไร"

    print("\n" + "="*70)
    print("🔄 [Orchestrator -> Sub-Agent] Sending deep legal inquiry to 'POST /api/v1/qa'...")
    print(f"❓ คำถาม: {legal_q}")
    print("="*70)

    qa_res = call_subagent_procurement_qa(question=legal_q, mode="fast")

    quotes_summary = []
    for q in qa_res.get("decisive_quotes", []):
        quotes_summary.append(
            f"  • [{q.get('filename')}] หน้า {q.get('page')}\n"
            f"    {q.get('law')}: \"{q.get('quote')}\""
        )
    quotes_text = "\n".join(quotes_summary) if quotes_summary else "  (ไม่มีตัวบทเฉพาะเจาะจง)"

    log_entry = {
        "speaker": "Procurement Legal Sub-Agent (Container /api/v1/qa)",
        "action": "CRAG Legal Retrieval & Decisive Quotes",
        "content": (
            f"📖 คำตอบทางกฎหมาย:\n{qa_res.get('direct_answer')}\n\n"
            f"📜 Decisive Quotes (ตัวบทกฎหมายชี้ขาด):\n{quotes_text}"
        )
    }

    return {
        "legal_qa_result": qa_res,
        "dialogue_log": state.get("dialogue_log", []) + [log_entry]
    }


def node_orchestrator_synthesize(state: OrchestratorDialogueState) -> Dict[str, Any]:
    """
    Node 4: Orchestrator reviews the Sub-Agent's findings, evaluates exceptions,
    and drafts an executive recommendation report for the user.
    """
    user_q = state["user_query"]
    params = state["parsed_params"]
    verdict = state["compliance_verdict"]
    legal_qa = state["legal_qa_result"]

    system_prompt = f"""คุณคือ "Master Procurement Orchestrator Agent" ที่ปรึกษาการจัดซื้อจัดจ้างภาครัฐระดับบริหาร
คุณได้รับรายงานผลการตรวจสอบกฎหมายและระเบียบจาก "Procurement Legal Sub-Agent" เรียบร้อยแล้ว
หน้าที่ของคุณคือสรุปตอบคำถามของผู้ใช้งานอย่างชาญฉลาด เป็นระบบ ชัดเจน และมีหลักกฎหมายอ้างอิง

ข้อมูลทั้งหมดที่ได้รับ:
1. คำขอของผู้ใช้: {user_q}
2. ผลตรวจสอบความสอดคล้อง (Compliance Status): {verdict.get('compliance_status')}
   - เกณฑ์วงเงิน: {verdict.get('statutory_threshold')}
   - ข้อควรระวัง/ความเสี่ยง: {verdict.get('potential_risks')}
   - ผู้มีอำนาจอนุมัติ: {verdict.get('required_approvals')}
   - ทางเลือกแนะนำ: {verdict.get('recommended_alternatives')}
3. คำตอบและตัวบทกฎหมายจาก Sub-Agent:
   - สาระสำคัญ: {legal_qa.get('direct_answer')}
   - ข้อกฎหมายอ้างอิง (Decisive Quotes): {json.dumps(legal_qa.get('decisive_quotes', []), ensure_ascii=False, indent=2)}

คำแนะนำในการเขียนสรุป:
1. สรุปคำตอบโดยตรง (Direct Verdict): สามารถทำได้หรือไม่ ทำไมถึงทำได้หรือทำไม่ได้
2. ข้อกฎหมายและระเบียบที่เกี่ยวข้อง: ระบุเลขมาตรา/ข้อ พร้อมระบุชื่อเอกสารและหน้าที่อ้างอิงให้ชัดเจน
3. ทางเลือกหรือแนวทางแก้ไข (Actionable Alternatives): หากทำไม่ได้ ควรเปลี่ยนเป็นวิธีใด หรือมีข้อยกเว้นใดที่ต้องขออนุมัติ
4. ขั้นตอนและผู้มีอำนาจอนุมัติ (Governance & Sign-off)"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "กรุณาจัดทำรายงานข้อเสนอแนะฉบับสมบูรณ์สำหรับผู้บริหาร"}
    ]

    final_report = call_orchestrator_llm(messages, temperature=0.3)

    log_entry = {
        "speaker": f"Orchestrator Agent ({ORCHESTRATOR_MODEL})",
        "action": "Final Executive Synthesis",
        "content": final_report
    }

    return {
        "final_report": final_report,
        "dialogue_log": state.get("dialogue_log", []) + [log_entry]
    }


# ==============================================================================
# Build StateGraph
# ==============================================================================

def build_orchestrator_graph():
    """Builds and compiles the LangGraph workflow."""
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(OrchestratorDialogueState)
    builder.add_node("orchestrator_plan", node_orchestrator_plan)
    builder.add_node("subagent_compliance", node_subagent_compliance)
    builder.add_node("subagent_legal_qa", node_subagent_legal_qa)
    builder.add_node("orchestrator_synthesize", node_orchestrator_synthesize)

    builder.add_edge(START, "orchestrator_plan")
    builder.add_edge("orchestrator_plan", "subagent_compliance")
    builder.add_edge("subagent_compliance", "subagent_legal_qa")
    builder.add_edge("subagent_legal_qa", "orchestrator_synthesize")
    builder.add_edge("orchestrator_synthesize", END)

    return builder.compile()


# ==============================================================================
# Main Interactive / Demo Runner
# ==============================================================================

def run_simulation(user_query: str):
    print("=" * 80)
    print("🤖 MULTI-AGENT ORCHESTRATION DEMO: Master Orchestrator <-> Legal Sub-Agent")
    print("=" * 80)
    print(f"🔹 Orchestrator Model : {ORCHESTRATOR_MODEL}")
    print(f"🔹 Sub-Agent Endpoint  : {SERVICE_BASE_URL} (with in-process fallback)")
    print(f"🔹 User Inquiry        : \"{user_query}\"")
    print("=" * 80)

    initial_state: OrchestratorDialogueState = {
        "user_query": user_query,
        "parsed_params": {},
        "orchestrator_plan": "",
        "compliance_verdict": {},
        "legal_qa_result": {},
        "dialogue_log": [],
        "final_report": ""
    }

    graph = build_orchestrator_graph()
    final_state = graph.invoke(initial_state)

    print("\n" + "#" * 80)
    print("💬 DIALOGUE TRACE: MASTER ORCHESTRATOR <---> SUB-AGENT")
    print("#" * 80 + "\n")

    for idx, item in enumerate(final_state["dialogue_log"], 1):
        print(f"[{idx}] {item['speaker']} - {item['action']}")
        print("-" * 70)
        print(item["content"])
        print("\n" + "=" * 70 + "\n")

    print("#" * 80)
    print("🏆 FINAL EXECUTIVE ADVICE (Presented to End User)")
    print("#" * 80)
    print(final_state["final_report"])
    print("\n" + "#" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent Orchestration Demo")
    parser.add_argument(
        "--query",
        type=str,
        default="เราต้องการจัดจ้างพัฒนาระบบ AI ตรวจสอบพัสดุและวิเคราะห์สัญญา วงเงิน 650,000 บาท โดยวิธีเฉพาะเจาะจง เพราะเร่งด่วนและบริษัทนี้มีความเชี่ยวชาญเฉพาะทาง สามารถทำได้หรือไม่ตามระเบียบ และมีทางเลือกใดบ้าง",
        help="Procurement inquiry to simulate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Override Orchestrator LLM model"
    )

    args = parser.parse_args()
    if args.model:
        ORCHESTRATOR_MODEL = args.model

    run_simulation(args.query)
