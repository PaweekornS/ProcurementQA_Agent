# -*- coding: utf-8 -*-
"""
core/agent/graph.py

LangGraph-based Agentic Legal GraphRAG Workflow.
Implements the Autonomous Orchestrator Agent (ReAct) with 0-LLM Grounding Guardrail.

Flow:
  START -> orchestrator -> [tools_condition]
                              ├─ (has tool_calls) -> tools -> orchestrator (loop)
                              └─ (synthesis ready) -> guardrail -> END
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from core.agent.state import LegalAgentState
from core.agent.tools import LEGAL_TOOLS
from core.agent.guardrail import GroundingGuardrail

logger = logging.getLogger("agentic_legal_graphrag")

SYSTEM_ORCHESTRATOR_PROMPT = """คุณคือที่ปรึกษากฎหมายและผู้เชี่ยวชาญด้านพระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560 และระเบียบกระทรวงการคลังฯ ประจำสำนักงานพัฒนารัฐบาลดิจิทัล (DGA)

หน้าที่ของคุณ:
1. วิเคราะห์ข้อหารือของผู้ใช้ วางแผน และเลือกใช้เครื่องมือ (Tools) ที่มีอยู่อย่างเหมาะสม:
   - หากผู้ใช้ถามถึง 'มาตรา' หรือ 'ข้อ' เฉพาะเจาะจง -> ใช้ `exact_statute_lookup`
   - หากผู้ใช้ถามเชิงเนื้อหาหรือไม่มีเลขมาตราแน่ชัด -> ใช้ `hybrid_statutory_search`
   - หากพบมาตราหลักแล้วต้องการดูระเบียบกระทรวง/หนังสือเวียนที่รองรับ -> ใช้ `knowledge_graph_traversal`
   - หากผู้ใช้ถามเรื่องวงเงินงบประมาณและวิธีจัดซื้อจัดจ้าง -> ใช้ `verify_procurement_threshold` ตรวจสอบทันที
2. เมื่อได้ข้อมูลพยานหลักฐานเพียงพอ ให้สังเคราะห์คำตอบที่มีโครงสร้างชัดเจน:
   - **คำตอบชัดเจนโดยตรง (Direct Answer):** สรุปว่าทำได้/ไม่ได้ ภายใต้เงื่อนไขใด
   - **ฐานข้อกฎหมายและระเบียบที่เกี่ยวข้อง (Applicable Laws):** ระบุชื่อกฎหมายและมาตรา/ข้ออย่างชัดเจน
   - **ข้อกำหนดและข้อยกเว้น (Exceptions/Conditions):** ระบุเงื่อนไขพิเศษหรือข้อควรระวัง (เช่น การห้ามแบ่งซื้อแบ่งจ้าง)
3. สำคัญ: ห้ามแต่งมาตราหรือกุข้อกฎหมายขึ้นมาเองเด็ดขาด อ้างอิงเฉพาะข้อมูลที่ได้จาก Tools เท่านั้น
"""


class AgenticLegalGraphRAG:
    """Compiled LangGraph Agentic Workflow for Thai Procurement Legal QA."""

    def __init__(self, model_client=None):
        if model_client is not None and hasattr(model_client, "bind_tools"):
            self.model_client = model_client
        else:
            from langchain_openai import ChatOpenAI
            api_key = getattr(model_client, "api_key", None) or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = getattr(model_client, "base_url", None) or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            model_name = getattr(model_client, "model_name", None) or os.getenv("LLM_MODEL", "qwen/qwen3.5-9b")
            try:
                self.model_client = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.0
                )
            except Exception:
                self.model_client = model_client
        self.tools = LEGAL_TOOLS
        self.workflow = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(LegalAgentState)

        # 1. Register Nodes
        graph.add_node("orchestrator", self._orchestrator_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("guardrail", self._guardrail_node)

        # 2. Register Edges
        graph.add_edge(START, "orchestrator")
        graph.add_conditional_edges(
            "orchestrator",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "guardrail"
            }
        )
        graph.add_edge("tools", "orchestrator")
        graph.add_edge("guardrail", END)

        return graph.compile()

    def _orchestrator_node(self, state: LegalAgentState) -> Dict[str, Any]:
        """Orchestrator node: Uses LLM to plan, select tools, or synthesize answer."""
        messages = list(state.get("messages", []))
        
        # Ensure system message is present
        if not messages or not isinstance(messages[0], SystemMessage):
            system_prompt = SYSTEM_ORCHESTRATOR_PROMPT
            org_id = state.get("org_id", "DGA")
            if org_id != "PUBLIC":
                system_prompt += f"\nคุณกำลังให้คำปรึกษาแก่เจ้าหน้าที่หน่วยงาน: {org_id}"
            messages.insert(0, SystemMessage(content=system_prompt))

        # Check if model supports native tool binding (e.g. ChatOpenAI / OpenRouter)
        if hasattr(self.model_client, "bind_tools"):
            model_with_tools = self.model_client.bind_tools(self.tools)
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Fallback for models without native tool binding:
        # Prompt-based reasoning execution
        response = self._fallback_chat_invoke(messages, state)
        return {"messages": [response]}

    def _guardrail_node(self, state: LegalAgentState) -> Dict[str, Any]:
        """0-LLM Deterministic Guardrail Node."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        synthesized_text = last_message.content if last_message else ""

        # Extract contexts retrieved during the tool calls
        retrieved_contexts = state.get("retrieved_context", [])
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    data = json.loads(m.content)
                    if isinstance(data, list):
                        retrieved_contexts.extend(data)
                    elif isinstance(data, dict):
                        retrieved_contexts.append(data)
                except Exception:
                    pass

        verdict = GroundingGuardrail.audit(synthesized_text, retrieved_contexts)
        
        # Formulate applicable laws list from cited sections and retrieved context
        applicable_laws = [f"มาตรา {s}" for s in verdict.get("cited_sections", [])]
        for ctx in retrieved_contexts:
            entry = ctx.get("entry")
            if entry and entry not in applicable_laws:
                applicable_laws.append(entry)

        final_response = {
            "status": "SUCCESS" if verdict["passed"] else "FLAGGED",
            "direct_answer": synthesized_text,
            "decisive_quotes": [],
            "applicable_laws": applicable_laws,
            "grounding_score": verdict["grounding_score"],
            "cited_sections": verdict["cited_sections"],
            "guardrail_warnings": verdict["warnings"],
            "organization_id": state.get("org_id", "DGA"),
        }

        # If warnings exist, attach a brief professional legal disclaimer
        if verdict["warnings"]:
            final_response["disclaimer"] = "ข้อสังเกตเพิ่มเติม: " + " | ".join(verdict["warnings"])

        return {
            "guardrail_verdict": verdict,
            "final_response": final_response,
            "retrieved_context": retrieved_contexts
        }

    def _fallback_chat_invoke(self, messages: List[BaseMessage], state: LegalAgentState) -> AIMessage:
        """Invokes model when tool calling is simulated via direct completion."""
        # Simple single-pass synthesis if model does not bind tools directly
        prompt_text = ""
        for m in messages:
            prefix = "User: " if isinstance(m, HumanMessage) else "Assistant: " if isinstance(m, AIMessage) else "System: "
            prompt_text += f"{prefix}{m.content}\n"

        if hasattr(self.model_client, "generate_response"):
            resp_text = self.model_client.generate_response(prompt_text)
        elif hasattr(self.model_client, "invoke"):
            res = self.model_client.invoke(prompt_text)
            resp_text = getattr(res, "content", str(res))
        else:
            resp_text = str(messages[-1].content)

        return AIMessage(content=resp_text)

    def invoke(self, query: str, org_id: str = "DGA", mode: str = "deep") -> Dict[str, Any]:
        """Entrypoint for executing the LangGraph Agent."""
        initial_state: LegalAgentState = {
            "messages": [HumanMessage(content=query.strip())],
            "user_query": query.strip(),
            "org_id": org_id,
            "mode": mode,
            "retrieved_context": [],
            "citations_used": [],
            "guardrail_verdict": {},
            "final_response": {}
        }
        final_state = self.workflow.invoke(initial_state)
        resp = dict(final_state.get("final_response", {}))
        resp["retrieved_context"] = final_state.get("retrieved_context", [])
        return resp
