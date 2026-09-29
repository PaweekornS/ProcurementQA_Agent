# -*- coding: utf-8 -*-
"""
tests/test_langgraph_agent.py

Unit and Integration tests for LangGraph-based Agentic Legal GraphRAG:
1. Verify GroundingGuardrail (0-LLM Fact Checking & Anti-Hallucination).
2. Verify Tool Definitions (exact lookup, hybrid search, graph traversal, threshold).
3. Verify Agentic Graph execution and State propagation.
"""

import unittest
import json
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from core.agent.guardrail import GroundingGuardrail
from core.agent.state import LegalAgentState
from core.agent.tools import verify_procurement_threshold
from core.agent.graph import AgenticLegalGraphRAG


class TestLangGraphAgent(unittest.TestCase):

    def test_01_guardrail_valid_citations(self):
        """Guardrail should pass when synthesized text cites sections present in context."""
        retrieved_context = [
            {"entry": "พระราชบัญญัติการจัดซื้อจัดจ้างฯ พ.ศ. ๒๕๖๐ | มาตรา ๕๖ (๒) (ข)", "section_num": 56, "content_thai": "การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจง ให้กระทำได้..."},
            {"entry": "ระเบียบกระทรวงการคลังฯ พ.ศ. ๒๕๖๐ | ข้อ ๗๙", "clause_num": 79, "content_thai": "การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจงตามมาตรา ๕๖ (๒) (ข)..."}
        ]
        synthesized_text = "การจัดหาพัสดุในกรณีนี้ สามารถดำเนินการโดยวิธีเฉพาะเจาะจงได้ตาม พระราชบัญญัติฯ มาตรา ๕๖ (๒) (ข) ประกอบ ระเบียบกระทรวงการคลัง ข้อ ๗๙ เนื่องจากวงเงินไม่เกิน 500,000 บาท"

        verdict = GroundingGuardrail.audit(synthesized_text, retrieved_context)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["grounding_score"], 1.0)
        self.assertEqual(len(verdict["ungrounded_sections"]), 0)
        self.assertIn("56", verdict["cited_sections"])

    def test_02_guardrail_intercept_hallucination(self):
        """Guardrail should detect and flag citations not present in retrieved context."""
        retrieved_context = [
            {"entry": "มาตรา 56", "section_num": 56, "content_thai": "..."}
        ]
        # Text hallucinates Section 112 and Section 999
        synthesized_text = "ตามมาตรา 112 และ มาตรา 999 สามารถสั่งจ้างได้โดยไม่ต้องขออนุมัติ"

        verdict = GroundingGuardrail.audit(synthesized_text, retrieved_context)
        self.assertFalse(verdict["passed"])
        self.assertIn("112", verdict["ungrounded_sections"])
        self.assertIn("999", verdict["ungrounded_sections"])
        self.assertGreater(len(verdict["warnings"]), 0)

    def test_03_guardrail_threshold_ceiling_warning(self):
        """Guardrail should warn if specific selection is recommended above 500,000 THB without exception."""
        retrieved_context = [
            {"entry": "มาตรา 56", "section_num": 56, "content_thai": "..."}
        ]
        synthesized_text = "โครงการนี้มีวงเงิน 1,500,000 บาท สามารถเลือกใช้วิธีเฉพาะเจาะจงได้ทันที"

        verdict = GroundingGuardrail.audit(synthesized_text, retrieved_context)
        # Should flag warning regarding 500,000 ceiling
        has_ceiling_warning = any("500,000" in w for w in verdict["warnings"])
        self.assertTrue(has_ceiling_warning)

    def test_04_verify_threshold_tool(self):
        """verify_procurement_threshold tool must return valid audit JSON."""
        res_str = verify_procurement_threshold.invoke({
            "procurement_item": "คอมพิวเตอร์พกพา",
            "estimated_budget": 450000.0,
            "proposed_method": "เฉพาะเจาะจง"
        })
        res = json.loads(res_str)
        self.assertTrue(res["is_compliant"])
        self.assertEqual(res["compliance_status"], "PASSED")

    def test_05_agent_orchestration_flow(self):
        """AgenticLegalGraphRAG should initialize workflow and compile without error."""
        mock_model = MagicMock()
        mock_model.bind_tools = MagicMock(return_value=mock_model)
        mock_model.invoke = MagicMock(return_value=AIMessage(content="สามารถใช้วิธีเฉพาะเจาะจงได้ตามมาตรา 56"))

        agent = AgenticLegalGraphRAG(model_client=mock_model)
        self.assertIsNotNone(agent.workflow)

        response = agent.invoke(query="วงเงิน 300,000 บาท ใช้วิธีเฉพาะเจาะจงได้ไหม", org_id="DGA")
        self.assertIn("status", response)
        self.assertEqual(response["organization_id"], "DGA")


if __name__ == "__main__":
    unittest.main()
