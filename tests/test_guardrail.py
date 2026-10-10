# -*- coding: utf-8 -*-
"""
tests/test_guardrail.py

Unit tests for GroundingGuardrail (0-LLM fact checking & anti-hallucination), the final node of
the agentic RAG workflow.
"""

import unittest

from core.agent.guardrail import GroundingGuardrail


class TestGroundingGuardrail(unittest.TestCase):

    def test_01_guardrail_valid_citations(self):
        """Guardrail should pass when synthesized text cites sections present in context."""
        retrieved_context = [
            {"entry": "พระราชบัญญัติการจัดซื้อจัดจ้างฯ พ.ศ. ๒๕๖๐ | มาตรา ๕๖ (๒) (ข)", "section_num": 56, "content": "การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจง ให้กระทำได้..."},
            {"entry": "ระเบียบกระทรวงการคลังฯ พ.ศ. ๒๕๖๐ | ข้อ ๗๙", "clause_num": 79, "content": "การจัดซื้อจัดจ้างโดยวิธีเฉพาะเจาะจงตามมาตรา ๕๖ (๒) (ข)..."}
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
            {"entry": "มาตรา 56", "section_num": 56, "content": "..."}
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
            {"entry": "มาตรา 56", "section_num": 56, "content": "..."}
        ]
        synthesized_text = "โครงการนี้มีวงเงิน 1,500,000 บาท สามารถเลือกใช้วิธีเฉพาะเจาะจงได้ทันที"

        verdict = GroundingGuardrail.audit(synthesized_text, retrieved_context)
        # Should flag warning regarding 500,000 ceiling
        has_ceiling_warning = any("500,000" in w for w in verdict["warnings"])
        self.assertTrue(has_ceiling_warning)


if __name__ == "__main__":
    unittest.main()
