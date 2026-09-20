"""Completeness & Grounding Auditor Agent for LegalGraphRAG CRAG"""
import json
import re
from typing import Dict, Any, List
from core.prompt import get_prompt


class CompletenessAuditor:
    """Agent 3: Audits whether all sub-issues are answered and supported by statutory context."""

    def __init__(self, model):
        self.model = model

    def audit(
        self,
        issues: List[Dict[str, Any]],
        law_used: List[Dict[str, Any]],
        draft_answer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Audits answer completeness and legal grounding against retrieved laws.
        
        Returns:
            Dict with:
                - 'is_complete': bool
                - 'all_issues_lack_law': bool
                - 'missing_issues': List of dicts (issue_id, missing_aspect, search_query)
        """
        # If draft status is already NO_LAW_FOUND or laws are completely empty
        if draft_answer.get("status") == "NO_LAW_FOUND" or not law_used:
            return {
                "is_complete": True,
                "all_issues_lack_law": True,
                "missing_issues": []
            }

        # If there is only 1 issue and laws were retrieved, fast heuristic can check if reasoning is substantive
        reasoning = draft_answer.get("legal_reasoning", "")
        direct = draft_answer.get("direct_answer", "")

        # Format concise issues
        issues_text = "\n".join([
            f"- {iss.get('issue_id', 'Q')}: {iss.get('sub_query', '')}"
            for iss in issues
        ])

        # Format concise law context (entry + snippet of description)
        law_lines = []
        for law in law_used[:6]:
            entry = law.get("entry", "")
            desc = str(law.get("description", "")).replace("\n", " ").strip()[:200]
            law_lines.append(f"* {entry}: {desc}")
        law_context = "\n".join(law_lines)

        draft_text = f"คำตอบตรง (Direct Answer): {direct}\nเหตุผลและข้อกฎหมาย (Reasoning): {reasoning}"

        prompt = (
            get_prompt("AUDIT_COMPLETENESS_PROMPT")
            .replace("{issues_text}", issues_text)
            .replace("{law_context}", law_context)
            .replace("{draft_answer}", draft_text)
        )

        response = self.model.generate_response(prompt, max_length=512)

        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        first = cleaned.find("{")
        last = cleaned.rfind("}")
        parsed = {}
        if first != -1 and last != -1 and last > first:
            json_str = cleaned[first:last + 1]
            try:
                parsed = json.loads(json_str)
            except Exception:
                fixed = re.sub(r",\s*([\]}])", r"\1", json_str)
                try:
                    parsed = json.loads(fixed)
                except Exception:
                    pass

        is_complete = bool(parsed.get("is_complete", True))
        all_lack = bool(parsed.get("all_issues_lack_law", False))
        missing_issues = parsed.get("missing_issues", [])
        if not isinstance(missing_issues, list):
            missing_issues = []

        # If there are missing issues, is_complete MUST be False
        if missing_issues and len(missing_issues) > 0:
            is_complete = False

        return {
            "is_complete": is_complete,
            "all_issues_lack_law": all_lack,
            "missing_issues": missing_issues
        }
