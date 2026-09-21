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
                "is_complete": False,
                "all_issues_lack_law": True,
                "has_ungrounded_content": False,
                "recommend_fallback_no_law": True,
                "missing_issues": []
            }

        reasoning = draft_answer.get("legal_reasoning", "")
        direct = draft_answer.get("direct_answer", "")

        # Format concise issues
        issues_text = "\n".join([
            f"- {iss.get('issue_id', 'Q')}: {iss.get('sub_query', '')}"
            for iss in issues
        ])

        # Format law context (up to 10 laws, 600 chars each for comprehensive grounding check)
        law_lines = []
        for law in law_used[:10]:
            entry = law.get("entry", "")
            desc = str(law.get("description", "")).replace("\n", " ").strip()[:600]
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
        has_ungrounded = bool(parsed.get("has_ungrounded_content", False))
        recommend_fallback = bool(parsed.get("recommend_fallback_no_law", False))
        raw_missing = parsed.get("missing_issues", [])
        if not isinstance(raw_missing, list):
            raw_missing = []

        norm_missing = []
        for m in raw_missing:
            if isinstance(m, dict):
                norm_missing.append({
                    "issue_id": m.get("issue_id", "Q"),
                    "missing_aspect": m.get("missing_aspect", str(m)),
                    "search_query": m.get("search_query", "")
                })
            elif isinstance(m, str) and m.strip():
                norm_missing.append({
                    "issue_id": "Q",
                    "missing_aspect": m.strip(),
                    "search_query": f"{m.strip()} ระเบียบกระทรวงการคลัง พ.ร.บ. จัดซื้อจัดจ้าง"
                })

        # If there are missing issues or ungrounded content or fallback recommendation, is_complete MUST be False
        if norm_missing or has_ungrounded or recommend_fallback or all_lack:
            is_complete = False

        return {
            "is_complete": is_complete,
            "all_issues_lack_law": all_lack,
            "has_ungrounded_content": has_ungrounded,
            "recommend_fallback_no_law": recommend_fallback or all_lack,
            "missing_issues": norm_missing
        }
