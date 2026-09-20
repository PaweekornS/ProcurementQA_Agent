"""Issue Decomposer & Intent Classifier Agent for LegalGraphRAG CRAG"""
import json
import re
from typing import Dict, Any, List
from core.prompt import get_prompt


class IssueDecomposer:
    """Agent 1: Decomposes complex procurement inquiries into atomic legal issues/sub-queries."""

    def __init__(self, model):
        self.model = model

    def decompose(self, fact: str, name: str = "ผู้สอบถาม") -> Dict[str, Any]:
        """
        Decomposes the inquiry into sub-issues and extracts procurement features.
        
        Returns:
            Dict with:
                - 'issues': List of dicts, each with 'issue_id', 'topic', 'sub_query', 'search_keywords'
                - 'procurement_features': Dict with procurement entity categories for graph traversal
        """
        prompt = get_prompt("INTENT_DECOMPOSE_PROMPT").replace("{fact}", fact)
        response = self.model.generate_response(prompt, max_length=1024)

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

        # Normalize and guarantee schema
        raw_issues = parsed.get("issues", [])
        if not isinstance(raw_issues, list) or len(raw_issues) == 0:
            raw_issues = [
                {
                    "issue_id": "Q1",
                    "topic": fact[:80] if fact else "ประเด็นการจัดซื้อจัดจ้าง",
                    "sub_query": fact,
                    "search_keywords": [fact[:60]] if fact else ["การจัดซื้อจัดจ้าง"]
                }
            ]

        # Ensure valid fields in every issue
        valid_issues = []
        for idx, item in enumerate(raw_issues, start=1):
            if not isinstance(item, dict):
                continue
            iid = item.get("issue_id") or f"Q{idx}"
            sub_q = item.get("sub_query") or fact
            topic = item.get("topic") or sub_q[:60]
            kw = item.get("search_keywords", [])
            if isinstance(kw, str):
                kw = [kw]
            elif not isinstance(kw, list):
                kw = []
            valid_issues.append({
                "issue_id": str(iid),
                "topic": str(topic).strip(),
                "sub_query": str(sub_q).strip(),
                "search_keywords": [str(k).strip() for k in kw if str(k).strip()]
            })

        if not valid_issues:
            valid_issues = [
                {
                    "issue_id": "Q1",
                    "topic": fact[:80] if fact else "ประเด็นการจัดซื้อจัดจ้าง",
                    "sub_query": fact,
                    "search_keywords": [fact[:60]] if fact else ["การจัดซื้อจัดจ้าง"]
                }
            ]

        # Extract features for graph traversal compatibility
        feat = parsed.get("procurement_features", {})
        if not isinstance(feat, dict):
            feat = {}
        if "defendant_info" not in feat or not feat["defendant_info"]:
            feat["defendant_info"] = [name if name else "หน่วยงานของรัฐ / ผู้สอบถาม"]
        if "criminal_acts" not in feat or not feat["criminal_acts"]:
            feat["criminal_acts"] = [issue["topic"] for issue in valid_issues]
        if "victim_property_details" not in feat or not feat["victim_property_details"]:
            feat["victim_property_details"] = ["พัสดุ / ขอบเขตงาน (TOR) / สัญญา"]
        if "intent_remorse" not in feat:
            feat["intent_remorse"] = []

        return {
            "issues": valid_issues,
            "procurement_features": feat
        }
