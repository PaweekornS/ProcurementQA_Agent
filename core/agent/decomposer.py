"""Issue Decomposer & Intent Classifier Agent for ProcurementQA Agent Agentic Workflow"""
import json
import re
from typing import Dict, Any, List
from core.prompts import get_prompt

# Fields the decomposer extracts from a question (INTENT_DECOMPOSE_PROMPT)
FEATURE_FIELDS = ("stakeholders", "procurement_topics", "scope_and_budget", "conditions_or_exceptions")


class IssueDecomposer:
    """Decomposes complex procurement inquiries into atomic legal issues/sub-queries."""

    def __init__(self, model):
        self.model = model

    def decompose(self, question: str) -> Dict[str, Any]:
        """
        Decomposes the inquiry into sub-issues and extracts procurement features.
        
        Returns:
            Dict with:
                - 'issues': List of dicts, each with 'issue_id', 'topic', 'sub_query', 'search_keywords'
                - 'procurement_features': {field: [values]} for FEATURE_FIELDS (trace output)
        """
        prompt = get_prompt("INTENT_DECOMPOSE_PROMPT").replace("{question}", question)
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
                    "topic": question[:80] if question else "ประเด็นการจัดซื้อจัดจ้าง",
                    "sub_query": question,
                    "search_keywords": [question[:60]] if question else ["การจัดซื้อจัดจ้าง"]
                }
            ]

        # Ensure valid fields in every issue
        valid_issues = []
        for idx, item in enumerate(raw_issues, start=1):
            if not isinstance(item, dict):
                continue
            iid = item.get("issue_id") or f"Q{idx}"
            topic = item.get("topic") or question[:60]
            sq = item.get("sub_query") or question
            kws = item.get("search_keywords")
            if not isinstance(kws, list) or len(kws) == 0:
                kws = [sq[:50]]
            valid_issues.append({
                "issue_id": str(iid),
                "topic": str(topic),
                "sub_query": str(sq),
                "search_keywords": [str(k) for k in kws]
            })

        if not valid_issues:
            valid_issues = [{
                "issue_id": "Q1",
                "topic": question[:80] if question else "ประเด็นการจัดซื้อจัดจ้าง",
                "sub_query": question,
                "search_keywords": [question[:60]] if question else ["การจัดซื้อจัดจ้าง"]
            }]

        raw_features = parsed.get("procurement_features")
        raw_features = raw_features if isinstance(raw_features, dict) else {}
        features = {}
        for field in FEATURE_FIELDS:
            values = raw_features.get(field, [])
            values = values if isinstance(values, list) else [values]
            features[field] = [str(v) for v in values if str(v).strip()]
        return {"issues": valid_issues, "procurement_features": features}
