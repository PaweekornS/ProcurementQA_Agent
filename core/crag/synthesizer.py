"""Legal Synthesizer & Adjudicator Agent for LegalGraphRAG CRAG"""
import json
import re
from typing import Dict, Any, List, Optional
from core.judge.judge_crime import judge_crime_all, FALLBACK_NO_LAW_ANSWER


class LegalSynthesizer:
    """Agent 2: Synthesizes legal answers from retrieved statutory context and maps answers to sub-issues."""

    def __init__(self, model):
        self.model = model

    def synthesize(
        self,
        law_used: List[Dict[str, Any]],
        retrieved_facts: List[Dict[str, Any]],
        case_description: str,
        issues: Optional[List[Dict[str, Any]]] = None,
        unfound_issues: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes legal judgment and explanation.
        
        Args:
            law_used: Top candidate statutory rules/sections
            retrieved_facts: Top retrieved past cases or FAQ opinions
            case_description: Original user inquiry / fact
            issues: Decomposed sub-issues from IssueDecomposer
            unfound_issues: Issues for which no statutory rules were found after retry
        """
        # If no laws or facts were retrieved at all, return standard fallback
        if not law_used and not retrieved_facts:
            return {
                "status": "NO_LAW_FOUND",
                "direct_answer": FALLBACK_NO_LAW_ANSWER,
                "decisive_quotes": [],
                "applicable_laws": [],
                "exceptions_or_conditions": "",
                "law_article": []
            }

        # Contextual prompt enrichment: if issues are provided, append issue outline
        enriched_case = case_description
        if issues and len(issues) > 1:
            issue_list_str = "\n".join([f"- {iss['issue_id']}: {iss['sub_query']}" for iss in issues])
            enriched_case = (
                f"{case_description}\n\n"
                f"ประเด็นที่ต้องตอบให้ครบถ้วนทุกข้อ:\n{issue_list_str}"
            )

        result = judge_crime_all(self.model, law_used, retrieved_facts, enriched_case)

        # Clean direct_answer: ensure no section citations are inside direct_answer
        if result.get("direct_answer"):
            cleaned_direct = result["direct_answer"]
            # Remove patterns like (ตามมาตรา 56) or ตามระเบียบข้อ 25
            cleaned_direct = re.sub(r"\(?\s*(?:ตาม)?(?:มาตรา|ระเบียบข้อ|ข้อ|พ\.ร\.บ\.)\s*\d+[^\)]*?\)?", "", cleaned_direct).strip()
            # Clean leading/trailing punctuation left behind
            cleaned_direct = re.sub(r"^[\s,.-]+", "", cleaned_direct).strip()
            if cleaned_direct:
                result["direct_answer"] = cleaned_direct

        return result
