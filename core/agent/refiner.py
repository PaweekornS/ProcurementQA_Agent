"""Query Refiner Agent for LegalGraphRAG Agentic Workflow"""
import json
import re
from typing import Dict, Any, List
from core.prompt import get_prompt
from core.graph_construct.graph_db import GraphDBManager


class QueryRefiner:
    """Generates focused search queries and traverses graph neighbors for missing legal aspects."""

    def __init__(self, model):
        self.model = model

    def refine(
        self,
        missing_issues: List[Dict[str, Any]],
        original_fact: str,
        existing_laws: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Refines search queries and identifies graph neighbor candidates for missing issues.
        
        Returns:
            Dict with:
                - 'refined_queries': List of sharp search strings
                - 'target_topics': List of topic categories
                - 'neighbor_laws': List of additional candidate law dicts pulled from the graph
        """
        if not missing_issues:
            return {
                "refined_queries": [],
                "target_topics": [],
                "neighbor_laws": []
            }

        # 1. Format missing aspects summary
        missing_aspects_lines = []
        pre_suggested_queries = []
        for m in missing_issues:
            iid = m.get("issue_id", "Q")
            aspect = m.get("missing_aspect", "")
            sq = m.get("search_query", "")
            missing_aspects_lines.append(f"- [{iid}] {aspect}")
            if sq:
                pre_suggested_queries.append(sq)

        missing_aspects_str = "\n".join(missing_aspects_lines)

        # 2. Existing law entries
        existing_entries = [str(l.get("entry", "")) for l in existing_laws if l.get("entry")]
        existing_laws_str = ", ".join(existing_entries[:8]) if existing_entries else "ไม่มี"

        # 3. LLM Query Refinement
        prompt = (
            get_prompt("QUERY_REFINE_PROMPT")
            .replace("{missing_aspects}", missing_aspects_str)
            .replace("{original_fact}", original_fact[:800])
            .replace("{existing_laws}", existing_laws_str)
        )
        response = self.model.generate_response(prompt, max_length=512)

        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        first = cleaned.find("{")
        last = cleaned.rfind("}")
        parsed = {}
        if first != -1 and last != -1 and last > first:
            try:
                parsed = json.loads(cleaned[first:last + 1])
            except Exception:
                pass

        refined_queries = parsed.get("refined_queries", [])
        if not isinstance(refined_queries, list) or not refined_queries:
            # Fallback to pre-suggested search queries or missing aspects directly
            refined_queries = pre_suggested_queries if pre_suggested_queries else [
                m.get("missing_aspect", "") for m in missing_issues if m.get("missing_aspect")
            ]

        target_topics = parsed.get("target_topics", [])
        if not isinstance(target_topics, list):
            target_topics = []

        return {
            "refined_queries": [str(q) for q in refined_queries if q],
            "target_topics": [str(t) for t in target_topics if t],
            "neighbor_laws": []
        }

    def expand_via_graph(
        self,
        db: GraphDBManager,
        existing_laws: List[Dict[str, Any]],
        max_expand: int = 5
    ) -> List[Dict[str, Any]]:
        """Traverses the Tri-Store / Neo4j knowledge graph to fetch related regulations and exceptions."""
        expanded_chunks: List[Dict[str, Any]] = []
        seen_entries = {l.get("entry") for l in existing_laws if l.get("entry")}

        for law in existing_laws[:3]:
            entry = law.get("entry")
            if not entry or not hasattr(db, "get_related_clauses"):
                continue

            try:
                related = db.get_related_clauses(entry, max_hops=1, limit=max_expand)
                for r in related:
                    r_entry = r.get("entry")
                    if r_entry and r_entry not in seen_entries:
                        seen_entries.add(r_entry)
                        expanded_chunks.append({
                            "entry": r_entry,
                            "content": r.get("content", ""),
                            "source": r.get("source", "Graph Expansion"),
                            "score": 0.85
                        })
            except Exception:
                pass

        return expanded_chunks[:max_expand]
