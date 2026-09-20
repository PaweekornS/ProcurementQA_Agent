"""Query Refiner Agent for LegalGraphRAG CRAG"""
import json
import re
from typing import Dict, Any, List
from core.prompt import get_prompt
from core.graph_construct.graph_db import GraphDBManager


class QueryRefiner:
    """Agent 4: Generates focused search queries and traverses graph neighbors for missing legal aspects."""

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
            json_str = cleaned[first:last + 1]
            try:
                parsed = json.loads(json_str)
            except Exception:
                fixed = re.sub(r",\s*([\]}])", r"\1", json_str)
                try:
                    parsed = json.loads(fixed)
                except Exception:
                    pass

        refined_queries = parsed.get("refined_queries", [])
        if not isinstance(refined_queries, list):
            refined_queries = []

        # Merge in any high-quality pre-suggested queries from the auditor
        for sq in pre_suggested_queries:
            if sq and sq not in refined_queries:
                refined_queries.append(sq)

        # If LLM failed to produce refined queries, use the missing aspects directly
        if not refined_queries:
            refined_queries = [
                f"{m.get('missing_aspect', '')} ระเบียบกระทรวงการคลัง พ.ร.บ. จัดซื้อจัดจ้าง"
                for m in missing_issues if m.get("missing_aspect")
            ]

        # 4. Graph Neighbor Expansion: Pull 1-hop related statutory nodes
        neighbor_laws = []
        try:
            db = GraphDBManager.get_db()
            seen_ids = {l.get("id") for l in existing_laws if l.get("id")}
            for law in existing_laws[:5]:
                node_id = law.get("id")
                if not node_id:
                    continue
                # Traverse RELATED_TO or RELATES_TO_LAW edges
                for rel in ["RELATED_TO", "RELATES_TO_LAW"]:
                    neighbor_ids = db.get_neighbors(node_id, rel)
                    for nid in neighbor_ids:
                        if nid not in seen_ids:
                            ndata = db.get_node(nid)
                            if ndata and (ndata.get("entry") or ndata.get("description")):
                                neighbor_laws.append({
                                    "id": nid,
                                    "entry": ndata.get("entry", ""),
                                    "description": ndata.get("description", ""),
                                    "crimes": ndata.get("crimes", []),
                                    "judge_dep": ndata.get("judge_dep", []),
                                    "related_laws": ndata.get("related_laws", []),
                                    "rerank_score": 0.85  # Boost adjacent statutory neighbor
                                })
                                seen_ids.add(nid)
        except Exception:
            pass

        return {
            "refined_queries": [q.strip() for q in refined_queries if q.strip()],
            "target_topics": parsed.get("target_topics", []),
            "neighbor_laws": neighbor_laws
        }
