import json
import re
from typing import Dict, Any, List, Optional


from core.crag.classifier import IssueDecomposer
from core.crag.synthesizer import LegalSynthesizer
from core.crag.auditor import CompletenessAuditor
from core.crag.refiner import QueryRefiner

from core.graph_construct.feature_graph import query_similar_nodes
from core.graph_construct.graph_db import GraphDBManager
from core.crag.intra_doc_search import intra_doc_search
from core.crag.intra_doc_graph import traverse_intra_doc_graph, build_intra_doc_relations
from core.utils.util import concat_feature_descriptions, filter_facts
from core.judge.judge_crime import FALLBACK_NO_LAW_ANSWER


def merge_and_dedup_laws(law_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merges multiple lists of retrieved laws, keeping the highest rerank_score per entry."""
    seen = {}
    for lst in law_lists:
        for item in lst:
            lid = item.get("id") or item.get("entry")
            if not lid:
                continue
            curr_score = float(item.get("rerank_score", 0.0))
            if lid not in seen or curr_score > float(seen[lid].get("rerank_score", 0.0)):
                seen[lid] = item
    
    merged = list(seen.values())
    merged.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    return merged


class CRAGPipeline:
    """
    Coordinates the Multi-Agent Corrective RAG (CRAG) lifecycle:
    1. Issue Decomposition & Feature Extraction
    2. Multi-Aspect Hybrid Retrieval & Graph Traversal
    3. Legal Synthesis
    4. Completeness & Grounding Auditing
    5. Query Refinement Loop (max_retry=1)
    """

    def __init__(self, model, retrieve_config: Optional[Dict[str, Any]] = None, max_retry: int = 1):
        self.model = model
        self.retrieve_config = retrieve_config or {
            "top_retrieve": True,
            "direct_retrieve": True,
            "augment_retrieve": False,
            "top_retrieve_top_k": 3,
            "direct_retrieve_top_k": 5
        }
        self.max_retry = max_retry

        self.classifier = IssueDecomposer(model)
        self.synthesizer = LegalSynthesizer(model)
        self.auditor = CompletenessAuditor(model)
        self.refiner = QueryRefiner(model)

    def process_case_item(
        self,
        item: Dict[str, Any],
        law_to_crime: List[Dict[str, Any]],
        cases_db: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processes a single inquiry item through the full CRAG multi-agent loop.
        """
        raw_fact = item.get("description", "")
        name = item.get("name", "ผู้สอบถาม")

        # -------------------------------------------------------------
        # Step 1: Issue Decomposition & Feature Extraction (Agent 1)
        # -------------------------------------------------------------
        decomp_result = self.classifier.decompose(raw_fact, name=name)
        issues = decomp_result.get("issues", [])
        features = decomp_result.get("procurement_features", {})
        item["feature"] = features
        item["issues"] = issues

        # -------------------------------------------------------------
        # Step 2: Multi-Aspect Retrieval across Sub-Queries
        # -------------------------------------------------------------
        retrieved_law_batches = []
        all_retrieved_facts = []

        # 2.1 Primary query with features
        feature_query = concat_feature_descriptions(features, raw_text=raw_fact)
        _, init_facts, init_laws = query_similar_nodes(self.model, feature_query, self.retrieve_config)
        if init_laws:
            retrieved_law_batches.append(init_laws)
        if init_facts:
            all_retrieved_facts.extend(init_facts)

        # 2.2 Sub-query searches for multi-issue inquiries (ensures no sub-issue is crowded out)
        if len(issues) > 1:
            for iss in issues:
                sub_q = iss.get("sub_query", "")
                kws = " ".join(iss.get("search_keywords", []))
                q_text = f"{sub_q} {kws}".strip()
                if len(q_text) > 5:
                    _, sub_facts, sub_laws = query_similar_nodes(self.model, q_text, self.retrieve_config)
                    if sub_laws:
                        retrieved_law_batches.append(sub_laws)
                    if sub_facts:
                        all_retrieved_facts.extend(sub_facts)

        candidate_laws = merge_and_dedup_laws(retrieved_law_batches)

        # Handle NO_LAW_FOUND if initial retrieval is completely blank
        if not candidate_laws and not all_retrieved_facts:
            item["judge_result"] = {
                "status": "NO_LAW_FOUND",
                "direct_answer": FALLBACK_NO_LAW_ANSWER,
                "legal_reasoning": FALLBACK_NO_LAW_ANSWER,
                "applicable_laws": [],
                "exceptions_or_conditions": "",
                "law_article": []
            }
            item["retrieved_laws"] = []
            item["retrieved_facts"] = []
            item["used_laws"] = []
            item["used_facts"] = []
            item["crag_meta"] = {"issues": issues, "retries": 0, "complete": False}
            return item

        # Ensure at least 10 candidate laws for synthesis to cover multi-part questions
        max_laws = max(10, int(self.retrieve_config.get("direct_retrieve_top_k", 8)))
        law_used = candidate_laws[:max_laws]
        fact_used = filter_facts(law_used, all_retrieved_facts) if all_retrieved_facts else []

        # -------------------------------------------------------------
        # Step 3: Legal Synthesis (Agent 2)
        # -------------------------------------------------------------
        draft_answer = self.synthesizer.synthesize(
            law_used=law_used,
            retrieved_facts=fact_used,
            case_description=raw_fact,
            issues=issues
        )

        direct_text = str(draft_answer.get("direct_answer", "")).strip()
        if direct_text.startswith("ไม่พบข้อกฎหมาย") or direct_text.startswith("ไม่อยู่ในขอบเขต"):
            draft_answer["status"] = "NO_LAW_FOUND"


        audit_res = {}
        # If classified as NO_LAW_FOUND, do not immediately give up if we have retries left!
        # Treat all issues as missing issues to trigger refiner loop
        if draft_answer.get("status") == "NO_LAW_FOUND":
            is_complete = False
            missing_issues = [
                {
                    "issue_id": iss.get("issue_id", f"Q{i+1}"),
                    "missing_aspect": iss.get("sub_query", iss.get("topic", "")),
                    "search_query": f"{iss.get('sub_query', '')} {' '.join(iss.get('search_keywords', []))}".strip()
                }
                for i, iss in enumerate(issues)
            ]
            audit_res = {
                "is_complete": False,
                "all_issues_lack_law": True,
                "has_ungrounded_content": False,
                "recommend_fallback_no_law": True,
                "missing_issues": missing_issues
            }
        else:
            # -------------------------------------------------------------
            # Step 4: Completeness Auditing (Agent 3)
            # -------------------------------------------------------------
            audit_res = self.auditor.audit(issues, law_used, draft_answer)
            is_complete = audit_res.get("is_complete", True)
            missing_issues = audit_res.get("missing_issues", [])

            # If auditor flagged ungrounded content or recommended fallback but missing_issues was empty,
            # formulate missing issues from the sub-queries so refiner can search for the missing laws.
            if not is_complete and not missing_issues:
                missing_issues = [
                    {
                        "issue_id": iss.get("issue_id", f"Q{i+1}"),
                        "missing_aspect": iss.get("sub_query", iss.get("topic", "")),
                        "search_query": f"{iss.get('sub_query', '')} {' '.join(iss.get('search_keywords', []))}".strip()
                    }
                    for i, iss in enumerate(issues)
                ]

        retry_count = 0

        # -------------------------------------------------------------
        # Step 5: Intra-Doc Scan & Query Refiner Active Loop [max_retry=1]
        # -------------------------------------------------------------
        if not is_complete and missing_issues and self.max_retry > 0:
            retry_count += 1
            new_law_batches = []

            # 5.1 Intra-Document Legal Graph Traversal (Cross-Citation, Adjacency & Chapter Edges)
            try:
                db_inst = GraphDBManager.get_db()
                for miss in missing_issues:
                    aspect = miss.get("search_query") or miss.get("missing_aspect") or ""
                    graph_neighbors = traverse_intra_doc_graph(db_inst, law_used, aspect, top_k=2)
                    if graph_neighbors:
                        new_law_batches.append(graph_neighbors)
            except Exception:
                pass

            # 5.2 Targeted Intra-Document Scan across already retrieved candidate docs
            intra_matched = []
            for miss in missing_issues:
                aspect = miss.get("search_query") or miss.get("missing_aspect") or ""
                if len(aspect.strip()) > 3:
                    m_nodes, is_found = intra_doc_search(candidate_laws, aspect, top_k=2)
                    if m_nodes:
                        intra_matched.extend(m_nodes)

            if intra_matched:
                new_law_batches.append(intra_matched)

            # 5.3 Agent 4 Refiner (Graph neighbors + targeted refined search)
            refine_res = self.refiner.refine(missing_issues, raw_fact, law_used)
            refined_queries = refine_res.get("refined_queries", [])
            neighbor_laws = refine_res.get("neighbor_laws", [])

            if neighbor_laws:
                new_law_batches.append(neighbor_laws)

            for rq in refined_queries[:2]:
                if len(rq.strip()) > 5:
                    _, r_facts, r_laws = query_similar_nodes(self.model, rq, self.retrieve_config)
                    if r_laws:
                        new_law_batches.append(r_laws)
                    if r_facts:
                        all_retrieved_facts.extend(r_facts)

            if new_law_batches:
                candidate_laws = merge_and_dedup_laws([candidate_laws] + new_law_batches)
                # Filter coarse multi-page raw chunk bundles (_p...) to protect context token budget
                clean_candidates = [
                    l for l in candidate_laws
                    if not re.search(r"_p\d+", str(l.get("id", "")))
                ]
                if clean_candidates:
                    candidate_laws = clean_candidates
                law_used = candidate_laws[:max_laws]
                fact_used = filter_facts(law_used, all_retrieved_facts) if all_retrieved_facts else []

            # Re-synthesize with full context
            draft_answer = self.synthesizer.synthesize(
                law_used=law_used,
                retrieved_facts=fact_used,
                case_description=raw_fact,
                issues=issues,
                unfound_issues=missing_issues
            )

            direct_text = str(draft_answer.get("direct_answer", "")).strip()
            if direct_text.startswith("ไม่พบข้อกฎหมาย") or direct_text.startswith("ไม่อยู่ในขอบเขต"):
                draft_answer["status"] = "NO_LAW_FOUND"


            # Post-retry audit to strictly verify grounding
            post_audit = self.auditor.audit(issues, law_used, draft_answer)
            audit_res = post_audit
            is_complete = post_audit.get("is_complete", True)

        # -------------------------------------------------------------
        # Step 6: Grounding Fallback Gate
        # -------------------------------------------------------------
        # Fallback gracefully to NO_LAW_FOUND only when genuine statutory absence:
        # 1. Draft status is explicitly NO_LAW_FOUND (synthesizer found no basis)
        # 2. All issues lack statutory basis or auditor explicitly recommended fallback
        should_fallback = (
            draft_answer.get("status") == "NO_LAW_FOUND"
            or (audit_res.get("all_issues_lack_law", False) and not draft_answer.get("applicable_laws"))
        )


        if should_fallback:
            draft_answer = {
                "status": "NO_LAW_FOUND",
                "direct_answer": FALLBACK_NO_LAW_ANSWER,
                "legal_reasoning": FALLBACK_NO_LAW_ANSWER,
                "applicable_laws": [],
                "exceptions_or_conditions": "",
                "law_article": []
            }
        else:
            if not draft_answer.get("legal_reasoning"):
                draft_answer["legal_reasoning"] = draft_answer.get("direct_answer", "")

        item["judge_result"] = draft_answer
        item["retrieved_laws"] = candidate_laws
        item["retrieved_facts"] = all_retrieved_facts
        item["used_laws"] = law_used
        item["used_facts"] = fact_used
        item["crag_meta"] = {
            "issues": issues,
            "retries": retry_count,
            "audited_complete": is_complete,
            "audit_result": audit_res,
            "missing_issues": missing_issues if not is_complete else []
        }
        return item
