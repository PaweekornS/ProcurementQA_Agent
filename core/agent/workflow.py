# -*- coding: utf-8 -*-
"""
core/agent/workflow.py

LangGraph-based Adaptive & Self-Reflective Agentic RAG Workflow for Thai Procurement Legal QA.
Orchestrates:
  START -> analyze_query -> retrieve_and_rerank -> jev_gatekeeper
                                                        ├─ (no relevant chunks & retry < max) -> rewrite_query -> retrieve_and_rerank
                                                        └─ (sufficient context) -> generate_answer -> guardrail -> END
"""

import os
import time
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from core.agent.state import AgenticRAGState
from core.agent.guardrail import GroundingGuardrail
from core.utils.agent_logger import AgentTraceLogger
from core.crag.classifier import IssueDecomposer
from core.crag.synthesizer import LegalSynthesizer
from core.crag.refiner import QueryRefiner
from core.crag.jev_filter import JevChunkFilter
from core.graph_construct.feature_graph import query_similar_nodes
from core.utils.util import concat_feature_descriptions


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


class ProcurementAgenticWorkflow:
    """Compiled LangGraph Agentic RAG Workflow with Adaptive Query Rewriting and Grounding Guardrail."""

    def __init__(self, model, retrieve_config: Optional[Dict[str, Any]] = None, max_retries: int = 2):
        self.model = model
        self.retrieve_config = retrieve_config or {
            "top_retrieve": True,
            "direct_retrieve": True,
            "augment_retrieve": False,
            "top_retrieve_top_k": 3,
            "direct_retrieve_top_k": 5
        }
        self.max_retries = max_retries
        self.classifier = IssueDecomposer(model)
        self.synthesizer = LegalSynthesizer(model)
        self.refiner = QueryRefiner(model)
        self.jev_filter = JevChunkFilter()
        self.logger = AgentTraceLogger.get_instance()
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgenticRAGState)

        # Add Nodes
        workflow.add_node("analyze_query", self._node_analyze_query)
        workflow.add_node("retrieve_and_rerank", self._node_retrieve_and_rerank)
        workflow.add_node("jev_gatekeeper", self._node_jev_gatekeeper)
        workflow.add_node("rewrite_query", self._node_rewrite_query)
        workflow.add_node("generate_answer", self._node_generate_answer)
        workflow.add_node("guardrail", self._node_guardrail)

        # Add Edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "retrieve_and_rerank")
        workflow.add_edge("retrieve_and_rerank", "jev_gatekeeper")

        # Conditional Edge after Jev Gatekeeper
        workflow.add_conditional_edges(
            "jev_gatekeeper",
            self._should_rewrite,
            {
                "rewrite": "rewrite_query",
                "generate": "generate_answer"
            }
        )
        workflow.add_edge("rewrite_query", "retrieve_and_rerank")
        
        # Self-Reflective Conditional Edge after Answer Generation
        workflow.add_conditional_edges(
            "generate_answer",
            self._after_generation_route,
            {
                "rewrite": "rewrite_query",
                "guardrail": "guardrail"
            }
        )
        workflow.add_edge("guardrail", END)

        return workflow.compile()

    def _should_rewrite(self, state: AgenticRAGState) -> str:
        """Route to rewrite_query if no chunks passed gatekeeper and retries remain."""
        if state.get("needs_rewrite", False) and state.get("retry_count", 0) < state.get("max_retries", self.max_retries):
            return "rewrite"
        return "generate"

    def _after_generation_route(self, state: AgenticRAGState) -> str:
        """Self-reflect after generation: if status is NO_LAW_FOUND, trigger query rewrite retry."""
        if (
            (state.get("status") == "NO_LAW_FOUND" or not state.get("applicable_laws"))
            and state.get("retry_count", 0) < state.get("max_retries", self.max_retries)
        ):
            return "rewrite"
        return "guardrail"

    def _node_analyze_query(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        raw_query = state.get("raw_query", "")
        name = state.get("name", "ผู้สอบถาม")

        decomp = self.classifier.decompose(raw_query, name=name)
        issues = decomp.get("issues", [])
        features = decomp.get("procurement_features", {})

        self.logger.log_step(
            case_id=case_id,
            step_num=1,
            step_name="Query Analysis & Issue Decomposition",
            message=f"Identified {len(issues)} sub-issue(s)",
            details={"Stakeholder": name, "Issues": len(issues)},
            emoji="🟢"
        )

        trace_event = {
            "step": "analyze_query",
            "issues_count": len(issues),
            "features": features
        }

        return {
            "features": features,
            "issues": issues,
            "current_query": raw_query,
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_retrieve_and_rerank(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        features = state.get("features", {})
        issues = state.get("issues", [])
        raw_query = state.get("current_query") or state.get("raw_query", "")

        retrieved_batches = []
        feature_query = concat_feature_descriptions(features, raw_text=raw_query)
        _, _, init_laws = query_similar_nodes(self.model, feature_query, self.retrieve_config)
        if init_laws:
            retrieved_batches.append(init_laws)

        # Multi-aspect sub-query retrieval
        if len(issues) > 1:
            for iss in issues:
                sub_q = iss.get("sub_query", "")
                kws = " ".join(iss.get("search_keywords", []))
                q_text = f"{sub_q} {kws}".strip()
                if len(q_text) > 5:
                    _, _, sub_laws = query_similar_nodes(self.model, q_text, self.retrieve_config)
                    if sub_laws:
                        retrieved_batches.append(sub_laws)

        # If rewritten in previous retry, retrieve using the refined query directly
        current_q = state.get("current_query", "")
        orig_q = state.get("raw_query", "")
        if current_q and current_q != orig_q:
            _, _, refined_laws = query_similar_nodes(self.model, current_q, self.retrieve_config)
            if refined_laws:
                retrieved_batches.append(refined_laws)

        candidate_laws = merge_and_dedup_laws(retrieved_batches)

        # Knowledge Graph Traversal & Topological Backup:
        # 1. Deterministic statutory entity resolution
        # 2. Multi-hop statutory citation traversal (EMPOWERS, CITED_BY, CITES, NEXT_SECTION)
        # 3. Topic & Crime graph topology expansion
        try:
            from core.graph_construct.citation_linker import LegalCitationLinker
            from core.graph_construct.graph_db import GraphDBManager
            db_inst = GraphDBManager.get_db()
            graph_results = LegalCitationLinker.graph_search_backup(
                db=db_inst,
                query=raw_query,
                features=features,
                issues=issues,
                seed_laws=candidate_laws,
                top_k=4
            )
            if graph_results:
                self.logger.log_step(
                    case_id=case_id,
                    step_num=2.5,
                    step_name="Knowledge Graph Search & Relationship Traversal",
                    message=f"Discovered {len(graph_results)} section(s) via Graph Citations & Entity Lookup",
                    details={"Graph Results": [g.get("entry", "")[:70] for g in graph_results[:3]]},
                    emoji="🕸️"
                )
                candidate_laws = merge_and_dedup_laws([candidate_laws, graph_results])
        except Exception as e:
            pass

        existing_cands = state.get("retrieved_candidates", [])
        all_candidates = merge_and_dedup_laws([existing_cands, candidate_laws])

        self.logger.log_step(
            case_id=case_id,
            step_num=2,
            step_name="Hybrid Retrieval & Opper Reranker",
            message=f"Retrieved and reranked {len(candidate_laws)} candidate chunks (Total accumulated: {len(all_candidates)})",
            emoji="🔍"
        )

        trace_event = {
            "step": "retrieve_and_rerank",
            "candidates_count": len(all_candidates),
            "top_candidates": [c.get("entry", "") for c in all_candidates[:5]]
        }

        return {
            "retrieved_candidates": all_candidates,
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_jev_gatekeeper(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        raw_query = state.get("raw_query", "")
        candidates = state.get("retrieved_candidates", [])

        use_jev = os.getenv("USE_JEV_FILTER", "false").lower() in ("true", "1", "yes")

        if use_jev and self.jev_filter.enabled and candidates:
            passed = self.jev_filter.filter_chunks(
                question=raw_query,
                candidate_laws=candidates,
                max_keep=10,
                min_keep=0  # Allow 0 to trigger self-reflective rewrite
            )
            passed_ids = {p.get("id") or p.get("entry") for p in passed}
            dropped = [c for c in candidates if (c.get("id") or c.get("entry")) not in passed_ids]
        else:
            passed = candidates[:10]
            dropped = candidates[10:]

        needs_rewrite = (len(passed) == 0)

        self.logger.log_step(
            case_id=case_id,
            step_num=3,
            step_name="Jev Gatekeeper Relevance Evaluation",
            message=f"{len(passed)} chunks kept, {len(dropped)} dropped",
            details={
                "Decision": "Rewrite Query (Insufficient Context)" if needs_rewrite else "Proceed to Generation",
                "Threshold": self.jev_filter.threshold
            },
            emoji="⚖️"
        )

        trace_event = {
            "step": "jev_gatekeeper",
            "passed_count": len(passed),
            "dropped_count": len(dropped),
            "decision": "rewrite" if needs_rewrite else "generate"
        }

        return {
            "passed_chunks": passed,
            "dropped_chunks": dropped,
            "needs_rewrite": needs_rewrite,
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_rewrite_query(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        retry_count = state.get("retry_count", 0) + 1
        raw_query = state.get("raw_query", "")
        issues = state.get("issues", [])
        existing_laws = state.get("retrieved_candidates", [])

        missing_issues = [
            {
                "issue_id": iss.get("issue_id", f"Q{i+1}"),
                "missing_aspect": iss.get("sub_query", iss.get("topic", "")),
                "search_query": f"{iss.get('sub_query', '')} {' '.join(iss.get('search_keywords', []))}".strip()
            }
            for i, iss in enumerate(issues)
        ]
        if not missing_issues:
            missing_issues = [{
                "issue_id": "Q1",
                "missing_aspect": raw_query,
                "search_query": raw_query
            }]

        try:
            refine_result = self.refiner.refine(
                missing_issues=missing_issues,
                original_fact=raw_query,
                existing_laws=existing_laws
            )
            refined_queries = refine_result.get("refined_queries", [])
            new_query = refined_queries[0] if refined_queries else missing_issues[0]["search_query"]
        except Exception:
            new_query = missing_issues[0]["search_query"] if missing_issues else raw_query

        self.logger.log_step(
            case_id=case_id,
            step_num=3.5,
            step_name="Query Transformation & Self-Correction",
            message=f"Attempt {retry_count}/{state.get('max_retries', self.max_retries)}: {new_query[:80]}...",
            details={"Refined Query": new_query[:100]},
            emoji="🔄"
        )

        trace_event = {
            "step": "rewrite_query",
            "retry_count": retry_count,
            "new_query": new_query
        }

        return {
            "current_query": new_query,
            "retry_count": retry_count,
            "needs_rewrite": False,
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_generate_answer(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        raw_query = state.get("raw_query", "")
        passed_chunks = state.get("passed_chunks", [])
        candidates = state.get("retrieved_candidates", [])

        # Fallback to candidate laws if passed chunks is empty after all retries
        effective_laws = passed_chunks if passed_chunks else candidates[:10]

        if not effective_laws:
            synth_res = {
                "status": "NO_LAW_FOUND",
                "direct_answer": "ไม่พบข้อกฎหมาย ระเบียบ หรือประกาศที่เกี่ยวข้องกับประเด็นข้อหารือนี้ในฐานข้อมูลการจัดซื้อจัดจ้างภาครัฐ",
                "decisive_quotes": [],
                "applicable_laws": [],
                "exceptions_or_conditions": ""
            }
        else:
            synth_res = self.synthesizer.synthesize(
                law_used=effective_laws,
                retrieved_facts=[],
                case_description=raw_query,
                issues=state.get("issues", [])
            )

        self.logger.log_step(
            case_id=case_id,
            step_num=4,
            step_name="Answer Generation & Statutory Synthesis",
            message=f"Status: {synth_res.get('status', 'COMPLIANT')}, Laws: {len(synth_res.get('applicable_laws', []))}",
            emoji="✍️"
        )

        trace_event = {
            "step": "generate_answer",
            "status": synth_res.get("status"),
            "applicable_laws": synth_res.get("applicable_laws", [])
        }

        return {
            "synthesized_answer": synth_res.get("direct_answer", ""),
            "direct_answer": synth_res.get("direct_answer", ""),
            "decisive_quotes": synth_res.get("decisive_quotes", []),
            "applicable_laws": synth_res.get("applicable_laws", []),
            "exceptions": synth_res.get("exceptions_or_conditions", ""),
            "status": synth_res.get("status", "COMPLIANT"),
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_guardrail(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        direct_answer = state.get("direct_answer", "")
        passed_chunks = state.get("passed_chunks", []) or state.get("retrieved_candidates", [])

        verdict = GroundingGuardrail.audit(direct_answer, passed_chunks)
        cited_sections = verdict.get("cited_sections", [])

        self.logger.log_step(
            case_id=case_id,
            step_num=5,
            step_name="Faithfulness & Grounding Guardrail",
            message=f"Verdict: {'Passed' if verdict['passed'] else 'Warnings'}, Citations: {cited_sections}",
            details={"Grounding Score": verdict.get("grounding_score", 1.0)},
            emoji="🛡️"
        )

        trace_record = {
            "case_id": case_id,
            "rag_mode": "agentic",
            "retries": state.get("retry_count", 0),
            "traces": state.get("trace_events", []),
            "verdict": verdict,
            "applicable_laws": state.get("applicable_laws", []),
            "direct_answer": direct_answer
        }
        self.logger.append_trace(trace_record)

        return {
            "guardrail_verdict": verdict,
            "trace_events": state.get("trace_events", []) + [{"step": "guardrail", "verdict": verdict}]
        }

    def invoke(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Entrypoint for executing the LangGraph Agentic Workflow on a case item."""
        raw_fact = case.get("fact") or case.get("description") or case.get("question", "")
        name = case.get("name", ["ผู้สอบถาม"])
        if isinstance(name, list) and len(name) > 0:
            name = name[0]

        initial_state: AgenticRAGState = {
            "case_id": case.get("id", 0),
            "raw_query": raw_fact,
            "current_query": raw_fact,
            "name": str(name),
            "features": {},
            "issues": [],
            "retrieved_candidates": [],
            "passed_chunks": [],
            "dropped_chunks": [],
            "retry_count": 0,
            "max_retries": self.max_retries,
            "needs_rewrite": False,
            "synthesized_answer": "",
            "direct_answer": "",
            "decisive_quotes": [],
            "applicable_laws": [],
            "exceptions": "",
            "status": "COMPLIANT",
            "guardrail_verdict": {},
            "trace_events": []
        }

        final_state = self.app.invoke(initial_state)

        return {
            "name": name,
            "description": raw_fact,
            "feature": final_state.get("features", {}),
            "judge_result": {
                "status": final_state.get("status", "COMPLIANT"),
                "direct_answer": final_state.get("direct_answer", ""),
                "decisive_quotes": final_state.get("decisive_quotes", []),
                "applicable_laws": final_state.get("applicable_laws", []),
                "law_article": final_state.get("applicable_laws", []),
                "exceptions_or_conditions": final_state.get("exceptions", ""),
                "guardrail_verdict": final_state.get("guardrail_verdict", {})
            },
            "retrieved_laws": final_state.get("passed_chunks", []) or final_state.get("retrieved_candidates", []),
            "retrieved_facts": [],
            "used_laws": final_state.get("passed_chunks", []) or final_state.get("retrieved_candidates", []),
            "used_facts": [],
            "crag_meta": {
                "issues": final_state.get("issues", []),
                "retries": final_state.get("retry_count", 0),
                "complete": True
            }
        }
