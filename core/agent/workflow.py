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
from core.agent.decomposer import IssueDecomposer
from core.agent.synthesizer import LegalSynthesizer
from core.agent.refiner import QueryRefiner
from core.retrieval.retriever import get_retriever
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

    def __init__(self, model, max_retries: int = 2):
        self.model = model
        self.max_retries = max_retries
        self.classifier = IssueDecomposer(model)
        self.synthesizer = LegalSynthesizer(model)
        self.refiner = QueryRefiner(model)
        self.logger = AgentTraceLogger.get_instance()
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgenticRAGState)

        # Add Nodes
        workflow.add_node("analyze_query", self._node_analyze_query)
        workflow.add_node("retrieve_and_rerank", self._node_retrieve_and_rerank)
        workflow.add_node("rewrite_query", self._node_rewrite_query)
        workflow.add_node("generate_answer", self._node_generate_answer)
        workflow.add_node("guardrail", self._node_guardrail)

        # Add Edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "retrieve_and_rerank")

        # Route after retrieval: if no candidates found, trigger rewrite
        workflow.add_conditional_edges(
            "retrieve_and_rerank",
            self._after_retrieval_route,
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

    def _after_retrieval_route(self, state: AgenticRAGState) -> str:
        """Route to rewrite_query if no candidates were found and retries remain."""
        if not state.get("retrieved_candidates") and state.get("retry_count", 0) < state.get("max_retries", self.max_retries):
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

        tools = list(state.get("tools_used", []))
        if "Decomposer" not in tools:
            tools.append("Decomposer")

        return {
            "features": features,
            "issues": issues,
            "current_query": raw_query,
            "tools_used": tools,
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_retrieve_and_rerank(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        features = state.get("features", {})
        issues = state.get("issues", [])
        raw_query = state.get("raw_query", "")
        current_query = state.get("current_query") or raw_query
        org_id = state.get("org_id") or os.getenv("DEFAULT_ORG_ID", "DGA")

        # Query variants: the question, its feature-enriched form, every sub-question and, after a
        # rewrite, the refined query. The retriever fuses their recall and reranks once against all.
        queries = [raw_query, concat_feature_descriptions(features, raw_text=raw_query)]
        if len(issues) > 1:
            for iss in issues:
                sub_q = f"{iss.get('sub_query', '')} {' '.join(iss.get('search_keywords', []))}".strip()
                if len(sub_q) > 5:
                    queries.append(sub_q)
        if current_query != raw_query:
            queries.append(current_query)

        result = get_retriever().retrieve(queries, org_id)
        candidate_laws = result.laws()

        # Retries add to what earlier rounds found instead of replacing it
        all_candidates = merge_and_dedup_laws([state.get("retrieved_candidates", []), candidate_laws])

        self.logger.log_step(
            case_id=case_id,
            step_num=2,
            step_name="Hybrid Retrieval, Graph Expansion & Rerank",
            message=(f"{len(result.queries)} query variant(s): {result.stats.get('kept', 0)} reranked chunk(s) "
                     f"+ {result.stats.get('graph_carried', 0)} cited by them (total accumulated: {len(all_candidates)})"),
            emoji="🔍"
        )

        trace_event = {
            "step": "retrieve_and_rerank",
            "candidates_count": len(all_candidates),
            "retrieval": result.stats,
            "top_candidates": [c.get("entry", "") for c in all_candidates[:5]]
        }

        tools = list(state.get("tools_used", []))
        if "Hybrid" not in tools:
            tools.append("Hybrid")
        if result.stats.get("graph_carried") and "Graph" not in tools:
            tools.append("Graph")

        return {
            "retrieved_candidates": all_candidates,
            "tools_used": tools,
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

        tools = list(state.get("tools_used", []))
        tools.append("Rewriter")

        return {
            "current_query": new_query,
            "retry_count": retry_count,
            "needs_rewrite": False,
            "tools_used": tools,
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_generate_answer(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        raw_query = state.get("raw_query", "")

        # gate only top 15 chunks after hybrid-retrieval
        effective_laws = state.get("retrieved_candidates", [])[:15]

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

        tools = list(state.get("tools_used", []))
        if "Synthesizer" not in tools:
            tools.append("Synthesizer")

        return {
            "synthesized_answer": synth_res.get("direct_answer", ""),
            "direct_answer": synth_res.get("direct_answer", ""),
            "decisive_quotes": synth_res.get("decisive_quotes", []),
            "applicable_laws": synth_res.get("applicable_laws", []),
            "exceptions": synth_res.get("exceptions_or_conditions", ""),
            "status": synth_res.get("status", "COMPLIANT"),
            "issues_breakdown": synth_res.get("issues_breakdown", []),
            "tools_used": tools,
            "trace_events": state.get("trace_events", []) + [trace_event]
        }

    def _node_guardrail(self, state: AgenticRAGState) -> Dict[str, Any]:
        case_id = state.get("case_id", 0)
        direct_answer = state.get("direct_answer", "")
        candidates = state.get("retrieved_candidates", [])

        verdict = GroundingGuardrail.audit(direct_answer, candidates)
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

        tools = list(state.get("tools_used", []))
        if "Guardrail" not in tools:
            tools.append("Guardrail")

        return {
            "guardrail_verdict": verdict,
            "tools_used": tools,
            "trace_events": state.get("trace_events", []) + [{"step": "guardrail", "verdict": verdict}]
        }

    def invoke(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Entrypoint for executing the LangGraph Agentic Workflow on a case item."""
        start_time = time.time()
        raw_fact = case.get("question", "")
        name = case.get("asker", "ผู้สอบถาม")

        active_org = str(case.get("org_id") or os.getenv("DEFAULT_ORG_ID", "DGA"))
        initial_state: AgenticRAGState = {
            "case_id": case.get("id", 0),
            "raw_query": raw_fact,
            "current_query": raw_fact,
            "name": str(name),
            "org_id": active_org,
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
            "issues_breakdown": [],
            "tools_used": [],
            "trace_events": []
        }

        final_state = self.app.invoke(initial_state)
        elapsed = time.time() - start_time
        retrieved_laws = final_state.get("retrieved_candidates", [])

        self.logger.log_summary(
            case_id=case.get("id", 0),
            status=final_state.get("status", "COMPLIANT"),
            retries=final_state.get("retry_count", 0),
            chunks_count=len(retrieved_laws),
            tools_used=final_state.get("tools_used", []),
            duration=elapsed
        )

        issues_breakdown = final_state.get("issues_breakdown", [])

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
                "issues_breakdown": issues_breakdown,
                "guardrail_verdict": final_state.get("guardrail_verdict", {})
            },
            "issues_breakdown": issues_breakdown,
            "retrieved_laws": retrieved_laws,
            "retrieved_facts": [],
            "used_laws": retrieved_laws,
            "used_facts": [],
            "crag_meta": {
                "issues": final_state.get("issues", []),
                "retries": final_state.get("retry_count", 0),
                "complete": True
            }
        }
