# -*- coding: utf-8 -*-
"""
core/agent/state.py

LangGraph AgentState schema for Thai Procurement LegalGraphRAG.
Defines the state passed across Orchestrator, Tool Execution, and Guardrail nodes.
"""

from typing import Annotated, Sequence, TypedDict, List, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merges two dictionaries."""
    res = dict(a or {})
    res.update(b or {})
    return res


def append_unique_contexts(
    existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Appends newly retrieved context items while deduplicating by ID or entry."""
    seen = {item.get("id") or item.get("clause_id") or item.get("entry") for item in existing if item}
    res = list(existing or [])
    for it in (new_items or []):
        key = it.get("id") or it.get("clause_id") or it.get("entry")
        if key and key not in seen:
            res.append(it)
            seen.add(key)
        elif not key:
            res.append(it)
    return res


class LegalAgentState(TypedDict):
    """Full execution state for the LangGraph Legal Orchestrator."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    org_id: str
    mode: str  # "fast" or "deep"
    retrieved_context: Annotated[List[Dict[str, Any]], append_unique_contexts]
    citations_used: Annotated[List[str], operator.add]
    guardrail_verdict: Dict[str, Any]
    final_response: Dict[str, Any]


class AgenticRAGState(TypedDict):
    """Detailed state machine schema for Procurement Agentic RAG workflow."""
    case_id: Any
    raw_query: str
    current_query: str
    name: str
    features: Dict[str, Any]
    issues: List[Dict[str, Any]]
    retrieved_candidates: List[Dict[str, Any]]
    passed_chunks: List[Dict[str, Any]]
    dropped_chunks: List[Dict[str, Any]]
    retry_count: int
    max_retries: int
    needs_rewrite: bool
    synthesized_answer: str
    direct_answer: str
    decisive_quotes: List[str]
    applicable_laws: List[str]
    exceptions: str
    status: str
    guardrail_verdict: Dict[str, Any]
    issues_breakdown: List[Dict[str, Any]]
    tools_used: List[str]
    trace_events: List[Dict[str, Any]]

