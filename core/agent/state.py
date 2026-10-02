# -*- coding: utf-8 -*-
"""
core/agent/state.py

LangGraph state schema for the Thai Procurement agentic RAG workflow (core/agent/workflow.py).
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgenticRAGState(TypedDict):
    """Detailed state machine schema for Procurement Agentic RAG workflow."""
    case_id: Any
    raw_query: str
    current_query: str
    name: str
    org_id: str
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

