# -*- coding: utf-8 -*-
"""
core/agent/tools.py

Standardized Tool definitions for the LangGraph Orchestrator Agent.
Wraps Tri-Store operations (PostgreSQL SSOT, Qdrant VectorDB, Neo4j GraphDB)
into high-level tools that the LLM Orchestrator can autonomously invoke.
"""

import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

from core.database import StorageManager


def _get_storage() -> StorageManager:
    return StorageManager.get_instance()


_local_law_cache = None

def _get_local_laws():
    global _local_law_cache
    if _local_law_cache is None:
        path = "./datas/law_to_crime.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _local_law_cache = json.load(f)
        else:
            _local_law_cache = []
    return _local_law_cache


@tool
def exact_statute_lookup(section: str, doc_keyword: Optional[str] = "") -> str:
    """
    [EXACT LOOKUP] Search for the exact legal text of a specific Section (มาตรา) or Regulation Clause (ข้อ).
    Use this tool when the query mentions a specific section number (e.g. "มาตรา 56", "ข้อ 79").
    
    Args:
        section: Section number or clause number (e.g. "56", "มาตรา 56", "ข้อ 79").
        doc_keyword: Optional document keyword (e.g. "พระราชบัญญัติ", "ระเบียบกระทรวงการคลัง").
    """
    import re
    norm = str(section).translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    m = re.search(r"(\d+)", norm)
    if not m:
        return f"ไม่พบตัวเลขมาตราหรือข้อในคำค้น: '{section}'"
    
    num = int(m.group(1))
    is_clause = "ข้อ" in str(section)

    records = []
    try:
        storage = _get_storage()
        if is_clause:
            records = storage.pg.lookup_clause(doc_keyword or "", num)
        else:
            records = storage.pg.lookup_section(doc_keyword or "", num)
    except Exception:
        records = []

    # Fallback to local law_to_crime corpus if PostgreSQL is not seeded
    if not records:
        laws = _get_local_laws()
        target_pattern = rf"ข้อ\s*{num}\b" if is_clause else rf"(มาตรา|ม\.)\s*{num}\b"
        ar_to_th = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
        th_num = str(num).translate(ar_to_th)
        target_th_pattern = rf"ข้อ\s*{th_num}\b" if is_clause else rf"(มาตรา|ม\.)\s*{th_num}\b"
        
        for item in laws:
            entry = item.get("id", "")
            if (re.search(target_pattern, entry) or re.search(target_th_pattern, entry)):
                if not doc_keyword or (doc_keyword in entry):
                    items_list = item.get("items", [])
                    content = items_list[0].get("text", "") if items_list else ""
                    records.append({
                        "clause_id": entry,
                        "doc_title": entry.split("|")[0].strip() if "|" in entry else entry,
                        "entry": entry,
                        "content_thai": content[:1200],
                        "related_laws": items_list[0].get("related_laws", []) if items_list else [],
                        "topics": items_list[0].get("crime", []) if items_list else []
                    })
                    break

    if not records:
        return f"ไม่พบข้อมูลสำหรับ '{section}' ในฐานข้อมูลพระราชบัญญัติหรือระเบียบ"

    top = records[0]
    res = {
        "clause_id": top.get("clause_id"),
        "doc_title": top.get("doc_title"),
        "entry": top.get("entry"),
        "content_thai": top.get("content_thai", "")[:1200],
        "related_laws": top.get("related_laws", []),
        "topics": top.get("topics", [])
    }
    return json.dumps(res, ensure_ascii=False)


@tool
def hybrid_statutory_search(query: str, top_k: int = 4) -> str:
    """
    [CONCEPTUAL & SEMANTIC SEARCH] Search procurement laws, ministerial regulations, and circulars
    using dense semantic vectors + sparse Thai BM25 with Reciprocal Rank Fusion (RRF).
    Use this tool when the inquiry is descriptive or asks for rules without knowing the specific section number.

    Args:
        query: Descriptive inquiry in Thai (e.g. "การจ้างที่ปรึกษาโดยวิธีคัดเลือก", "หลักเกณฑ์การแบ่งซื้อแบ่งจ้าง").
        top_k: Number of relevant clauses to retrieve (default: 4).
    """
    results = []
    try:
        from core.graph_construct.feature_graph import get_embedding
        storage = _get_storage()
        emb = get_embedding(query.strip())
        results = storage.hybrid_search_clauses(
            query_text=query.strip(),
            query_dense=emb,
            top_k=top_k
        )
    except Exception:
        results = []

    # Fallback to local feature graph & Thai BM25 index if Qdrant/PostgreSQL not yet seeded
    if not results:
        try:
            from core.graph_construct.feature_graph import search_similar_nodes_direct
            _, local_laws = search_similar_nodes_direct(None, None, query_text=query.strip(), top_k=top_k)
            for r in local_laws[:top_k]:
                results.append({
                    "clause_id": r.get("id"),
                    "entry": r.get("entry"),
                    "content_thai": r.get("description", ""),
                    "topics": r.get("crimes", []),
                    "score": r.get("rerank_score", 0.9)
                })
        except Exception:
            pass

    if not results:
        return "ไม่พบข้อกฎหมายหรือระเบียบที่สอดคล้องกับคำค้นหานี้"

    formatted = []
    for r in results:
        formatted.append({
            "clause_id": r.get("clause_id"),
            "entry": r.get("entry"),
            "preview": r.get("content_thai", "")[:600],
            "topics": r.get("topics", []),
            "relevance_score": round(float(r.get("score", 0.0)), 4)
        })
    return json.dumps(formatted, ensure_ascii=False)


@tool
def knowledge_graph_traversal(clause_id: str) -> str:
    """
    [GRAPH EXPANSION] Traverse the legal knowledge graph in Neo4j to find related ministerial rules,
    circular letters (หนังสือเวียน), adjacent sections, and past precedent rulings linked to this clause.
    Use this tool AFTER finding a primary statute to discover secondary implementing regulations.

    Args:
        clause_id: Unique clause identifier returned from lookup or search (e.g. "ACT_2560_SEC_56").
    """
    storage = _get_storage()
    graph_ctx = storage.traverse_clause_graph(clause_id)
    summary = {
        "adjacent_sections": [f"{a.get('entry', '')} ({a.get('clause_id', '')})" for a in graph_ctx.get("adjacent_sections", [])[:3]],
        "cited_clauses": [f"{c.get('entry', '')}: {c.get('quote', '')}" for c in graph_ctx.get("cited_clauses", [])[:3]],
        "subordinate_rules": [f"{s.get('document_title', '')} - {s.get('entry', '')}" for s in graph_ctx.get("subordinate_laws", [])[:4]],
        "precedent_cases": [f"{p.get('case_id', '')}: {p.get('question', '')[:100]}" for p in graph_ctx.get("related_cases", [])[:2]],
    }
    return json.dumps(summary, ensure_ascii=False)


@tool
def verify_procurement_threshold(
    procurement_item: str,
    estimated_budget: float,
    proposed_method: str,
    justification_reason: Optional[str] = None
) -> str:
    """
    [FAST COMPLIANCE AUDIT] Immediately check if a proposed procurement method and budget amount
    comply with statutory thresholds under Thai Procurement Law (e.g. <= 500,000 THB for specific selection).
    Runs deterministically in ~2ms without LLM latency.

    Args:
        procurement_item: Description of item or service to purchase/hire.
        estimated_budget: Estimated budget in Thai Baht (THB).
        proposed_method: Method name (e.g. "เฉพาะเจาะจง", "e-bidding", "คัดเลือก").
        justification_reason: Optional reason (e.g. "จำเป็นเร่งด่วน", "วงเงินไม่เกิน 500,000 บาท").
    """
    from core.mcp_service import ProcurementService
    service = ProcurementService.get_instance()
    verdict = service.verify_compliance(
        procurement_item=procurement_item,
        estimated_budget=estimated_budget,
        proposed_method=proposed_method,
        justification_reason=justification_reason
    )
    return json.dumps(verdict, ensure_ascii=False)


LEGAL_TOOLS = [
    exact_statute_lookup,
    hybrid_statutory_search,
    knowledge_graph_traversal,
    verify_procurement_threshold,
]
