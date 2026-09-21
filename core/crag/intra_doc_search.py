# -*- coding: utf-8 -*-
"""
intra_doc_search.py

Performs high-precision, document-constrained re-retrieval (Targeted Intra-Document Scan).
When the Auditor flags that an initial answer is incomplete or missing a specific
sub-issue, condition, or threshold, this module searches strictly within the candidate .md
statutes and regulations that were already retrieved in Step 2.

Zero network/API latency: operates purely in-memory using the pre-loaded Graph DB.
"""

import os
import re
import ast
from typing import List, Dict, Any, Set, Optional, Tuple
import numpy as np

from core.graph_construct.graph_db import GraphDBManager
from core.graph_construct.feature_graph import get_embedding
from core.graph_construct.hybrid_reranker import get_reranker, expand_numeric_query


def normalize_doc_name(raw_name: str) -> str:
    """Normalize a document name or path to a clean title for matching."""
    if not raw_name:
        return ""
    base = os.path.basename(str(raw_name).strip())
    base = re.sub(r"\.md$", "", base, flags=re.IGNORECASE)
    # Remove leading subfolder prefixes if any
    base = re.sub(r"^.*?typhoon_ocr[/\\]+", "", base)
    base = re.sub(r"\s+", " ", base)
    return base.strip()


def extract_candidate_docs(candidate_laws: List[Dict[str, Any]]) -> List[str]:
    """
    Extract unique normalized document titles from retrieved candidate laws.
    """
    doc_titles = []
    seen = set()

    for law in candidate_laws:
        # 1. Check related_laws list
        rel = law.get("related_laws")
        if isinstance(rel, str):
            try:
                rel = ast.literal_eval(rel)
            except Exception:
                rel = [rel]
        if isinstance(rel, list) and rel:
            first_elem = rel[0]
            if isinstance(first_elem, str) and (".md" in first_elem or "พ.ศ." in first_elem):
                norm = normalize_doc_name(first_elem)
                if norm and norm not in seen:
                    doc_titles.append(norm)
                    seen.add(norm)

        # 2. Check entry / id
        entry = law.get("entry") or law.get("id") or ""
        norm_entry = normalize_doc_name(entry.split("|")[0])
        if norm_entry and norm_entry not in seen:
            doc_titles.append(norm_entry)
            seen.add(norm_entry)

    return doc_titles


def is_chunk_in_target_docs(law_node: Dict[str, Any], target_docs: List[str]) -> bool:
    """Check if a law node belongs to any of the target documents."""
    if not target_docs:
        return False

    # Check related_laws
    rel = law_node.get("related_laws")
    if isinstance(rel, str):
        try:
            rel = ast.literal_eval(rel)
        except Exception:
            rel = [rel]
    if isinstance(rel, list) and rel:
        doc_raw = normalize_doc_name(rel[0])
        for td in target_docs:
            if td in doc_raw or doc_raw in td:
                return True

    # Check entry
    entry_raw = normalize_doc_name(law_node.get("entry", "").split("|")[0])
    for td in target_docs:
        if td in entry_raw or entry_raw in td:
            return True

    return False


def intra_doc_search(
    candidate_laws: List[Dict[str, Any]],
    missing_aspect: str,
    top_k: int = 3,
    min_rerank_threshold: float = 0.15
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Perform targeted search strictly within the candidate documents for a missing aspect.

    Returns:
        (matching_law_nodes, is_found)
        - matching_law_nodes: List of law dicts matching the missing aspect.
        - is_found: True if at least one solid match exists, False if the target documents
          genuinely lack the requested statutory provision.
    """
    if not missing_aspect or len(missing_aspect.strip()) < 3:
        return [], False

    target_docs = extract_candidate_docs(candidate_laws)
    if not target_docs:
        return [], False

    db = GraphDBManager.get_db()
    all_laws = db.get_nodes_by_type("Laws")
    if not all_laws:
        return [], False

    # 1. Filter chunks belonging ONLY to the candidate documents
    scoped_chunks = [l for l in all_laws if is_chunk_in_target_docs(l, target_docs)]
    if not scoped_chunks:
        return [], False

    # Exclude chunks that are already in candidate_laws[:5]
    existing_ids = {str(c.get("id")) for c in candidate_laws[:5] if c.get("id")}
    candidate_pool = [c for c in scoped_chunks if str(c.get("id")) not in existing_ids]
    if not candidate_pool:
        candidate_pool = scoped_chunks

    # 2. Compute query expansion for numbers/Thai numerals
    expanded_query = expand_numeric_query(missing_aspect)

    # 3. Dense Cosine Similarity
    query_emb = get_embedding(expanded_query)
    scored_candidates = []

    for c in candidate_pool:
        c_id = c.get("id")
        desc = c.get("description", "")
        # Lexical boost
        lex_score = 0.0
        for word in missing_aspect.split():
            if len(word) > 2 and word in desc:
                lex_score += 0.2

        # Dense similarity
        dense_sim = 0.0
        if c_id in db.embeddings.get("Laws", {}):
            node_vec = db.embeddings["Laws"][c_id]
            if node_vec is not None and len(node_vec) > 0 and query_emb is not None:
                dot = np.dot(query_emb, node_vec)
                norm_q = np.linalg.norm(query_emb)
                norm_n = np.linalg.norm(node_vec)
                if norm_q > 0 and norm_n > 0:
                    dense_sim = float(dot / (norm_q * norm_n))

        combined_sim = dense_sim + lex_score
        scored_candidates.append((c, combined_sim))

    # Sort top candidates
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [x[0] for x in scored_candidates[:min(15, len(scored_candidates))]]

    # 4. GPU Cross-Encoder Reranking
    reranker = get_reranker()
    matched_nodes = []

    if reranker and reranker.model is not None and top_candidates:
        rerank_input = [
            {"id": c.get("id"), "description": c.get("description", ""), "data": c}
            for c in top_candidates
        ]
        reranked = reranker.rerank(
            expanded_query,
            rerank_input,
            top_k=top_k,
            threshold=min_rerank_threshold
        )
        for item in reranked:
            c_data = item.get("data", {})
            c_data["rerank_score"] = item.get("rerank_score", 0.0)
            matched_nodes.append(c_data)
    else:
        # Fallback to top cosine candidates
        matched_nodes = top_candidates[:top_k]

    # 5. Check if any chunk genuinely matches
    is_found = len(matched_nodes) > 0 and (
        any(m.get("rerank_score", 1.0) >= min_rerank_threshold for m in matched_nodes)
    )

    return matched_nodes, is_found
