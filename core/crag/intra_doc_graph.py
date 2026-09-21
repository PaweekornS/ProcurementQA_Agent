# -*- coding: utf-8 -*-
"""
intra_doc_graph.py

Implements Intra-Document Legal Graph Construction and Traversal for CRAG.
Builds and traverses in-memory graph edges within each legal statute/regulation:
1. CITES_CLAUSE / CITED_BY: Cross-statutory citations (e.g. "ตามมาตรา 56", "ตามข้อ 22")
2. ADJACENT_SECTION: Sequential legal clauses (e.g. Section 25 <-> Section 26, Section 27)
3. SAME_CHAPTER: Structural grouping of sections under the same chapter (หมวด)

Zero network latency: Operates directly on the in-memory NetworkX MultiDiGraph in RAM.
"""

import os
import re
import ast
from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np

from core.graph_construct.graph_db import GraphDBManager
from core.graph_construct.feature_graph import get_embedding
from core.graph_construct.hybrid_reranker import get_reranker, expand_numeric_query

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def thai_to_arabic(text: str) -> str:
    """Translate Thai numeral digits to Arabic numeral digits."""
    return str(text).translate(TH_TO_AR)


def normalize_doc_name(raw_name: str) -> str:
    """Normalize a document name or path to a clean title for matching."""
    if not raw_name:
        return ""
    base = os.path.basename(str(raw_name).strip())
    base = re.sub(r"\.md$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^.*?typhoon_ocr[/\\]+", "", base)
    base = re.sub(r"\s+", " ", base)
    return base.strip()


def extract_doc_and_sections(law_node: Dict[str, Any]) -> Tuple[str, Set[int], Set[int], Optional[int]]:
    """
    Extracts:
        - doc_name: normalized document title
        - sections: set of Section numbers (มาตรา) as integers
        - clauses: set of Regulation Clause numbers (ข้อ) as integers
        - chapter: Chapter number (หมวด) as integer if present
    """
    doc_name = ""
    rel = law_node.get("related_laws")
    if isinstance(rel, str):
        try:
            rel = ast.literal_eval(rel)
        except Exception:
            rel = [rel]
    if isinstance(rel, list) and rel:
        first_elem = rel[0]
        if isinstance(first_elem, str) and (".md" in first_elem or "พ.ศ." in first_elem):
            doc_name = normalize_doc_name(first_elem)

    entry = str(law_node.get("entry", ""))
    if not doc_name:
        doc_name = normalize_doc_name(entry.split("|")[0])

    combined_text = f"{entry} {str(law_node.get('description', ''))[:800]}"
    combined_norm = thai_to_arabic(combined_text)

    # Extract sections (มาตรา)
    sections = set()
    for m in re.finditer(r"มาตรา\s*(\d+)", combined_norm):
        try:
            sections.add(int(m.group(1)))
        except ValueError:
            pass

    # Extract clauses (ข้อ)
    clauses = set()
    for m in re.finditer(r"ข้อ\s*(\d+)", combined_norm):
        try:
            clauses.add(int(m.group(1)))
        except ValueError:
            pass

    # Extract chapter (หมวด)
    chapter = None
    ch_m = re.search(r"หมวด\s*(\d+)", combined_norm)
    if ch_m:
        try:
            chapter = int(ch_m.group(1))
        except ValueError:
            pass

    return doc_name, sections, clauses, chapter


def extract_citations(text: str) -> Tuple[Set[int], Set[int]]:
    """
    Extract citations like 'ตามมาตรา 56', 'ตามข้อ 22', 'ตามความในมาตรา 100'.
    Returns (cited_sections, cited_clauses).
    """
    text_norm = thai_to_arabic(str(text))
    cited_sections = set()
    cited_clauses = set()

    for m in re.finditer(r"ตาม(?:ความใน)?มาตรา\s*(\d+)", text_norm):
        try:
            cited_sections.add(int(m.group(1)))
        except ValueError:
            pass

    for m in re.finditer(r"ตาม(?:ความใน)?ข้อ\s*(\d+)", text_norm):
        try:
            cited_clauses.add(int(m.group(1)))
        except ValueError:
            pass

    return cited_sections, cited_clauses


def build_intra_doc_relations(db) -> Dict[str, int]:
    """
    Builds and caches intra-document structural relations in the in-memory graph.
    Runs once upon initialization (takes < 50 ms).
    """
    if getattr(db, "_intra_doc_built", False):
        return getattr(db, "_intra_doc_stats", {})

    all_laws = db.get_nodes_by_type("Laws")
    if not all_laws:
        return {}

    # 1. Group law nodes by document
    doc_nodes: Dict[str, List[Dict[str, Any]]] = {}
    node_meta: Dict[str, Dict[str, Any]] = {}

    for law in all_laws:
        nid = str(law.get("id", ""))
        if not nid:
            continue
        # Skip coarse multi-page raw chunk bundles (_p...) from intra-doc structural relations
        is_page_chunk = bool(re.search(r"_p\d+", nid))
        doc_name, secs, clss, chap = extract_doc_and_sections(law)
        if not doc_name or is_page_chunk:
            continue

        doc_nodes.setdefault(doc_name, []).append(law)
        node_meta[nid] = {
            "doc": doc_name,
            "sections": secs,
            "clauses": clss,
            "chapter": chap,
            "node": law
        }

    edge_counts = {"ADJACENT_SECTION": 0, "CITES_CLAUSE": 0, "SAME_CHAPTER": 0}


    # 2. Build edges within each document
    for doc_name, nodes in doc_nodes.items():
        if len(nodes) < 2:
            continue

        # A. Build Sequential Adjacency (ADJACENT_SECTION)
        # Sort nodes by their minimum clause or section number if available, else by entry/id
        def sort_key(n):
            nid = n.get("id", "")
            meta = node_meta.get(nid, {})
            nums = list(meta.get("sections", set())) + list(meta.get("clauses", set()))
            return min(nums) if nums else 999999

        sorted_nodes = sorted(nodes, key=sort_key)
        for i in range(len(sorted_nodes) - 1):
            curr_id = sorted_nodes[i].get("id")
            next_id = sorted_nodes[i + 1].get("id")
            if curr_id and next_id and curr_id != next_id:
                db.add_edge(curr_id, next_id, "ADJACENT_SECTION", {"direction": "next"})
                db.add_edge(next_id, curr_id, "ADJACENT_SECTION", {"direction": "prev"})
                edge_counts["ADJACENT_SECTION"] += 2

        # B. Build Citation Edges (CITES_CLAUSE & CITED_BY)
        # Create fast lookup: section_num -> list of node_ids, clause_num -> list of node_ids
        sec_to_nodes: Dict[int, List[str]] = {}
        cls_to_nodes: Dict[int, List[str]] = {}
        chap_to_nodes: Dict[int, List[str]] = {}

        for n in nodes:
            nid = n.get("id")
            meta = node_meta.get(nid, {})
            for s in meta.get("sections", set()):
                sec_to_nodes.setdefault(s, []).append(nid)
            for c in meta.get("clauses", set()):
                cls_to_nodes.setdefault(c, []).append(nid)
            ch = meta.get("chapter")
            if ch is not None:
                chap_to_nodes.setdefault(ch, []).append(nid)

        for n in nodes:
            src_id = n.get("id")
            desc = str(n.get("description", ""))
            cited_s, cited_c = extract_citations(desc)

            for cs in cited_s:
                target_ids = sec_to_nodes.get(cs, [])
                for tid in target_ids:
                    if tid != src_id:
                        db.add_edge(src_id, tid, "CITES_CLAUSE", {"target_section": cs})
                        db.add_edge(tid, src_id, "CITED_BY", {"source_section": cs})
                        edge_counts["CITES_CLAUSE"] += 2

            for cc in cited_c:
                target_ids = cls_to_nodes.get(cc, [])
                for tid in target_ids:
                    if tid != src_id:
                        db.add_edge(src_id, tid, "CITES_CLAUSE", {"target_clause": cc})
                        db.add_edge(tid, src_id, "CITED_BY", {"source_clause": cc})
                        edge_counts["CITES_CLAUSE"] += 2

        # C. Build Chapter Community Edges (SAME_CHAPTER)
        for ch, ch_node_ids in chap_to_nodes.items():
            if len(ch_node_ids) > 1 and len(ch_node_ids) <= 15:
                for i in range(len(ch_node_ids)):
                    for j in range(i + 1, min(i + 4, len(ch_node_ids))):
                        u, v = ch_node_ids[i], ch_node_ids[j]
                        db.add_edge(u, v, "SAME_CHAPTER", {"chapter": ch})
                        db.add_edge(v, u, "SAME_CHAPTER", {"chapter": ch})
                        edge_counts["SAME_CHAPTER"] += 2

    db._intra_doc_built = True
    db._intra_doc_stats = edge_counts
    return edge_counts


def traverse_intra_doc_graph(
    db,
    seed_laws: List[Dict[str, Any]],
    missing_aspect: str,
    top_k: int = 3,
    min_rerank_threshold: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Traverses the intra-document legal graph starting from seed laws.
    Expands 1-hop across CITES_CLAUSE, ADJACENT_SECTION, and SAME_CHAPTER relations.
    Reranks discovered neighbors using the GPU Cross-Encoder against missing_aspect.
    """
    if not seed_laws:
        return []

    # Ensure relations are built
    build_intra_doc_relations(db)

    seed_ids = {str(l.get("id")) for l in seed_laws if l.get("id")}
    candidate_neighbor_ids: Set[str] = set()

    # Traverse 1-hop for each top seed law
    for law in seed_laws[:5]:
        src_id = law.get("id")
        if not src_id:
            continue

        for rel in ["CITES_CLAUSE", "CITED_BY", "ADJACENT_SECTION", "SAME_CHAPTER"]:
            nbrs = db.get_neighbors(src_id, rel)
            for nid in nbrs:
                if nid not in seed_ids:
                    candidate_neighbor_ids.add(nid)

    if not candidate_neighbor_ids:
        return []

    # Retrieve candidate node objects
    candidate_nodes = []
    for nid in candidate_neighbor_ids:
        # Skip coarse multi-page chunks from being returned
        if re.search(r"_p\d+", str(nid)):
            continue
        ndata = db.get_node(nid)
        if ndata and (ndata.get("entry") or ndata.get("description")):
            node_dict = dict(ndata)
            node_dict["id"] = nid
            candidate_nodes.append(node_dict)



    if not candidate_nodes:
        return []

    # If missing_aspect is provided, score and rerank neighbors
    query_text = expand_numeric_query(missing_aspect) if missing_aspect else ""
    if len(query_text.strip()) > 3:
        reranker = get_reranker()
        if reranker and reranker.model is not None:
            rerank_input = [
                {"id": c.get("id"), "description": c.get("description", ""), "data": c}
                for c in candidate_nodes[:20]
            ]
            reranked = reranker.rerank(
                query_text,
                rerank_input,
                top_k=top_k,
                threshold=min_rerank_threshold
            )
            matched = []
            for item in reranked:
                c_data = item.get("data", {})
                c_data["rerank_score"] = item.get("rerank_score", 0.0)
                matched.append(c_data)
            return matched

    # Fallback to top candidates
    return candidate_nodes[:top_k]
