# -*- coding: utf-8 -*-
"""
core/graph_construct/citation_linker.py

Legal Citation & Relationship Linker for Thai Procurement Knowledge Graph.
Extracts and builds rich, deterministic graph edges across sections:
1. Cross-Statute Empowered Edges: (กฎกระทรวง/ระเบียบ) -[EMPOWERED_BY]-> (พ.ร.บ. มาตรา X)
2. Reverse Empowered Edges: (พ.ร.บ. มาตรา X) -[EMPOWERS]-> (กฎกระทรวง/ระเบียบ)
3. Inter-Section Citation Edges: (Section A) -[CITES]-> (Section B) and (Section B) -[CITED_BY]-> (Section A)
4. Sequential Section Edges: (Section N) -[NEXT_SECTION]-> (Section N+1) and (Section N+1) -[PREV_SECTION]-> (Section N)
"""

import re
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


class LegalCitationLinker:
    """Extracts and injects cross-statute and intra-statute citation edges into GraphDB."""

    ACT_TITLE = "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560"

    @classmethod
    def link_citations(cls, db) -> Dict[str, int]:
        """
        Scans all Law nodes in GraphDB and connects rich citation edges.
        
        Args:
            db: GraphDB instance from GraphDBManager.get_db()
            
        Returns:
            Dictionary with counts of edges added per relation type.
        """
        # 1. Index all Law nodes
        law_nodes = {}
        sec_lookup: Dict[Tuple[str, bool, int], str] = {}
        doc_sections: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        for node_id, node_info in db.nodes_data.items():
            if node_info.get("type") == "Laws":
                law_data = node_info.get("data", {})
                entry = law_data.get("entry", "")
                desc = law_data.get("description", "")
                doc_name = entry.split("|")[0].strip() if "|" in entry else entry
                sec_part = entry.split("|")[1].strip() if "|" in entry else ""

                law_nodes[node_id] = {
                    "entry": entry,
                    "doc_name": doc_name,
                    "sec_part": sec_part,
                    "description": desc,
                }

                # Parse section number
                norm_sec = sec_part.translate(TH_TO_AR)
                m = re.search(r"(\d+)", norm_sec)
                if m:
                    num = int(m.group(1))
                    is_clause = "ข้อ" in sec_part
                    sec_lookup[(doc_name, is_clause, num)] = node_id
                    doc_sections[doc_name].append((num, node_id))

        counts = {
            "CITES": 0,
            "CITED_BY": 0,
            "EMPOWERED_BY": 0,
            "EMPOWERS": 0,
            "NEXT_SECTION": 0,
            "PREV_SECTION": 0,
        }

        # 2. Extract Cross-Statute and Inter-Section Citation Edges
        for node_id, info in law_nodes.items():
            doc_name = info["doc_name"]
            text_norm = info["description"].translate(TH_TO_AR)
            is_intro = any(k in info["sec_part"] for k in ["คำนำ", "บทนำ", "ทั่วไป", "หมายเหตุ"])

            # 2.1 Preamble / Empowering Citations (อาศัยอำนาจตามความในมาตรา X แห่งพระราชบัญญัติ...)
            if is_intro or "อาศัยอำนาจ" in text_norm:
                m_emp = re.finditer(r"(?:มาตรา|ม\.)\s*(\d+)", text_norm)
                for match in m_emp:
                    act_sec_num = int(match.group(1))
                    target_act_id = sec_lookup.get((cls.ACT_TITLE, False, act_sec_num))
                    if target_act_id and target_act_id != node_id:
                        db.add_edge(node_id, target_act_id, "EMPOWERED_BY", {"section": act_sec_num})
                        db.add_edge(target_act_id, node_id, "EMPOWERS", {"section": act_sec_num})
                        counts["EMPOWERED_BY"] += 1
                        counts["EMPOWERS"] += 1

            # 2.2 Direct Section Citations: 'ตามมาตรา X' or 'มาตรา X'
            for match in re.finditer(r"(?:ตาม|แห่ง)?\s*(?:มาตรา|ม\.)\s*(\d+)", text_norm):
                sec_num = int(match.group(1))
                # If current doc is the Act itself, target is within Act. If subordinate doc, target is also the Act!
                target_id = sec_lookup.get((cls.ACT_TITLE, False, sec_num))
                if target_id and target_id != node_id:
                    db.add_edge(node_id, target_id, "CITES", {"section": sec_num})
                    db.add_edge(target_id, node_id, "CITED_BY", {"section": sec_num})
                    counts["CITES"] += 1
                    counts["CITED_BY"] += 1

            # 2.3 Direct Clause Citations: 'ตามข้อ Y' or 'ข้อ Y' within the same document
            for match in re.finditer(r"(?:ตาม|แห่ง)?\s*(?:ข้อ|ระเบียบฯ\s*ข้อ)\s*(\d+)", text_norm):
                clause_num = int(match.group(1))
                target_id = sec_lookup.get((doc_name, True, clause_num))
                if target_id and target_id != node_id:
                    db.add_edge(node_id, target_id, "CITES", {"clause": clause_num})
                    db.add_edge(target_id, node_id, "CITED_BY", {"clause": clause_num})
                    counts["CITES"] += 1
                    counts["CITED_BY"] += 1

        # 3. Add Sequential Section Edges (NEXT_SECTION / PREV_SECTION)
        for doc_name, sec_list in doc_sections.items():
            # Deduplicate by section number and sort
            unique_secs = {}
            for num, nid in sec_list:
                if num not in unique_secs:
                    unique_secs[num] = nid
            sorted_nums = sorted(unique_secs.keys())
            for i in range(len(sorted_nums) - 1):
                cur_num = sorted_nums[i]
                next_num = sorted_nums[i + 1]
                # Only link if immediately adjacent
                if next_num == cur_num + 1:
                    u = unique_secs[cur_num]
                    v = unique_secs[next_num]
                    db.add_edge(u, v, "NEXT_SECTION", {"from": cur_num, "to": next_num})
                    db.add_edge(v, u, "PREV_SECTION", {"from": next_num, "to": cur_num})
                    counts["NEXT_SECTION"] += 1
                    counts["PREV_SECTION"] += 1

        print(
            f"[CitationLinker] Success: Added {counts['EMPOWERED_BY']} EMPOWERED_BY, "
            f"{counts['EMPOWERS']} EMPOWERS, {counts['CITES']} CITES, "
            f"{counts['CITED_BY']} CITED_BY, {counts['NEXT_SECTION']} NEXT_SECTION edges."
        )
        return counts

    @classmethod
    def traverse_legal_neighbors(
        cls,
        db,
        seed_law_ids: List[str],
        relation_types: Optional[List[str]] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Traverses statutory relationship edges starting from high-confidence seed law nodes.
        
        Args:
            db: GraphDB instance.
            seed_law_ids: List of law node IDs or entries.
            relation_types: Edge types to traverse. Default: ['EMPOWERS', 'CITED_BY', 'CITES', 'NEXT_SECTION'].
            top_k: Maximum neighbor nodes to return.
            
        Returns:
            List of neighbor law candidate dictionaries.
        """
        if not seed_law_ids:
            return []

        rel_filter = set(relation_types or ["EMPOWERS", "CITED_BY", "CITES", "NEXT_SECTION"])
        discovered = {}

        # Resolve node IDs if entries or descriptions were passed
        resolved_ids = set()
        for s in seed_law_ids:
            if s in db.nodes_data:
                resolved_ids.add(s)
            else:
                for nid, ninfo in db.nodes_data.items():
                    if ninfo.get("type") == "Laws" and (s in ninfo.get("data", {}).get("entry", "")):
                        resolved_ids.add(nid)

        for seed_id in resolved_ids:
            if seed_id not in db.graph:
                continue

            # Candidate neighbors (both outgoing and incoming edges)
            adjacent_nodes = []
            if hasattr(db.graph, "successors"):
                for succ in db.graph.successors(seed_id):
                    adjacent_nodes.append((seed_id, succ, succ, "out"))
            if hasattr(db.graph, "predecessors"):
                for pred in db.graph.predecessors(seed_id):
                    adjacent_nodes.append((pred, seed_id, pred, "in"))

            for u, v, target_nid, direction in adjacent_nodes:
                if target_nid in resolved_ids or target_nid in discovered:
                    continue

                edata = db.graph.get_edge_data(u, v) or {}
                # Handle MultiDiGraph dictionary of edges: {0: {...}, 1: {...}}
                edge_items = edata.values() if isinstance(edata, dict) and any(isinstance(k, int) for k in edata.keys()) else [edata]

                for edge in edge_items:
                    rel = edge.get("relation_type") or edge.get("type") if isinstance(edge, dict) else None
                    if rel and (rel in rel_filter or "ALL" in rel_filter):
                        ninfo = db.nodes_data.get(target_nid)
                        if ninfo and ninfo.get("type") == "Laws":
                            data = ninfo.get("data", {})
                            entry = data.get("entry", "")
                            discovered[target_nid] = {
                                "id": target_nid,
                                "entry": entry,
                                "description": data.get("description", ""),
                                "crimes": data.get("crimes", []),
                                "judge_dep": data.get("judge_dep", []),
                                "related_laws": data.get("related_laws", []),
                                "traversed_via": f"{rel} ({direction})",
                                "source_seed": db.nodes_data.get(seed_id, {}).get("data", {}).get("entry", ""),
                                "rerank_score": 0.88  # High deterministic graph traversal prior
                            }
                            break  # Found match for this neighbor node

        results = list(discovered.values())
        return results[:top_k]

    @classmethod
    def search_by_entity(cls, db, query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Deterministic Graph Node lookup by statutory section/clause entities extracted from query text.
        """
        if not query_text or not db or not hasattr(db, "nodes_data"):
            return []

        text_norm = query_text.translate(TH_TO_AR)
        extracted_sections = set(int(m.group(1)) for m in re.finditer(r"(?:มาตรา|ม\.)\s*(\d+)", text_norm))
        extracted_clauses = set(int(m.group(1)) for m in re.finditer(r"(?:ข้อ|ระเบียบฯ\s*ข้อ)\s*(\d+)", text_norm))

        if not extracted_sections and not extracted_clauses:
            return []

        matches = []
        for nid, ninfo in db.nodes_data.items():
            if ninfo.get("type") != "Laws":
                continue
            data = ninfo.get("data", {})
            entry = data.get("entry", "")
            sec_part = entry.split("|")[1].strip() if "|" in entry else ""
            norm_sec = sec_part.translate(TH_TO_AR)
            m = re.search(r"(\d+)", norm_sec)
            if not m:
                continue
            num = int(m.group(1))

            matched = False
            if "มาตรา" in sec_part and num in extracted_sections:
                matched = True
            elif "ข้อ" in sec_part and num in extracted_clauses:
                matched = True

            if matched:
                matches.append({
                    "id": nid,
                    "entry": entry,
                    "description": data.get("description", ""),
                    "crimes": data.get("crimes", []),
                    "judge_dep": data.get("judge_dep", []),
                    "related_laws": data.get("related_laws", []),
                    "traversed_via": f"Entity-Exact (Section {num})",
                    "rerank_score": 0.95
                })

        # Also expand 1 hop from these exact entity matches
        if matches:
            seed_ids = [m["id"] for m in matches[:2]]
            expansion = cls.traverse_legal_neighbors(db, seed_ids, top_k=top_k)
            for exp in expansion:
                if not any(m["id"] == exp["id"] for m in matches):
                    matches.append(exp)

        return matches[:top_k]

    @classmethod
    def search_by_graph_features(
        cls,
        db,
        features: Optional[Dict[str, Any]] = None,
        issues: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Topological graph search traversing from Crime/Feature topic nodes to Law nodes.
        """
        if not db or not hasattr(db, "nodes_data"):
            return []

        keywords = set()
        if features:
            for k in ["criminal_acts", "victim_property_details"]:
                for val in features.get(k, []):
                    if val and len(val.strip()) > 3:
                        keywords.add(val.strip())
        if issues:
            for iss in issues:
                for kw in iss.get("search_keywords", []):
                    if len(kw.strip()) > 2:
                        keywords.add(kw.strip())

        if not keywords:
            return []

        # Find matching Crime nodes in the Graph
        matched_crime_nodes = []
        for nid, ninfo in db.nodes_data.items():
            if ninfo.get("type") == "Crimes":
                crime_name = ninfo.get("data", {}).get("name", "")
                if any(kw in crime_name or crime_name in kw for kw in keywords):
                    matched_crime_nodes.append(nid)

        law_nodes_found = {}
        for cnid in matched_crime_nodes[:5]:
            if cnid not in db.graph:
                continue
            neighbors = []
            if hasattr(db.graph, "successors"):
                neighbors.extend(db.graph.successors(cnid))
            if hasattr(db.graph, "predecessors"):
                neighbors.extend(db.graph.predecessors(cnid))

            for n_id in neighbors:
                ninfo = db.nodes_data.get(n_id)
                if ninfo and ninfo.get("type") == "Laws" and n_id not in law_nodes_found:
                    data = ninfo.get("data", {})
                    law_nodes_found[n_id] = {
                        "id": n_id,
                        "entry": data.get("entry", ""),
                        "description": data.get("description", ""),
                        "crimes": data.get("crimes", []),
                        "judge_dep": data.get("judge_dep", []),
                        "related_laws": data.get("related_laws", []),
                        "traversed_via": f"Graph-Topic ({cnid})",
                        "rerank_score": 0.86
                    }

        return list(law_nodes_found.values())[:top_k]

    @classmethod
    def graph_search_backup(
        cls,
        db,
        query: str,
        features: Optional[Dict[str, Any]] = None,
        issues: Optional[List[Dict[str, Any]]] = None,
        seed_laws: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Comprehensive Knowledge Graph Backup Search.
        Activates when hybrid retrieval returns low confidence, fails gatekeeper,
        or requires statutory multi-hop recovery.
        """
        combined = []
        seen_ids = set()

        # 1. Exact entity section extraction & traversal
        entity_matches = cls.search_by_entity(db, query, top_k=top_k)
        for em in entity_matches:
            if em["id"] not in seen_ids:
                seen_ids.add(em["id"])
                combined.append(em)

        # 2. Multi-hop Statutory Traversal from any seed laws available
        if seed_laws:
            seed_ids = [s.get("id") or s.get("entry") for s in seed_laws[:4] if (s.get("id") or s.get("entry"))]
            traversed = cls.traverse_legal_neighbors(db, seed_ids, top_k=top_k)
            for tr in traversed:
                if tr["id"] not in seen_ids:
                    seen_ids.add(tr["id"])
                    combined.append(tr)

        # 3. Topic & Crime Graph Topology search
        feature_matches = cls.search_by_graph_features(db, features=features, issues=issues, top_k=top_k)
        for fm in feature_matches:
            if fm["id"] not in seen_ids:
                seen_ids.add(fm["id"])
                combined.append(fm)

        return combined[:top_k]
