# -*- coding: utf-8 -*-
"""
core/graph_construct/citation_linker.py

Legal Citation & Relationship Linker for Thai Procurement Knowledge Graph.
Extracts and builds rich, deterministic graph edges across sections:
1. Cross-Statute Empowered Edges: (กฎกระทรวง/ระเบียบ) -[EMPOWERED_BY]-> (พ.ร.บ. มาตรา X)
2. Reverse Empowered Edges: (พ.ร.บ. มาตรา X) -[EMPOWERS]-> (กฎกระทรวง/ระเบียบ)
3. Inter-Section Citation Edges: (Section A) -[CITES]-> (Section B) and (Section B) -[CITED_BY]-> (Section A)
4. Sequential Section Edges: (Section N) -[NEXT_SECTION]-> (Section N+1) and (Section N+1) -[PREV_SECTION]-> (Section N)

Edge extraction (`extract_legal_edges`) is a pure function over "legal units" so the same rules
feed both the in-memory GraphDB (`link_citations`) and the Neo4j tri-store migration.
"""

import re
from typing import Dict, Any, List, Set, Tuple, Optional
from collections import defaultdict

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

ACT_TITLE = "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560"

# Canonical (forward) relation -> materialized reverse relation for the in-memory graph
REVERSE_RELATIONS = {
    "EMPOWERED_BY": "EMPOWERS",
    "CITES": "CITED_BY",
    "NEXT_SECTION": "PREV_SECTION",
}

_LEADING_HEADERS = re.compile(r"^(?:\s*\[[^\]]*\]+)+")
_PART_RE = re.compile(r"ตอนที่\s*(\d+)")
# 'ม.' is deliberately not accepted as an abbreviation: in this corpus it is overwhelmingly
# metres/millimetres inside price tables. Bracketed '[มาตรา N]' markers are structure, not citations.
_SECTION_REF = re.compile(r"(?<!\[)มาตรา\s*(\d+)")
_CLAUSE_REF = re.compile(r"(?<!\[)ข้อ\s*(\d+)")
_YEAR_RE = re.compile(r"พ\.ศ\.\s*(\d{4})")
_INTRO_LABELS = ("คำนำ", "บทนำ", "หมายเหตุ")


def parse_unit_label(label: str) -> Tuple[Optional[str], Optional[int], int, str]:
    """
    Parse a chunk label such as 'มาตรา ๕๓ (ตอนที่ 2)', 'ข้อ ๗๙' or 'ทั่วไป (ตอนที่ 124)'.

    Returns (kind, number, part, base_label) where kind is 'section' (มาตรา), 'clause' (ข้อ)
    or None for un-numbered chunks; part defaults to 1 for un-split chunks.
    """
    norm = str(label or "").translate(TH_TO_AR).strip()
    m_part = _PART_RE.search(norm)
    part = int(m_part.group(1)) if m_part else 1
    base = re.sub(r"\s*\(\s*ตอนที่\s*\d+\s*\)\s*$", "", norm).strip()
    m = re.match(r"(มาตรา|ข้อ)\s*(\d+)", base)
    if not m:
        return None, None, part, base
    return ("section" if m.group(1) == "มาตรา" else "clause"), int(m.group(2)), part, base


def _refers_elsewhere(text: str, end: int, kind: str) -> bool:
    """
    True when a 'มาตรา N' / 'ข้อ N' mention at text[:end] is qualified by 'แห่ง<another law>'.
    Section refs to the procurement Act ('แห่งพระราชบัญญัติการจัดซื้อจัดจ้าง...') or to the
    current document ('...นี้') stay resolvable.
    """
    tail = re.sub(r"\s+", "", text[end:end + 120])
    idx = tail.find("แห่ง")
    if idx == -1 or idx > 20:
        return False
    after = tail[idx + len("แห่ง"):]
    if re.match(r"(?:พระราชบัญญัติ|ระเบียบ|ประกาศ|กฎกระทรวง)นี้", after):
        return False
    if kind == "section" and after.startswith("พระราชบัญญัติ"):
        rest = after[len("พระราชบัญญัติ"):]
        # Truncated mention ('แห่งพระราชบัญญัติ' at chunk end) is assumed to be the Act
        return bool(rest) and not rest.startswith("การจัดซื้อ")
    return True


def _names_other_year(text: str, start: int, doc_name: str) -> bool:
    """True when the words just before a mention name a document of a different B.E. year
    (e.g. 'ระเบียบ...พ.ศ. 2535 ข้อ 143' read inside a 2560 document)."""
    years = _YEAR_RE.findall(text[max(0, start - 80):start])
    if not years:
        return False
    doc_years = _YEAR_RE.findall(str(doc_name).translate(TH_TO_AR))
    return bool(doc_years) and years[-1] not in doc_years


def extract_legal_edges(units: List[Dict[str, Any]], act_title: str = ACT_TITLE) -> List[Dict[str, Any]]:
    """
    Build deterministic statutory edges from legal units.

    Args:
        units: dicts with keys 'id', 'doc_name', 'label' (e.g. 'มาตรา ๕๖') and 'text'.
        act_title: doc_name of the parent Act that subordinate documents derive authority from.

    Returns:
        Deduplicated edges {'source_id', 'target_id', 'rel_type', 'props'} with rel_type in
        EMPOWERED_BY (subordinate -> Act), CITES and NEXT_SECTION (reading order).
    """
    parsed = []
    # (doc, kind, num) -> id of the first chunk (lowest part) of that section/clause
    head: Dict[Tuple[str, str, int], Tuple[int, str]] = {}
    for u in units:
        kind, num, part, base = parse_unit_label(u.get("label", ""))
        parsed.append((u, kind, num, part, base))
        if kind:
            key = (u["doc_name"], kind, num)
            if key not in head or part < head[key][0]:
                head[key] = (part, u["id"])

    def resolve(doc: str, kind: str, num: int) -> Optional[str]:
        hit = head.get((doc, kind, num))
        return hit[1] if hit else None

    # Keyed per (src, tgt, is_structural): a citation and a reading-order edge between the same
    # pair are independent facts and must not overwrite each other.
    edges: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}

    def add(src: str, tgt: str, rel: str, props: Dict[str, Any]):
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, rel == "NEXT_SECTION")
        prev = edges.get(key)
        # EMPOWERED_BY is the stronger statement about the same citation, so it wins over CITES
        if prev and not (rel == "EMPOWERED_BY" and prev["rel_type"] == "CITES"):
            return
        edges[key] = {"source_id": src, "target_id": tgt, "rel_type": rel, "props": props}

    # 1. Citations
    for u, kind, num, part, base in parsed:
        uid, doc = u["id"], u["doc_name"]
        own = (doc, kind, num)
        is_act = doc == act_title
        is_intro = base.startswith(_INTRO_LABELS)
        text = _LEADING_HEADERS.sub("", str(u.get("text", ""))).translate(TH_TO_AR)

        for m in _SECTION_REF.finditer(text):
            n = int(m.group(1))
            if (act_title, "section", n) == own or _refers_elsewhere(text, m.end(), "section"):
                continue
            tgt = resolve(act_title, "section", n)
            empowered = not is_act and (is_intro or "อาศัยอำนาจ" in text[max(0, m.start() - 150):m.start()])
            add(uid, tgt, "EMPOWERED_BY" if empowered else "CITES", {"section": n, "quote": f"มาตรา {n}"})

        for m in _CLAUSE_REF.finditer(text):
            n = int(m.group(1))
            if (
                (doc, "clause", n) == own
                or _refers_elsewhere(text, m.end(), "clause")
                or _names_other_year(text, m.start(), doc)
            ):
                continue
            add(uid, resolve(doc, "clause", n), "CITES", {"clause": n, "quote": f"ข้อ {n}"})

    # 2. Reading order: continuation parts of one unit, then unit N -> N+1 (no edge across gaps)
    groups: Dict[Tuple[str, Optional[str], Optional[str]], List[Tuple[int, int, str]]] = defaultdict(list)
    for u, kind, num, part, base in parsed:
        group_key = (u["doc_name"], kind, None if kind else base)
        groups[group_key].append((num or 0, part, u["id"]))
    for seq in groups.values():
        seq.sort()
        for (n1, p1, a), (n2, p2, b) in zip(seq, seq[1:]):
            if (n2 == n1 and p2 == p1 + 1) or (n2 == n1 + 1 and n1 != 0):
                add(a, b, "NEXT_SECTION", {"direction": "next"})

    return list(edges.values())


def resolve_case_citations(
    fact_text: str,
    cited_docs: List[str],
    units: List[Dict[str, Any]],
    act_title: str = ACT_TITLE,
) -> List[str]:
    """
    Resolve unit ids cited by an FAQ case: 'มาตรา N' -> Act section N, 'ข้อ N' -> clause N of
    a document the case itself references (never a same-numbered clause in an unrelated doc).
    """
    head: Dict[Tuple[str, str, int], Tuple[int, str]] = {}
    for u in units:
        kind, num, part, _ = parse_unit_label(u.get("label", ""))
        if kind:
            key = (u["doc_name"], kind, num)
            if key not in head or part < head[key][0]:
                head[key] = (part, u["id"])

    text = str(fact_text or "").translate(TH_TO_AR)
    found: List[str] = []
    for m in _SECTION_REF.finditer(text):
        if not _refers_elsewhere(text, m.end(), "section"):
            hit = head.get((act_title, "section", int(m.group(1))))
            if hit and hit[1] not in found:
                found.append(hit[1])
    for m in _CLAUSE_REF.finditer(text):
        if _refers_elsewhere(text, m.end(), "clause"):
            continue
        for doc in cited_docs:
            if _names_other_year(text, m.start(), doc):
                continue
            hit = head.get((doc, "clause", int(m.group(1))))
            if hit and hit[1] not in found:
                found.append(hit[1])
    return found


class LegalCitationLinker:
    """Extracts and injects cross-statute and intra-statute citation edges into GraphDB."""

    ACT_TITLE = ACT_TITLE

    @classmethod
    def link_citations(cls, db) -> Dict[str, int]:
        """
        Scans all Law nodes in GraphDB and connects rich citation edges (both directions).

        Args:
            db: GraphDB instance from GraphDBManager.get_db()

        Returns:
            Dictionary with counts of edges added per relation type.
        """
        units = []
        for node_id, node_info in db.nodes_data.items():
            if node_info.get("type") != "Laws":
                continue
            data = node_info.get("data", {})
            entry = data.get("entry", "")
            doc_name, _, label = entry.partition("|")
            units.append({
                "id": node_id,
                "doc_name": doc_name.strip(),
                "label": label.strip(),
                "text": data.get("description", ""),
            })

        counts = {rel: 0 for pair in REVERSE_RELATIONS.items() for rel in pair}
        for e in extract_legal_edges(units, cls.ACT_TITLE):
            rel, rev = e["rel_type"], REVERSE_RELATIONS[e["rel_type"]]
            db.add_edge(e["source_id"], e["target_id"], rel, e["props"])
            db.add_edge(e["target_id"], e["source_id"], rev, e["props"])
            counts[rel] += 1
            counts[rev] += 1

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
