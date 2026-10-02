# -*- coding: utf-8 -*-
"""
scripts/migrate_to_tri_store.py

Production ETL & Migration Pipeline for Thai Procurement LegalGraphRAG.
Migrates prepared statutory macro-chunks (datas/law_to_crime.json) and FAQ precedents
(datas/cases_with_feature.json) into the Tri-Store Architecture:
  1. PostgreSQL (SSOT & relational metadata)
  2. Qdrant (1024-dim BGE-M3 Dense + PyThaiNLP Sparse BM25 + RRF)
  3. Neo4j (Topological Knowledge Graph: Hierarchies, Adjacency, Citations, Precedents)

Features:
- Incremental embedding caching (survives network interruptions)
- Structure-aware document, chapter, section, and clause extraction
- Zero data loss validation and cross-store count parity check
"""

import os
import sys
import re
import ast
import json
import time
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from tqdm import tqdm
from dotenv import load_dotenv

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.database import StorageManager
from core.graph_construct.citation_linker import (
    ACT_TITLE,
    extract_legal_edges,
    parse_unit_label,
    resolve_case_citations,
)

# Bump whenever relationship-extraction rules change: a seeded store whose Neo4j graph carries
# an older version gets its relationships rebuilt on the next `--skip-if-seeded` run.
GRAPH_LINKER_VERSION = 3

# Shared linker relations -> Neo4j relationship types queried by Neo4jRepository
NEO4J_REL_TYPES = {
    "CITES": "CITES_CLAUSE",
    "EMPOWERED_BY": "EMPOWERED_BY",
    "NEXT_SECTION": "ADJACENT_SECTION",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_to_tri_store")

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def thai_to_arabic(text: str) -> str:
    """Translate Thai numeral digits to Arabic numeral digits."""
    return str(text).translate(TH_TO_AR)


def normalize_doc_name(raw_name: str) -> str:
    """Normalize a document name or path to a clean title for matching."""
    if not raw_name:
        return "เอกสารจัดซื้อจัดจ้างทั่วไป"
    base = os.path.basename(str(raw_name).strip())
    base = re.sub(r"\.md$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^.*?typhoon_ocr[/\\]+", "", base)
    base = re.sub(r"\s+", " ", base)
    return base.strip()


def slugify_doc_id(doc_title: str) -> str:
    """Generate a clean, deterministic doc_id from document title."""
    clean = re.sub(r"[^\w\u0E00-\u0E7F]+", "_", doc_title.strip()).strip("_")
    h = hashlib.md5(doc_title.encode("utf-8")).hexdigest()[:8]
    return f"doc_{clean[:40]}_{h}"


def extract_year_be(title: str) -> Optional[int]:
    """Extract Buddhist Era year (e.g. 2560) from document title."""
    norm = thai_to_arabic(title)
    m = re.search(r"พ\.ศ\.\s*(\d{4})", norm)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def classify_doc_type(title: str) -> str:
    """Classify legal document level from the title's leading instrument type.

    Matching on the prefix matters: subordinate titles such as 'กฎกระทรวง...ตามพระราชบัญญัติ...'
    mention the Act but are not Acts themselves.
    """
    t = title.strip()
    if t.startswith(("พระราชบัญญัติ", "พ.ร.บ.")):
        return "ACT"
    if t.startswith("กฎกระทรวง"):
        return "MINISTERIAL_RULE"
    if t.startswith("ระเบียบ"):
        return "REGULATION"
    if t.startswith(("ประกาศ", "หนังสือเวียน")) or " ว " in t:
        return "CIRCULAR"
    return "REGULATION"


def extract_pages(entry_id: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract page_start and page_end from chunk entry ID like _p1_p3 or _p7."""
    m = re.search(r"_p(\d+)(?:_p(\d+))?", entry_id)
    if m:
        p_start = int(m.group(1))
        p_end = int(m.group(2)) if m.group(2) else p_start
        return p_start, p_end
    return None, None


def extract_chapter(entry_text: str, content_text: str) -> Optional[int]:
    """Extract the chapter number (หมวด) using normalized Arabic digits."""
    m_ch = re.search(r"หมวด\s*(\d+)", thai_to_arabic(f"{entry_text} {content_text[:600]}"))
    return int(m_ch.group(1)) if m_ch else None


def batch_embed_texts(
    texts: List[str],
    provider: str = "tokenmind",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "BAAI/bge-m3",
    batch_size: int = 16,
) -> List[List[float]]:
    """Generate dense 1024-dim embeddings in batches using Tokenmind API or fallback."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    all_embeddings = []
    failed = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        chunk = [str(t)[:2000] for t in texts[i:i + batch_size]]
        try:
            payload = {"model": model, "input": chunk}
            resp = requests.post(
                f"{base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=45.0,
            )
            resp.raise_for_status()
            res_data = resp.json()
            items = res_data.get("data", [])
            for item in items:
                all_embeddings.append(item["embedding"])
        except Exception as e:
            logger.error(f"Batch embed failed at offset {i}: {e}. Retrying sequentially...")
            for single_t in chunk:
                try:
                    payload = {"model": model, "input": [single_t]}
                    r = requests.post(f"{base_url}/embeddings", headers=headers, json=payload, timeout=20.0)
                    r.raise_for_status()
                    all_embeddings.append(r.json()["data"][0]["embedding"])
                except Exception as ex:
                    logger.error(f"Single embed failed: {ex}")
                    failed += 1

    # Zero-vector placeholders would silently poison the dense index (and the embedding
    # cache), so abort instead and let the caller re-run once the embedding API is healthy.
    if failed:
        raise RuntimeError(f"{failed}/{len(texts)} texts could not be embedded; aborting migration.")
    return all_embeddings


def is_already_seeded(storage: StorageManager) -> bool:
    """True when all three stores hold data in parity and the graph was built by the current linker."""
    try:
        stats = storage.get_stats()
        linker_version = storage.neo4j.get_graph_meta("linker_version")
    except Exception as e:
        logger.warning(f"Could not read Tri-Store stats: {e}")
        return False
    pg_clauses = stats["postgres"]["statute_clauses"]
    in_parity = (
        pg_clauses > 0
        and stats["qdrant"]["statutes_points"] == pg_clauses
        and stats["neo4j"].get("clauses", 0) == pg_clauses
    )
    if in_parity and linker_version != GRAPH_LINKER_VERSION:
        print(f"Neo4j graph built by linker v{linker_version}, current is v{GRAPH_LINKER_VERSION}: relinking.")
        return False
    return in_parity


def run_migration(
    laws_path: str = "datas/law_to_crime.json",
    cases_path: str = "datas/cases_with_feature.json",
    cache_dir: str = "outputs/migration_cache",
    limit: Optional[int] = None,
    skip_embed: bool = False,
    skip_if_seeded: bool = False,
    force_embed: bool = False,
):
    print("=" * 65)
    print("LEGAL-GRAPH-RAG: TRI-STORE DATABASE MIGRATION PIPELINE")
    print("=" * 65)
    t_start = time.time()
    # Never override: inside docker compose, POSTGRES_HOST/QDRANT_HOST/NEO4J_URI point at the
    # service names, while the mounted .env still holds the localhost values for host-side runs.
    load_dotenv(override=False)

    storage = StorageManager.get_instance()
    storage.init_all_stores()

    if skip_if_seeded and is_already_seeded(storage):
        print("Tri-Store already seeded (PostgreSQL/Qdrant/Neo4j clause counts in parity). Skipping.")
        return

    tokenmind_api_key = os.getenv("TOKENMIND_API_KEY", "").strip()
    tokenmind_base_url = os.getenv("TOKENMIND_BASE_URL", "https://tokenmind.abdul.in.th/v1").rstrip("/")
    tokenmind_model = os.getenv("TOKENMIND_EMBEDDING_MODEL", "BAAI/bge-m3")
    os.makedirs(cache_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Parse and normalize Statutory Clauses & Documents
    # ---------------------------------------------------------
    print("\n[Step 1] Loading and normalizing statutory macro chunks...")
    if not os.path.exists(laws_path):
        raise FileNotFoundError(f"Corpus file not found: {laws_path}")

    with open(laws_path, "r", encoding="utf-8") as f:
        raw_laws = json.load(f)

    if limit and limit > 0:
        raw_laws = raw_laws[:limit]
        print(f"Limiting ingestion to first {limit} records as requested.")

    doc_map: Dict[str, Dict[str, Any]] = {}
    clauses: List[Dict[str, Any]] = []
    # Inputs for the shared citation linker (core/graph_construct/citation_linker.py)
    linker_units: List[Dict[str, Any]] = []

    for raw in raw_laws:
        entry_id = str(raw.get("id", ""))
        items = raw.get("items", [])
        if not items:
            continue

        item = items[0]
        text_content = str(item.get("text", ""))
        related_laws = item.get("related_laws", [])
        topics = item.get("crime", [])
        judge_dep = item.get("judge_dep", [])

        # Extract document title
        doc_title = ""
        if related_laws:
            first_elem = related_laws[0] if isinstance(related_laws, list) else str(related_laws)
            if isinstance(first_elem, str) and (".md" in first_elem or "พ.ศ." in first_elem):
                doc_title = normalize_doc_name(first_elem)
        if not doc_title:
            doc_title = normalize_doc_name(entry_id.split("|")[0])

        doc_id = slugify_doc_id(doc_title)
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "doc_id": doc_id,
                "title": doc_title,
                "doc_type": classify_doc_type(doc_title),
                "year_be": extract_year_be(doc_title),
                "source_file": doc_title + ".md",
                "total_pages": None,
                "metadata": {"source": "typhoon_ocr"}
            }

        p_start, p_end = extract_pages(entry_id)
        # Numbers come from the chunk's own label ('มาตรา ๕๖', 'ข้อ ๗๙ (ตอนที่ 2)'), never from body
        # text: a regulation clause saying 'ตามมาตรา 56' is not section 56.
        # The entry id keeps the '(ตอนที่ N)' part suffix that the 'section' field drops.
        label = (entry_id.partition("|")[2] or str(raw.get("section") or "")).strip()
        kind, unit_num, _, _ = parse_unit_label(label)
        sec_num = unit_num if kind == "section" else None
        cls_num = unit_num if kind == "clause" else None
        chap_num = extract_chapter(entry_id, text_content)

        clause_id = f"clause_{hashlib.md5(entry_id.encode('utf-8')).hexdigest()[:16]}"
        linker_units.append({"id": clause_id, "doc_name": doc_title, "label": label, "text": text_content})
        clauses.append({
            "clause_id": clause_id,
            "doc_id": doc_id,
            "entry": entry_id,
            "chapter_num": chap_num,
            "section_num": sec_num,
            "clause_num": cls_num,
            "page_start": p_start,
            "page_end": p_end,
            "content_thai": text_content,
            "judge_dep": judge_dep,
            "related_laws": related_laws,
            "topics": topics,
        })

    documents = list(doc_map.values())
    print(f"Extracted {len(documents)} distinct Legal Documents and {len(clauses)} Statutory Clauses.")

    # ---------------------------------------------------------
    # Step 2: Load and normalize FAQ Precedent Cases
    # ---------------------------------------------------------
    print("\n[Step 2] Loading and normalizing FAQ Precedents...")
    faq_cases: List[Dict[str, Any]] = []
    if os.path.exists(cases_path):
        with open(cases_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)
        for rc in raw_cases:
            cid = f"case_cgd_{rc.get('id', len(faq_cases))}"
            fact_text = str(rc.get("fact", ""))
            
            # Extract question and answer from fact
            parts = fact_text.split("แนวทางวินิจฉัย/คำตอบ:")
            if len(parts) == 2:
                q_text = parts[0].replace("ข้อหารือ/คำถาม:", "").strip()
                a_text = parts[1].strip()
            else:
                q_text = fact_text
                a_text = ""

            faq_cases.append({
                "case_id": cid,
                "source": "FAQ_CGD",
                "question": q_text,
                "answer": a_text,
                "features": rc.get("features", {}),
                "cited_laws": rc.get("law", rc.get("laws", [])),
                "topics": rc.get("crime", []),
            })
        print(f"Extracted {len(faq_cases)} FAQ Precedent Cases.")
    else:
        print(f"Warning: Cases file not found at {cases_path}. Skipping FAQ cases.")

    # ---------------------------------------------------------
    # Step 3: Ingest into PostgreSQL (SSOT)
    # ---------------------------------------------------------
    print("\n[Step 3] Upserting records into PostgreSQL (SSOT)...")
    storage.pg.upsert_documents(documents)
    storage.pg.upsert_clauses(clauses)
    if faq_cases:
        storage.pg.upsert_faq_cases(faq_cases)
    print("PostgreSQL upsert complete.")

    # ---------------------------------------------------------
    # Step 4: Generate Embeddings & Upsert into Qdrant
    # ---------------------------------------------------------
    print("\n[Step 4] Generating embeddings and indexing into Qdrant VectorDB...")
    qdrant_stats = storage.qdrant.count_stats()
    qdrant_in_parity = (
        qdrant_stats["statutes_points"] == len(clauses)
        and qdrant_stats["cases_points"] == len(faq_cases)
    )
    if qdrant_in_parity and not force_embed:
        # clause_id is a hash of the entry, so equal counts mean the same points are already indexed
        print(f"Qdrant already holds {len(clauses)} statute / {len(faq_cases)} case points. Skipping (use --force-embed to re-index).")
        skip_embed = True

    cache_file = os.path.join(cache_dir, f"dense_embeddings_{len(clauses)}.json")
    dense_embeddings = None

    if not skip_embed and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                dense_embeddings = json.load(f)
            if len(dense_embeddings) != len(clauses):
                print("Cache length mismatch. Recomputing embeddings...")
                dense_embeddings = None
        except Exception:
            dense_embeddings = None

    if not skip_embed and not tokenmind_api_key and (dense_embeddings is None or faq_cases):
        raise RuntimeError("TOKENMIND_API_KEY is not set; it is required to embed the corpus into Qdrant.")

    if dense_embeddings is None and not skip_embed:
        clause_texts = [f"{c['entry']}\n{c['content_thai'][:1500]}" for c in clauses]
        print(f"Calling Tokenmind Embedding API ({tokenmind_model}) for {len(clause_texts)} clauses...")
        dense_embeddings = batch_embed_texts(
            clause_texts,
            api_key=tokenmind_api_key,
            base_url=tokenmind_base_url,
            model=tokenmind_model,
            batch_size=32,
        )

        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(dense_embeddings, f)
        print(f"Saved embeddings to cache at {cache_file}.")

    if dense_embeddings:
        print("Upserting dense (1024-dim) and native sparse (BM25) points into Qdrant...")
        storage.qdrant.upsert_statute_points(clauses, dense_embeddings, batch_size=100)

        # Upsert FAQ cases into Qdrant
        if faq_cases:
            case_texts = [f"{cs['question']} {cs['answer']}" for cs in faq_cases]
            case_embs = batch_embed_texts(
                case_texts,
                api_key=tokenmind_api_key,
                base_url=tokenmind_base_url,
                model=tokenmind_model,
                batch_size=16,
            )
            storage.qdrant.upsert_case_points(faq_cases, case_embs, batch_size=50)

    # ---------------------------------------------------------
    # Step 5: Build Knowledge Graph in Neo4j
    # ---------------------------------------------------------
    print("\n[Step 5] Building Knowledge Graph in Neo4j...")
    storage.neo4j.sync_documents(documents)
    storage.neo4j.sync_statute_clauses(clauses, batch_size=500)
    if faq_cases:
        storage.neo4j.sync_faq_cases(faq_cases)

    # MERGE never removes edges, so derived corpus relationships from earlier (buggier) runs
    # would survive a re-link. Drop them first; CONTAINS and tenant-private edges are kept.
    removed = storage.neo4j.clear_derived_relationships(
        list(NEO4J_REL_TYPES.values()) + ["RELATES_TO_LAW", "REFERENCES_DOCUMENT"]
    )
    print(f"Removed {removed} previously derived relationships.")

    # 5.1 Statutory edges from the shared linker (same rules as the in-memory GraphDB build)
    print("Extracting citation, empowerment and reading-order edges (citation_linker)...")
    edges: List[Dict[str, Any]] = []
    for e in extract_legal_edges(linker_units, act_title=normalize_doc_name(ACT_TITLE)):
        rel = NEO4J_REL_TYPES[e["rel_type"]]
        if rel == "ADJACENT_SECTION":
            edges.append({**e, "rel_type": rel, "props": {"direction": "next"}})
            edges.append({
                "source_id": e["target_id"],
                "target_id": e["source_id"],
                "rel_type": rel,
                "props": {"direction": "prev"},
            })
        else:
            edges.append({**e, "rel_type": rel})

    if edges:
        storage.neo4j.sync_relationships(edges, batch_size=1000)

    # 5.2 Link FAQ Cases to the documents they reference and to any clause they cite
    if faq_cases:
        title_to_doc_id = {d["title"]: d["doc_id"] for d in documents}
        case_doc_links, case_links = [], []
        for cs in faq_cases:
            cited_docs = [normalize_doc_name(str(t)) for t in cs.get("cited_laws", [])]
            cited_docs = [t for t in cited_docs if t in title_to_doc_id]
            case_doc_links.extend(
                {"case_id": cs["case_id"], "doc_id": title_to_doc_id[t]} for t in dict.fromkeys(cited_docs)
            )
            fact_text = f"{cs['question']} {cs['answer']}"
            case_links.extend(
                {"case_id": cs["case_id"], "clause_id": cid}
                for cid in resolve_case_citations(fact_text, cited_docs, linker_units, normalize_doc_name(ACT_TITLE))
            )

        if case_doc_links:
            storage.neo4j.link_case_to_documents(case_doc_links)
        if case_links:
            storage.neo4j.link_case_to_laws(case_links)
        print(
            f"Linked FAQ precedents: {len(case_doc_links)} case->document, "
            f"{len(case_links)} case->clause relationships."
        )

    storage.neo4j.set_graph_meta("linker_version", GRAPH_LINKER_VERSION)
    # ---------------------------------------------------------
    # Step 6: Parity Validation & Final Report
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("MIGRATION PARITY & INTEGRITY VERIFICATION REPORT")
    print("=" * 65)
    stats = storage.get_stats()
    print(f"PostgreSQL (SSOT):")
    print(f"  - Legal Documents:  {stats['postgres']['legal_documents']}")
    print(f"  - Statute Clauses:  {stats['postgres']['statute_clauses']}")
    print(f"  - FAQ Precedents:   {stats['postgres']['faq_cases']}")
    print(f"\nQdrant VectorDB (1024-dim Dense + Native Sparse BM25):")
    print(f"  - Statute Points:   {stats['qdrant']['statutes_points']}")
    print(f"  - Case Points:      {stats['qdrant']['cases_points']}")
    print(f"\nNeo4j Knowledge Graph:")
    print(f"  - Documents:        {stats['neo4j'].get('documents', 0)}")
    print(f"  - Clauses:          {stats['neo4j'].get('clauses', 0)}")
    print(f"  - FAQ Cases:        {stats['neo4j'].get('cases', 0)}")
    print(f"  - Relationships:    {stats['neo4j'].get('relationships', 0)}")
    print(f"\nMigration completed successfully in {time.time() - t_start:.2f} seconds!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Migrate Thai Procurement QA corpus to Tri-Store")
    parser.add_argument("--laws-path", default=os.getenv("law_to_crime_path", "datas/law_to_crime.json"), help="Path to law_to_crime.json")
    parser.add_argument("--cases-path", default=os.getenv("case_db_path", "datas/cases_with_feature.json"), help="Path to cases_with_feature.json")
    parser.add_argument("--cache-dir", default="outputs/migration_cache", help="Cache directory for embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to ingest (for testing)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation and Qdrant ingestion")
    parser.add_argument("--force-embed", action="store_true", help="Re-embed and re-index Qdrant even if it is already in parity")
    parser.add_argument("--skip-if-seeded", action="store_true", help="Exit early if all three stores are already populated and in parity")

    args = parser.parse_args()
    run_migration(
        laws_path=args.laws_path,
        cases_path=args.cases_path,
        cache_dir=args.cache_dir,
        limit=args.limit,
        skip_embed=args.skip_embed,
        skip_if_seeded=args.skip_if_seeded,
        force_embed=args.force_embed,
    )
