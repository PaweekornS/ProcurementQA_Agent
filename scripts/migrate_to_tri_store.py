# -*- coding: utf-8 -*-
"""
scripts/migrate_to_tri_store.py

Production ETL & Migration Pipeline for ProcurementQA Agent.
Chunks the OCR corpus (data_ocr/ -> outputs/corpus/chunks.jsonl, core/chunking) and migrates the
chunks and FAQ pairs into the Tri-Store Architecture:
  1. PostgreSQL (SSOT & relational metadata)
  2. Qdrant (1024-dim BGE-M3 Dense + PyThaiNLP Sparse BM25 + RRF)
  3. Neo4j (Topological Knowledge Graph: Hierarchies, Adjacency, Citations, Precedents)

Features:
- Incremental embedding caching (survives network interruptions)
- --reset-corpus replaces a tenant's corpus (and drops stores from before the chunk rename)
- Zero data loss validation and cross-store count parity check
"""

import os
import sys
import re
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
from core.utils.settings import env
from core.graph.citation_linker import (
    ACT_TITLE,
    extract_legal_edges,
    parse_unit_label,
    resolve_case_citations,
)

# Bump whenever relationship-extraction rules or derived chunk metadata (e.g. page ranges) change:
# a seeded store whose Neo4j graph carries an older version is re-upserted into PostgreSQL and
# relinked on the next `--skip-if-seeded` run (Qdrant is skipped while in parity).
GRAPH_LINKER_VERSION = 5  # 5: chunk ids / Chunk labels

# Chunks are sized to fit (core/chunking keeps them <= ~2,000 chars plus a short context header)
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "2500"))

# Shared linker relations -> Neo4j relationship types queried by Neo4jRepository
NEO4J_REL_TYPES = {
    "CITES": "CITES_CLAUSE",
    "EMPOWERED_BY": "EMPOWERED_BY",
    "NEXT_SECTION": "ADJACENT_SECTION",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_to_tri_store")

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_UNIT_REF = re.compile(r"(มาตรา|ข้อ)\s*(\d+)")


def thai_to_arabic(text: str) -> str:
    """Translate Thai numeral digits to Arabic numeral digits."""
    return str(text).translate(TH_TO_AR)


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
        chunk = [str(t)[:EMBED_MAX_CHARS] for t in texts[i:i + batch_size]]
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
    pg_chunks = stats["postgres"]["chunks"]
    in_parity = (
        pg_chunks > 0
        and stats["qdrant"]["chunk_points"] == pg_chunks
        and stats["neo4j"].get("chunks", 0) == pg_chunks
    )
    if in_parity and linker_version != GRAPH_LINKER_VERSION:
        print(f"Neo4j graph built by linker v{linker_version}, current is v{GRAPH_LINKER_VERSION}: relinking.")
        return False
    return in_parity


def load_corpus(chunks_path: str, limit: Optional[int] = None):
    """chunks.jsonl (core/chunking/corpus.py) -> (documents, chunks, faq_cases, linker_units)."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if limit and limit > 0:
        records = records[:limit]
        print(f"Limiting ingestion to first {limit} chunks as requested.")

    documents: Dict[str, Dict[str, Any]] = {}
    chunks: List[Dict[str, Any]] = []
    faq_cases: List[Dict[str, Any]] = []
    linker_units: List[Dict[str, Any]] = []
    for r in records:
        title = r["doc_title"]
        doc_id = slugify_doc_id(title)
        documents.setdefault(doc_id, {
            "doc_id": doc_id,
            "title": title,
            "doc_type": classify_doc_type(title),
            "year_be": extract_year_be(title),
            "source_file": r.get("source_file"),   # path under data_ocr/, returned with citations
            "total_pages": r.get("total_pages"),
            "metadata": {"source": "data_ocr", "chunker_doc_type": r.get("doc_type")},
        })
        # Numbers come from the chunk's own label ('มาตรา 56', 'ข้อ 79 (ตอนที่ 2)'), never from body
        # text: a regulation clause saying 'ตามมาตรา 56' is not section 56.
        kind, unit_num, _, _ = parse_unit_label(r["label"])
        text = r.get("embed_text") or r["content"]
        chunks.append({
            "chunk_id": r["chunk_id"],
            "doc_id": doc_id,
            "entry": r["entry"],
            "chapter_num": extract_chapter(" ".join(r.get("section_path", [])), ""),
            "section_num": unit_num if kind == "section" else None,
            "clause_num": unit_num if kind == "clause" else None,
            "page_start": r.get("page_start"),
            "page_end": r.get("page_end"),
            "content": text,
            "kind": r.get("kind"),
            "label": r["label"],
            "section_path": r.get("section_path", []),
        })
        linker_units.append({"id": r["chunk_id"], "doc_name": title, "label": r["label"], "text": text})
        if r.get("kind") == "faq":
            md = r.get("metadata", {})
            answer = md.get("answer", "")
            faq_cases.append({
                "case_id": r["chunk_id"],
                "source": "FAQ_CGD",
                "question": md.get("question", ""),
                "answer": answer,
                "cited_laws": list(dict.fromkeys(f"{k} {n}" for k, n in _UNIT_REF.findall(thai_to_arabic(answer)))),
                "topics": [md["category"]] if md.get("category") else [],
            })
    return list(documents.values()), chunks, faq_cases, linker_units


def run_migration(
    chunks_path: str = "outputs/corpus/chunks.jsonl",
    cache_dir: str = "outputs/migration_cache",
    limit: Optional[int] = None,
    skip_embed: bool = False,
    skip_if_seeded: bool = False,
    force_embed: bool = False,
    org_id: str = "DGA",
    reset_corpus: bool = False,
):
    print("=" * 65)
    print("PROCUREMENTQA AGENT: TRI-STORE MIGRATION")
    print("=" * 65)
    t_start = time.time()
    # Never override: inside docker compose, POSTGRES_HOST/QDRANT_HOST/NEO4J_URI point at the
    # service names, while the mounted .env still holds the localhost values for host-side runs.
    load_dotenv(override=False)

    storage = StorageManager.get_instance()
    if reset_corpus:
        # Stores created before the chunk rename: their data is rebuilt below from data_ocr/
        storage.pg.drop_legacy_schema()
        storage.qdrant.drop_legacy_collections()
    storage.init_all_stores()

    if skip_if_seeded and not reset_corpus and is_already_seeded(storage):
        print("Tri-Store already seeded (PostgreSQL/Qdrant/Neo4j chunk counts in parity). Skipping.")
        return

    tokenmind_api_key = env("TOKENMIND_API_KEY", "").strip()
    tokenmind_base_url = env("TOKENMIND_BASE_URL", "https://tokenmind.abdul.in.th/v1").rstrip("/")
    tokenmind_model = env("TOKENMIND_EMBEDDING_MODEL", "BAAI/bge-m3")
    os.makedirs(cache_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Steps 1-2: Corpus chunks, their documents, and FAQ pairs
    # ---------------------------------------------------------
    print(f"\n[Step 1] Loading chunks from {chunks_path}...")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Corpus file not found: {chunks_path}")
    documents, chunks, faq_cases, linker_units = load_corpus(chunks_path, limit)
    located = sum(1 for c in chunks if c["page_start"] is not None)
    print(f"{len(documents)} documents, {len(chunks)} chunks ({located} with page ranges), "
          f"{len(faq_cases)} FAQ pairs.")

    # ---------------------------------------------------------
    # Step 3: Ingest into PostgreSQL (SSOT)
    # ---------------------------------------------------------
    # Every corpus record is owned by one tenant; the stores read org_id from each record.
    for record in (*documents, *chunks, *faq_cases):
        record["org_id"] = org_id
    print(f"\nCorpus tenant (org_id): {org_id}")

    if reset_corpus:
        # Chunk ids are hashes of their entries, so a re-chunked corpus would otherwise sit next
        # to the previous one instead of replacing it. Tenant-uploaded documents are kept.
        print(f"\n[Reset] Removing the existing corpus of tenant {org_id} from all three stores...")
        print(f"  PostgreSQL: {storage.pg.delete_corpus(org_id)}")
        storage.qdrant.delete_corpus(org_id)
        print(f"  Neo4j: {storage.neo4j.delete_corpus(org_id)} nodes")
        force_embed = True

    print("\n[Step 3] Upserting records into PostgreSQL (SSOT)...")
    storage.pg.upsert_documents(documents)
    storage.pg.upsert_chunks(chunks)
    if faq_cases:
        storage.pg.upsert_faq_cases(faq_cases)
    print("PostgreSQL upsert complete.")

    # ---------------------------------------------------------
    # Step 4: Generate Embeddings & Upsert into Qdrant
    # ---------------------------------------------------------
    print("\n[Step 4] Generating embeddings and indexing into Qdrant VectorDB...")
    # BM25 length normalisation needs the corpus average; tenant uploads later use BM25_AVG_DOC_LEN
    vectorizer = storage.qdrant.vectorizer
    lengths = [len(vectorizer.tokens(c["content"])) for c in chunks]
    if lengths:
        vectorizer.avg_doc_len = sum(lengths) / len(lengths)
        print(f"BM25 average chunk length: {vectorizer.avg_doc_len:.1f} tokens "
              f"(set BM25_AVG_DOC_LEN={vectorizer.avg_doc_len:.0f} to match for tenant uploads)")
    qdrant_stats = storage.qdrant.count_stats()
    qdrant_in_parity = (
        qdrant_stats["chunk_points"] == len(chunks)
        and qdrant_stats["cases_points"] == len(faq_cases)
    )
    if qdrant_in_parity and not force_embed:
        # chunk_id is a hash of the entry, so equal counts mean the same points are already indexed
        print(f"Qdrant already holds {len(chunks)} chunk / {len(faq_cases)} FAQ points. Skipping (use --force-embed to re-index).")
        skip_embed = True

    # Keyed by content as well as count: a re-chunked corpus of the same size must not reuse vectors
    corpus_digest = hashlib.md5(
        "".join(c["chunk_id"] + c["content"] for c in chunks).encode("utf-8")
    ).hexdigest()[:10]
    cache_file = os.path.join(cache_dir, f"dense_embeddings_{len(chunks)}_{corpus_digest}.json")
    dense_embeddings = None

    if not skip_embed and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}...")
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                dense_embeddings = json.load(f)
            if len(dense_embeddings) != len(chunks):
                print("Cache length mismatch. Recomputing embeddings...")
                dense_embeddings = None
        except Exception:
            dense_embeddings = None

    if not skip_embed and not tokenmind_api_key and (dense_embeddings is None or faq_cases):
        raise RuntimeError("TOKENMIND_API_KEY is not set; it is required to embed the corpus into Qdrant.")

    if dense_embeddings is None and not skip_embed:
        # Unchanged input format (entry + text) so earlier embedding caches stay comparable;
        # batch_embed_texts caps each text at EMBED_MAX_CHARS
        clause_texts = [f"{c['entry']}\n{c['content']}" for c in chunks]
        print(f"Calling Tokenmind Embedding API ({tokenmind_model}) for {len(clause_texts)} chunks...")
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
        storage.qdrant.upsert_chunk_points(chunks, dense_embeddings, batch_size=100)

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
    storage.neo4j.sync_chunks(chunks, batch_size=500)
    if faq_cases:
        storage.neo4j.sync_faq_cases(faq_cases)

    # MERGE never removes edges, so derived corpus relationships from earlier (buggier) runs
    # would survive a re-link. Drop them first; CONTAINS and tenant-private edges are kept.
    removed = storage.neo4j.clear_derived_relationships(
        list(NEO4J_REL_TYPES.values()) + ["RELATES_TO_LAW", "REFERENCES_DOCUMENT"], org_id=org_id
    )
    print(f"Removed {removed} previously derived relationships.")

    # 5.1 Statutory edges from the shared linker (same rules as the in-memory store)
    print("Extracting citation, empowerment and reading-order edges (citation_linker)...")
    edges: List[Dict[str, Any]] = []
    for e in extract_legal_edges(linker_units, act_title=ACT_TITLE):
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
        # FAQ answers name documents in many short forms ('ระเบียบฯ'), so only titles quoted in
        # full count as document references; 'มาตรา N' always resolves to the Act
        case_doc_links, case_links = [], []
        for cs in faq_cases:
            qa_text = f"{cs['question']} {cs['answer']}"
            cited_docs = [d["title"] for d in documents if d["title"] in qa_text]
            case_doc_links.extend({"case_id": cs["case_id"], "doc_id": slugify_doc_id(t)} for t in cited_docs)
            case_links.extend(
                {"case_id": cs["case_id"], "chunk_id": cid}
                for cid in resolve_case_citations(qa_text, cited_docs, linker_units, ACT_TITLE)
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
    print(f"  - Documents:        {stats['postgres']['documents']}")
    print(f"  - Chunks:           {stats['postgres']['chunks']}")
    print(f"  - FAQ pairs:        {stats['postgres']['faq_cases']}")
    print(f"\nQdrant VectorDB (1024-dim Dense + Sparse BM25 with IDF):")
    print(f"  - Chunk points:     {stats['qdrant']['chunk_points']}")
    print(f"  - FAQ points:       {stats['qdrant']['cases_points']}")
    print(f"\nNeo4j Knowledge Graph:")
    print(f"  - Documents:        {stats['neo4j'].get('documents', 0)}")
    print(f"  - Chunks:           {stats['neo4j'].get('chunks', 0)}")
    print(f"  - FAQ Cases:        {stats['neo4j'].get('cases', 0)}")
    print(f"  - Relationships:    {stats['neo4j'].get('relationships', 0)}")
    print(f"\nMigration completed successfully in {time.time() - t_start:.2f} seconds!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Chunk data_ocr/ and migrate it into the tri-store")
    parser.add_argument("--ocr-dir", default="data_ocr", help="OCR markdown root: the only corpus source; it is chunked on every run")
    parser.add_argument("--corpus-dir", default="outputs/corpus", help="Where the chunked corpus is written (git-ignored)")
    parser.add_argument("--chunks-path", default=None, help="Ingest a prebuilt chunks.jsonl instead of chunking --ocr-dir")
    parser.add_argument("--cache-dir", default="outputs/migration_cache", help="Cache directory for embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to ingest (for testing)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation and Qdrant ingestion")
    parser.add_argument("--force-embed", action="store_true", help="Re-embed and re-index Qdrant even if it is already in parity")
    parser.add_argument("--org-id", default=env("DEFAULT_ORG_ID", "DGA"), help="Tenant that owns the seeded corpus (default: DEFAULT_ORG_ID, else DGA)")
    parser.add_argument("--skip-if-seeded", action="store_true", help="Exit early if all three stores are already populated and in parity")
    parser.add_argument("--reset-corpus", action="store_true", help="Delete the tenant's existing corpus from all three stores before ingesting (use after re-chunking)")

    args = parser.parse_args()
    if not args.chunks_path:
        from core.chunking.corpus import build
        if not os.path.isdir(args.ocr_dir):
            raise SystemExit(f"OCR corpus not found at '{args.ocr_dir}' (mount it, or pass --ocr-dir)")
        # Always rebuilt: chunking takes seconds and the stores must match the current OCR files
        summary = build(Path(args.ocr_dir), Path(args.corpus_dir), verbose=False)
        print(f"Chunked {summary['documents']} documents from {args.ocr_dir} into {summary['chunks']} chunks ({args.corpus_dir})")
        args.chunks_path = os.path.join(args.corpus_dir, "chunks.jsonl")
    run_migration(
        chunks_path=args.chunks_path,
        cache_dir=args.cache_dir,
        limit=args.limit,
        skip_embed=args.skip_embed,
        skip_if_seeded=args.skip_if_seeded,
        force_embed=args.force_embed,
        org_id=args.org_id,
        reset_corpus=args.reset_corpus,
    )
