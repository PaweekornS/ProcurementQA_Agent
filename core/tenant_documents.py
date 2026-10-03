# -*- coding: utf-8 -*-
"""
core/tenant_documents.py

Ingestion of tenant-private documents (OCR markdown: TOR, BOQ, contracts...) into the Tri-Store.
Each document is split into page-aware chunks, embedded, and written to PostgreSQL
(tenant_documents / tenant_chunks, RLS-scoped) and the Qdrant `tenant_documents` collection.
Only the owning tenant can retrieve these chunks; they are never PUBLIC.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from core.utils.settings import env
from core.graph_construct.citation_linker import ACT_TITLE, TH_TO_AR, _names_other_year, _refers_elsewhere

logger = logging.getLogger(__name__)

# Same alphabet the OCR service accepts, so a tenant id maps 1:1 across both services
ORG_ID_PATTERN = re.compile(r"^[A-Za-z0-9\u0E00-\u0E7F_-]{1,64}$")
TENANT_CHUNK_PREFIX = "tdoc:"

_PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s+of\s+(\d+)\s*-->", re.IGNORECASE)
_MAX_CHARS = 1200
_HARD_LIMIT = 1800

# A TOR has its own "ข้อ 1, ข้อ 2..." numbering, so a bare "ข้อ N" is only linked to the MoF
# regulation when the regulation is named just before it; "มาตรา N" defaults to the Act.
REGULATION_TITLE = "ระเบียบกระทรวงการคลังว่าด้วยการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. 2560"
_SECTION_REF = re.compile(r"มาตรา\s*(\d+)")
_CLAUSE_REF = re.compile(r"ข้อ\s*(\d+)")
_ANY_REF = re.compile(r"มาตรา\s*\d|ข้อ\s*\d")
_REGULATION_CUE = re.compile(r"ระเบียบ(?:กระทรวงการคลัง|ฯ)")


def is_tenant_chunk_id(value: Optional[str]) -> bool:
    return bool(value) and str(value).startswith(TENANT_CHUNK_PREFIX)


def make_doc_id(org_id: str, source_id: str) -> str:
    digest = hashlib.sha1(f"{org_id}|{source_id}".encode("utf-8")).hexdigest()[:16]
    return f"{TENANT_CHUNK_PREFIX}{org_id}:{digest}"


def split_pages(markdown: str) -> List[Dict[str, Any]]:
    """[(page, text)] using the OCR '<!-- Page N of M -->' markers; unmarked text is page None."""
    markers = list(_PAGE_MARKER.finditer(markdown))
    if not markers:
        return [{"page": None, "total": None, "text": markdown}]
    pages = []
    head = markdown[:markers[0].start()]
    if head.strip():
        pages.append({"page": None, "total": None, "text": head})
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(markdown)
        pages.append({"page": int(m.group(1)), "total": int(m.group(2)), "text": markdown[m.end():end]})
    return pages


def chunk_markdown(markdown: str) -> Dict[str, Any]:
    """Paragraph-packed chunks of ~_MAX_CHARS that never cross a page boundary."""
    chunks: List[Dict[str, Any]] = []
    total_pages = None
    for page in split_pages(markdown):
        total_pages = page["total"] or total_pages
        buf = ""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page["text"]) if p.strip()]
        pieces = []
        for p in paragraphs:
            while len(p) > _HARD_LIMIT:
                pieces.append(p[:_MAX_CHARS])
                p = p[_MAX_CHARS:]
            pieces.append(p)
        for p in pieces:
            if buf and len(buf) + len(p) + 2 > _MAX_CHARS:
                chunks.append({"content": buf, "page_start": page["page"], "page_end": page["page"]})
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            chunks.append({"content": buf, "page_start": page["page"], "page_end": page["page"]})
    return {"chunks": chunks, "total_pages": total_pages}


def extract_statute_refs(text: str) -> List[Dict[str, Any]]:
    """[{title, kind, num, quote}] for statute references in a tenant chunk (conservative)."""
    norm = str(text or "").translate(TH_TO_AR)
    refs: Dict[tuple, Dict[str, Any]] = {}

    def add(title: str, kind: str, num: int, m: re.Match):
        quote = norm[max(0, m.start() - 60):m.end() + 60].replace("\n", " ").strip()
        refs.setdefault((title, kind, num), {"title": title, "kind": kind, "num": num, "quote": quote})

    def elsewhere(m: re.Match, kind: str) -> bool:
        # Stop at the next reference so "ข้อ 175 และมาตรา 3 แห่งประมวลรัษฎากร" doesn't taint ข้อ 175
        nxt = _ANY_REF.search(norm, m.end())
        return _refers_elsewhere(norm[:nxt.start()] if nxt else norm, m.end(), kind)

    for m in _SECTION_REF.finditer(norm):
        if not elsewhere(m, "section") and not _names_other_year(norm, m.start(), ACT_TITLE):
            add(ACT_TITLE, "section", int(m.group(1)), m)
    for m in _CLAUSE_REF.finditer(norm):
        before = norm[max(0, m.start() - 80):m.start()]
        if (_REGULATION_CUE.search(before) and not elsewhere(m, "clause")
                and not _names_other_year(norm, m.start(), REGULATION_TITLE)):
            add(REGULATION_TITLE, "clause", int(m.group(1)), m)
    return list(refs.values())


def _sync_graph(storage, org_id: str, doc: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mirror the document into Neo4j and link chunks to the statutes they cite. Best-effort:
    vector/BM25 search already works without the graph, so a graph failure must not fail ingestion."""
    try:
        per_chunk = [(c["chunk_id"], extract_statute_refs(c["content"])) for c in chunks]
        wanted = {(r["title"], r["kind"], r["num"]) for _, refs in per_chunk for r in refs}
        resolved = storage.neo4j.resolve_statute_refs(
            [{"title": t, "kind": k, "num": n} for t, k, n in wanted])
        citations = [
            {"chunk_id": cid, "clause_id": clause_id, "quote": r["quote"]}
            for cid, refs in per_chunk
            for r in refs
            for clause_id in resolved.get((r["title"], r["kind"], r["num"]), [])
        ]
        linked = storage.neo4j.replace_tenant_document(org_id, doc, chunks, citations)
        return {"graph_status": "synced", "graph_citations": linked}
    except Exception as e:
        logger.exception("Neo4j sync failed for tenant document %s: %s", doc.get("doc_id"), e)
        return {"graph_status": f"failed: {type(e).__name__}", "graph_citations": 0}


def _embed(texts: List[str]) -> List[List[float]]:
    from core.graph_construct.feature_graph import batch_embed_tokenmind, get_embedding
    if env("EMBEDDING_PROVIDER", "local").lower() == "tokenmind":
        vectors = batch_embed_tokenmind(texts)
    else:
        vectors = [get_embedding(t) for t in texts]
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding returned {len(vectors)} vectors for {len(texts)} chunks")
    return vectors


def ingest_document(
    org_id: str,
    title: str,
    markdown: str,
    source_id: Optional[str] = None,
    source_file: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or replace one tenant document. Re-sending the same source_id replaces it."""
    from core.database import StorageManager

    parsed = chunk_markdown(markdown)
    if not parsed["chunks"]:
        raise ValueError("Document has no text to index")

    doc_id = make_doc_id(org_id, source_id or title)
    chunks = [
        {**c, "chunk_id": f"{doc_id}#{i:04d}", "chunk_index": i}
        for i, c in enumerate(parsed["chunks"])
    ]
    vectors = _embed([f"{title}\n{c['content']}" for c in chunks])

    storage = StorageManager.get_instance()
    doc = {
        "doc_id": doc_id,
        "title": title,
        "source_id": source_id or title,
        "source_file": source_file,
        "total_pages": parsed["total_pages"],
        "metadata": metadata or {},
    }
    storage.pg.replace_tenant_document(org_id, doc, chunks)
    storage.qdrant.upsert_tenant_chunks(org_id, doc_id, title, chunks, vectors)
    graph = _sync_graph(storage, org_id, doc, chunks)
    return {
        "doc_id": doc_id,
        "org_id": org_id,
        "title": title,
        "chunks": len(chunks),
        "total_pages": parsed["total_pages"],
        **graph,
    }


def delete_document(org_id: str, doc_id: str) -> bool:
    from core.database import StorageManager
    storage = StorageManager.get_instance()
    deleted = storage.pg.delete_tenant_document(org_id, doc_id)
    storage.qdrant.delete_tenant_document(org_id, doc_id)
    try:
        storage.neo4j.delete_tenant_document(org_id, doc_id)
    except Exception as e:
        logger.exception("Neo4j delete failed for tenant document %s: %s", doc_id, e)
    return deleted


def list_documents(org_id: str) -> List[Dict[str, Any]]:
    from core.database import StorageManager
    return StorageManager.get_instance().pg.list_tenant_documents(org_id)
