# -*- coding: utf-8 -*-
"""
scripts/rebuild_tenant_index.py

Rebuilds the Qdrant `tenant_documents` collection from the tenant chunks stored in PostgreSQL:
drops the collection, recreates it with the current configuration (BM25 sparse vectors with the
IDF modifier), re-embeds every chunk and upserts it again. PostgreSQL and Neo4j are untouched, so
tenants keep their documents; only the vector index is rebuilt.

Usage (Docker stack running):
    python scripts/rebuild_tenant_index.py            # rebuild
    python scripts/rebuild_tenant_index.py --check    # report whether a rebuild is needed
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

from core.database import StorageManager  # noqa: E402
from core.retrieval.embedding import batch_embed_tokenmind  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Rebuild the tenant document vector index from PostgreSQL")
    ap.add_argument("--check", action="store_true", help="Only report the collection's sparse configuration")
    args = ap.parse_args()

    storage = StorageManager.get_instance()
    storage.init_all_stores()
    qd = storage.qdrant
    has_idf = qd.tenant_collection_has_idf()
    print(f"Collection '{qd.TENANT_DOCS_COLLECTION}': sparse IDF modifier {'on' if has_idf else 'OFF'}")
    if args.check:
        return

    rows = storage.pg.all_tenant_chunks()
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[(r["org_id"], r["doc_id"], r["title"])].append(r)
    print(f"{len(rows)} chunks in {len(by_doc)} tenant documents "
          f"({len({k[0] for k in by_doc})} tenants) will be re-indexed.")

    # Embed first: if the embedding API fails, the existing collection is left as it was
    texts = [f"[{r['title']}]\n{r['content']}" for r in rows]
    vectors = batch_embed_tokenmind(texts) if texts else []
    vec_by_chunk = {r["chunk_id"]: v for r, v in zip(rows, vectors)}

    qd.recreate_tenant_collection()
    for (org_id, doc_id, title), chunks in by_doc.items():
        qd.upsert_tenant_chunks(org_id, doc_id, title, chunks, [vec_by_chunk[c["chunk_id"]] for c in chunks])
    print(f"Done. IDF modifier now {'on' if qd.tenant_collection_has_idf() else 'OFF'}; "
          f"{qd.client.count(collection_name=qd.TENANT_DOCS_COLLECTION).count} points.")


if __name__ == "__main__":
    main()
