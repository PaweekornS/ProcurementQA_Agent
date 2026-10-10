# -*- coding: utf-8 -*-
"""
core/database/qdrant_repository.py

Qdrant Vector Database Repository for ProcurementQA Agent.
Provides 1024-dim dense vector search (BGE-M3) and native sparse vector search (BM25)
with server-side Reciprocal Rank Fusion (RRF) and metadata payload filtering.
"""

import os
import re
import hashlib
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv(override=False)
logger = logging.getLogger("qdrant_repository")


class ThaiSparseVectorizer:
    """
    BM25 as Qdrant sparse vectors, with the same Thai tokenization as the in-memory index.

    Qdrant scores a sparse query as  sum_t  q(t) * d(t) * idf(t)  when the collection uses
    Modifier.IDF, with idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)) computed by Qdrant over the
    collection. With
        d(t) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avg_len))      (document side)
        q(t) = 1 for every distinct query term                                    (query side)
    that sum is exactly Okapi BM25. avg_len is the corpus average in tokens (measured by the
    migration; BM25_AVG_DOC_LEN otherwise).
    """

    K1 = 1.2
    B = 0.75
    DEFAULT_AVG_DOC_LEN = 170.0  # outputs/corpus average (tokens per chunk, header included)

    def __init__(self, max_vocab_hash: int = 1000000, avg_doc_len: Optional[float] = None):
        self.max_vocab_hash = max_vocab_hash
        self.avg_doc_len = float(avg_doc_len or os.getenv("BM25_AVG_DOC_LEN", self.DEFAULT_AVG_DOC_LEN))

    def _token_to_idx(self, token: str) -> int:
        """Deterministic hash to a positive 32-bit integer index."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        return (h % self.max_vocab_hash) + 1

    @staticmethod
    def tokens(text: str) -> List[str]:
        from core.retrieval.tokenize import thai_tokens
        return thai_tokens(text)

    def _sparse(self, weights: Dict[str, float]) -> Tuple[List[int], List[float]]:
        by_idx: Dict[int, float] = {}
        for token, w in weights.items():
            idx = self._token_to_idx(token)
            by_idx[idx] = by_idx.get(idx, 0.0) + w
        indices = sorted(by_idx)  # Qdrant requires ascending indices
        return indices, [round(by_idx[i], 5) for i in indices]

    def vectorize_document(self, text: str) -> Tuple[List[int], List[float]]:
        """BM25 term weights (saturated tf, length-normalised) for an indexed text."""
        toks = self.tokens(text)
        if not toks:
            return [], []
        norm = self.K1 * (1 - self.B + self.B * len(toks) / self.avg_doc_len)
        weights = {t: tf * (self.K1 + 1) / (tf + norm) for t, tf in Counter(toks).items()}
        return self._sparse(weights)

    def vectorize_query(self, text: str) -> Tuple[List[int], List[float]]:
        """Weight 1 per distinct query term; Qdrant multiplies in the IDF."""
        return self._sparse({t: 1.0 for t in set(self.tokens(text))})


class QdrantRepository:
    """Manages Qdrant collections, vector indexing, and hybrid searches."""

    CHUNKS_COLLECTION = "procurement_chunks"
    CASES_COLLECTION = "procurement_faq"
    # Collections from before the chunk/FAQ rename; dropped by drop_legacy_collections()
    LEGACY_COLLECTIONS = ("procurement_statutes", "procurement_cases")
    TENANT_DOCS_COLLECTION = "tenant_documents"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        api_key: Optional[str] = None,
        dense_dim: int = 1024,
    ):
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", 6333))
        self.grpc_port = int(grpc_port or os.getenv("QDRANT_GRPC_PORT", 6334))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY") or None
        self.dense_dim = int(dense_dim or os.getenv("QDRANT_DENSE_DIM", 1024))

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
            api_key=self.api_key,
            prefer_grpc=False,
            timeout=30.0,
        )
        self.vectorizer = ThaiSparseVectorizer()

    def init_collections(self):
        """Idempotently create collections and payload indexes."""
        existing = [c.name for c in self.client.get_collections().collections]

        # 1. Corpus chunks
        if self.CHUNKS_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=self.CHUNKS_COLLECTION,
                vectors_config={
                    "dense_bge_m3": models.VectorParams(
                        size=self.dense_dim,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
                    )
                },
                sparse_vectors_config={
                    # Qdrant applies IDF at query time; ThaiSparseVectorizer supplies the BM25 tf and length weights
                    "sparse_bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False), modifier=models.Modifier.IDF
                    )
                },
            )
            logger.info(f"Created Qdrant collection '{self.CHUNKS_COLLECTION}'.")

        # 2. FAQ pairs
        if self.CASES_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=self.CASES_COLLECTION,
                vectors_config={
                    "dense_bge_m3": models.VectorParams(
                        size=self.dense_dim,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
                    )
                },
                sparse_vectors_config={
                    "sparse_bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False), modifier=models.Modifier.IDF
                    )
                },
            )
            logger.info(f"Created Qdrant collection '{self.CASES_COLLECTION}'.")

        # 3. Tenant-private documents (OCR output)
        if self.TENANT_DOCS_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=self.TENANT_DOCS_COLLECTION,
                vectors_config={
                    "dense_bge_m3": models.VectorParams(
                        size=self.dense_dim,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
                    )
                },
                sparse_vectors_config={
                    "sparse_bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"Created Qdrant collection '{self.TENANT_DOCS_COLLECTION}'.")

        # Create Payload Field Indexes for fast filtering
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        """Create payload indexes on frequently filtered fields, including tenant org_id."""
        fields_chunks = [
            ("org_id", models.PayloadSchemaType.KEYWORD),
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("entry", models.PayloadSchemaType.KEYWORD),
            ("section_num", models.PayloadSchemaType.INTEGER),
            ("clause_num", models.PayloadSchemaType.INTEGER),
            ("chapter_num", models.PayloadSchemaType.INTEGER),
            ("kind", models.PayloadSchemaType.KEYWORD),
        ]
        for f_name, f_type in fields_chunks:
            try:
                self.client.create_payload_index(
                    collection_name=self.CHUNKS_COLLECTION,
                    field_name=f_name,
                    field_schema=f_type,
                )
            except Exception:
                pass

        fields_cases = [
            ("org_id", models.PayloadSchemaType.KEYWORD),
            ("case_id", models.PayloadSchemaType.KEYWORD),
            ("topics", models.PayloadSchemaType.KEYWORD),
        ]
        for f_name, f_type in fields_cases:
            try:
                self.client.create_payload_index(
                    collection_name=self.CASES_COLLECTION,
                    field_name=f_name,
                    field_schema=f_type,
                )
            except Exception:
                pass

        for f_name in ("org_id", "doc_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.TENANT_DOCS_COLLECTION,
                    field_name=f_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def upsert_chunk_points(
        self,
        chunks: List[Dict[str, Any]],
        dense_embeddings: List[List[float]],
        org_id: str = "PUBLIC",
        batch_size: int = 100,
    ):
        """Upsert chunk points with both dense and sparse vectors and org_id payload."""
        points = []
        for clause, dense_emb in zip(chunks, dense_embeddings):
            cid = str(clause["chunk_id"])
            text_content = str(clause.get("content", ""))
            sparse_indices, sparse_values = self.vectorizer.vectorize_document(text_content)

            # Convert string ID to a deterministic UUID string for Qdrant compatibility if needed
            point_id = hashlib.md5(cid.encode("utf-8")).hexdigest()

            vector_dict = {
                "dense_bge_m3": [float(x) for x in dense_emb],
            }
            if sparse_indices:
                vector_dict["sparse_bm25"] = models.SparseVector(
                    indices=sparse_indices, values=sparse_values
                )

            payload = {
                "chunk_id": cid,
                "org_id": str(clause.get("org_id", org_id)),
                "doc_id": str(clause.get("doc_id", "")),
                "entry": str(clause.get("entry", "")),
                "section_num": clause.get("section_num"),
                "clause_num": clause.get("clause_num"),
                "chapter_num": clause.get("chapter_num"),
                "kind": clause.get("kind"),
                "preview_text": text_content[:300],
            }

            points.append(
                models.PointStruct(id=point_id, vector=vector_dict, payload=payload)
            )

        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            self.client.upsert(collection_name=self.CHUNKS_COLLECTION, points=chunk)
            logger.info(f"Upserted {len(chunk)} chunk points to Qdrant ({i + len(chunk)}/{len(points)}).")

    def upsert_case_points(
        self,
        cases: List[Dict[str, Any]],
        dense_embeddings: List[List[float]],
        org_id: str = "PUBLIC",
        batch_size: int = 100,
    ):
        """Upsert FAQ case points with dense and sparse vectors and org_id payload."""
        points = []
        for case, dense_emb in zip(cases, dense_embeddings):
            cs_id = str(case["case_id"])
            q_text = str(case.get("question", ""))
            a_text = str(case.get("answer", ""))
            combined = f"{q_text} {a_text}"
            sparse_indices, sparse_values = self.vectorizer.vectorize_document(combined)

            point_id = hashlib.md5(cs_id.encode("utf-8")).hexdigest()

            vector_dict = {
                "dense_bge_m3": [float(x) for x in dense_emb],
            }
            if sparse_indices:
                vector_dict["sparse_bm25"] = models.SparseVector(
                    indices=sparse_indices, values=sparse_values
                )

            payload = {
                "case_id": cs_id,
                "org_id": str(case.get("org_id", org_id)),
                "question": q_text,
                "topics": case.get("topics", []),
            }

            points.append(
                models.PointStruct(id=point_id, vector=vector_dict, payload=payload)
            )

        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            self.client.upsert(collection_name=self.CASES_COLLECTION, points=chunk)
            logger.info(f"Upserted {len(chunk)} case points to Qdrant.")

    def hybrid_search_chunks(
        self,
        query_text: str,
        query_dense: List[float],
        top_k: int = 10,
        doc_filter: Optional[str] = None,
        section_filter: Optional[int] = None,
        org_id: str = "DGA",
    ) -> List[Dict[str, Any]]:
        """
        Executes server-side Reciprocal Rank Fusion (RRF) between dense and sparse vectors in Qdrant,
        filtering by tenant (visible: PUBLIC + org_id).
        """
        tenant_filter = models.Filter(
            should=[
                models.FieldCondition(key="org_id", match=models.MatchValue(value="PUBLIC")),
                models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
            ]
        )

        must_conditions = [tenant_filter]
        if doc_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_filter)
                )
            )
        if section_filter is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="section_num", match=models.MatchValue(value=section_filter)
                )
            )
        query_filter = models.Filter(must=must_conditions)

        sparse_indices, sparse_values = self.vectorizer.vectorize_query(query_text)

        # Attempt server-side fusion if sparse query has terms
        if sparse_indices and len(sparse_indices) > 0:
            try:
                response = self.client.query_points(
                    collection_name=self.CHUNKS_COLLECTION,
                    prefetch=[
                        models.Prefetch(
                            query=query_dense,
                            using="dense_bge_m3",
                            limit=max(30, top_k * 3),
                            filter=query_filter,
                        ),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_indices, values=sparse_values
                            ),
                            using="sparse_bm25",
                            limit=max(30, top_k * 3),
                            filter=query_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
                results = []
                for pt in response.points:
                    results.append({
                        "chunk_id": pt.payload.get("chunk_id"),
                        "score": pt.score,
                        "payload": pt.payload,
                    })
                return results
            except Exception as e:
                logger.warning(f"Qdrant RRF fusion query failed, falling back to dense-only: {e}")

        # Fallback to pure dense search (client.search() was removed in qdrant-client 1.16)
        response = self.client.query_points(
            collection_name=self.CHUNKS_COLLECTION,
            query=query_dense,
            using="dense_bge_m3",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "chunk_id": pt.payload.get("chunk_id"),
                "score": pt.score,
                "payload": pt.payload,
            }
            for pt in response.points
        ]

    def hybrid_search_cases(
        self,
        query_text: str,
        query_dense: List[float],
        top_k: int = 5,
        org_id: str = "DGA",
    ) -> List[Dict[str, Any]]:
        """Hybrid search over FAQ precedent cases, filtering by tenant (PUBLIC + org_id)."""
        tenant_filter = models.Filter(
            should=[
                models.FieldCondition(key="org_id", match=models.MatchValue(value="PUBLIC")),
                models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)),
            ]
        )
        sparse_indices, sparse_values = self.vectorizer.vectorize_query(query_text)
        if sparse_indices:
            try:
                response = self.client.query_points(
                    collection_name=self.CASES_COLLECTION,
                    prefetch=[
                        models.Prefetch(
                            query=query_dense,
                            using="dense_bge_m3",
                            limit=max(15, top_k * 3),
                            filter=tenant_filter,
                        ),
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_indices, values=sparse_values
                            ),
                            using="sparse_bm25",
                            limit=max(15, top_k * 3),
                            filter=tenant_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
                return [
                    {"case_id": pt.payload.get("case_id"), "score": pt.score, "payload": pt.payload}
                    for pt in response.points
                ]
            except Exception:
                pass

        # Dense-only fallback (client.search() was removed in qdrant-client 1.16)
        response = self.client.query_points(
            collection_name=self.CASES_COLLECTION,
            query=query_dense,
            using="dense_bge_m3",
            query_filter=tenant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {"case_id": pt.payload.get("case_id"), "score": pt.score, "payload": pt.payload}
            for pt in response.points
        ]

    @staticmethod
    def _own_tenant_filter(org_id: str, doc_id: Optional[str] = None) -> models.Filter:
        """Tenant documents are private: only the owner's org_id matches (no PUBLIC)."""
        must = [models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id))]
        if doc_id:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        return models.Filter(must=must)

    def upsert_tenant_chunks(
        self,
        org_id: str,
        doc_id: str,
        title: str,
        chunks: List[Dict[str, Any]],
        dense_embeddings: List[List[float]],
        batch_size: int = 100,
    ):
        """Replace a tenant document's points: drop the old ones, then upsert the new chunks."""
        self.delete_tenant_document(org_id, doc_id)
        points = []
        for chunk, dense_emb in zip(chunks, dense_embeddings):
            sparse_indices, sparse_values = self.vectorizer.vectorize_document(f"{title} {chunk['content']}")
            vector_dict = {"dense_bge_m3": [float(x) for x in dense_emb]}
            if sparse_indices:
                vector_dict["sparse_bm25"] = models.SparseVector(indices=sparse_indices, values=sparse_values)
            points.append(models.PointStruct(
                id=hashlib.md5(chunk["chunk_id"].encode("utf-8")).hexdigest(),
                vector=vector_dict,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "org_id": org_id,
                    "doc_id": doc_id,
                    "title": title,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                },
            ))
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.TENANT_DOCS_COLLECTION, points=points[i:i + batch_size])

    def drop_legacy_collections(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        for name in self.LEGACY_COLLECTIONS:
            if name in existing:
                self.client.delete_collection(name)
                logger.info("Dropped legacy Qdrant collection '%s'.", name)

    def delete_corpus(self, org_id: str) -> None:
        """Remove one tenant's seeded chunk and FAQ points before a re-ingest."""
        own = models.Filter(must=[models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id))])
        for collection in (self.CHUNKS_COLLECTION, self.CASES_COLLECTION):
            self.client.delete(collection_name=collection, points_selector=models.FilterSelector(filter=own), wait=True)

    def delete_tenant_document(self, org_id: str, doc_id: str):
        self.client.delete(
            collection_name=self.TENANT_DOCS_COLLECTION,
            points_selector=models.FilterSelector(filter=self._own_tenant_filter(org_id, doc_id)),
            wait=True,
        )

    def hybrid_search_tenant_chunks(
        self,
        query_text: str,
        query_dense: List[float],
        org_id: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """RRF over dense + sparse vectors, restricted to the tenant's own documents."""
        tenant_filter = self._own_tenant_filter(org_id)
        sparse_indices, sparse_values = self.vectorizer.vectorize_query(query_text)
        prefetch = [models.Prefetch(query=query_dense, using="dense_bge_m3", limit=max(30, top_k * 3), filter=tenant_filter)]
        if sparse_indices:
            prefetch.append(models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse_bm25",
                limit=max(30, top_k * 3),
                filter=tenant_filter,
            ))
        response = self.client.query_points(
            collection_name=self.TENANT_DOCS_COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [{"chunk_id": pt.payload.get("chunk_id"), "score": pt.score, "payload": pt.payload} for pt in response.points]

    def count_stats(self) -> Dict[str, int]:
        """Count total points in each collection."""
        try:
            chunk_cnt = self.client.count(collection_name=self.CHUNKS_COLLECTION).count
        except Exception:
            chunk_cnt = 0
        try:
            cases_cnt = self.client.count(collection_name=self.CASES_COLLECTION).count
        except Exception:
            cases_cnt = 0
        return {
            "chunk_points": chunk_cnt,
            "cases_points": cases_cnt,
        }
