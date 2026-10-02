# -*- coding: utf-8 -*-
"""
core/database/qdrant_repository.py

Qdrant Vector Database Repository for Thai Procurement LegalGraphRAG.
Provides 1024-dim dense vector search (BGE-M3) and native sparse vector search (BM25)
with server-side Reciprocal Rank Fusion (RRF) and metadata payload filtering.
"""

import os
import re
import math
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
    Computes sparse vector representations for Thai text using PyThaiNLP.
    Produces deterministic (indices, values) pairs compatible with Qdrant Sparse Vectors.
    """

    def __init__(self, max_vocab_hash: int = 1000000):
        self.max_vocab_hash = max_vocab_hash
        try:
            from pythainlp.tokenize import word_tokenize
            self._tokenize = lambda t: word_tokenize(t, engine="newmm")
        except ImportError:
            self._tokenize = lambda t: re.findall(r"\w+", t)

    def _token_to_idx(self, token: str) -> int:
        """Deterministic hash to a positive 32-bit integer index."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
        return (h % self.max_vocab_hash) + 1

    def vectorize(self, text: str) -> Tuple[List[int], List[float]]:
        """Converts Thai text into sparse index and TF-IDF weighted values."""
        if not text or not str(text).strip():
            return [], []
        
        tokens = [t.strip() for t in self._tokenize(str(text)) if len(t.strip()) > 1]
        if not tokens:
            return [], []

        counts = Counter(tokens)
        total_tokens = len(tokens)

        # Term frequency with sub-linear scaling (1 + log(tf))
        idx_val_map: Dict[int, float] = {}
        for token, count in counts.items():
            idx = self._token_to_idx(token)
            tf = 1.0 + math.log(count) if count > 0 else 0.0
            idx_val_map[idx] = idx_val_map.get(idx, 0.0) + tf

        # Sort indices ascending as required by sparse vector specifications
        sorted_indices = sorted(idx_val_map.keys())
        values = [round(idx_val_map[idx], 4) for idx in sorted_indices]
        return sorted_indices, values


class QdrantRepository:
    """Manages Qdrant collections, vector indexing, and hybrid searches."""

    STATUTES_COLLECTION = "procurement_statutes"
    CASES_COLLECTION = "procurement_cases"

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

        # 1. Procurement Statutes Collection
        if self.STATUTES_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=self.STATUTES_COLLECTION,
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
            logger.info(f"Created Qdrant collection '{self.STATUTES_COLLECTION}'.")

        # 2. Procurement Cases Collection
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
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"Created Qdrant collection '{self.CASES_COLLECTION}'.")

        # Create Payload Field Indexes for fast filtering
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        """Create payload indexes on frequently filtered fields, including tenant org_id."""
        fields_statutes = [
            ("org_id", models.PayloadSchemaType.KEYWORD),
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("entry", models.PayloadSchemaType.KEYWORD),
            ("section_num", models.PayloadSchemaType.INTEGER),
            ("clause_num", models.PayloadSchemaType.INTEGER),
            ("chapter_num", models.PayloadSchemaType.INTEGER),
            ("topics", models.PayloadSchemaType.KEYWORD),
        ]
        for f_name, f_type in fields_statutes:
            try:
                self.client.create_payload_index(
                    collection_name=self.STATUTES_COLLECTION,
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

    def upsert_statute_points(
        self,
        clauses: List[Dict[str, Any]],
        dense_embeddings: List[List[float]],
        org_id: str = "PUBLIC",
        batch_size: int = 100,
    ):
        """Upsert statutory clause points with both dense and sparse vectors and org_id payload."""
        points = []
        for clause, dense_emb in zip(clauses, dense_embeddings):
            cid = str(clause["clause_id"])
            text_content = str(clause.get("content_thai", ""))
            sparse_indices, sparse_values = self.vectorizer.vectorize(
                f"{clause.get('entry', '')} {text_content}"
            )

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
                "clause_id": cid,
                "org_id": str(clause.get("org_id", org_id)),
                "doc_id": str(clause.get("doc_id", "")),
                "entry": str(clause.get("entry", "")),
                "section_num": clause.get("section_num"),
                "clause_num": clause.get("clause_num"),
                "chapter_num": clause.get("chapter_num"),
                "topics": clause.get("topics", []),
                "preview_text": text_content[:300],
            }

            points.append(
                models.PointStruct(id=point_id, vector=vector_dict, payload=payload)
            )

        for i in range(0, len(points), batch_size):
            chunk = points[i:i + batch_size]
            self.client.upsert(collection_name=self.STATUTES_COLLECTION, points=chunk)
            logger.info(f"Upserted {len(chunk)} statute points to Qdrant ({i + len(chunk)}/{len(points)}).")

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
            sparse_indices, sparse_values = self.vectorizer.vectorize(combined)

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

    def hybrid_search_statutes(
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

        sparse_indices, sparse_values = self.vectorizer.vectorize(query_text)

        # Attempt server-side fusion if sparse query has terms
        if sparse_indices and len(sparse_indices) > 0:
            try:
                response = self.client.query_points(
                    collection_name=self.STATUTES_COLLECTION,
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
                        "clause_id": pt.payload.get("clause_id"),
                        "score": pt.score,
                        "payload": pt.payload,
                    })
                return results
            except Exception as e:
                logger.warning(f"Qdrant RRF fusion query failed, falling back to dense-only: {e}")

        # Fallback to pure dense search (client.search() was removed in qdrant-client 1.16)
        response = self.client.query_points(
            collection_name=self.STATUTES_COLLECTION,
            query=query_dense,
            using="dense_bge_m3",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "clause_id": pt.payload.get("clause_id"),
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
        sparse_indices, sparse_values = self.vectorizer.vectorize(query_text)
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

    def count_stats(self) -> Dict[str, int]:
        """Count total points in each collection."""
        try:
            statutes_cnt = self.client.count(collection_name=self.STATUTES_COLLECTION).count
        except Exception:
            statutes_cnt = 0
        try:
            cases_cnt = self.client.count(collection_name=self.CASES_COLLECTION).count
        except Exception:
            cases_cnt = 0
        return {
            "statutes_points": statutes_cnt,
            "cases_points": cases_cnt,
        }
