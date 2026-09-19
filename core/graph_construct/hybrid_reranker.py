# -*- coding: utf-8 -*-
"""
hybrid_reranker.py

Implements Hybrid Retrieval (BM25 with PyThaiNLP + Dense Embedding)
coupled with GPU-accelerated Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
and relevance gating for LegalGraphRAG.
"""

import os
import re
import sys
import threading
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from pythainlp.tokenize import word_tokenize

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    torch = None


class ThaiBM25Index:
    """In-memory BM25 index with PyThaiNLP tokenization."""

    def __init__(self):
        self.doc_ids: List[str] = []
        self.doc_types: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_objects: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []
        text_clean = re.sub(r"\s+", " ", str(text)).strip()
        # newmm is pure dictionary/C++ based: ultra-fast, robust, and zero PyTorch/meta-parameter warnings
        tokens = word_tokenize(text_clean, engine="newmm", keep_whitespace=False)
        return [t.strip().lower() for t in tokens if t.strip() and len(t.strip()) > 1]

    def build_index(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from documents.
        documents: List of dict with 'id', 'type', 'text'
        """
        if BM25Okapi is None:
            return

        self.doc_ids = []
        self.doc_types = []
        self.doc_texts = []
        self.doc_objects = []
        tokenized_corpus = []

        for doc in documents:
            text = doc.get("text") or doc.get("description") or ""
            if not text:
                continue
            self.doc_ids.append(doc["id"])
            self.doc_types.append(doc.get("type", "unknown"))
            self.doc_texts.append(text)
            self.doc_objects.append(doc)
            tokenized_corpus.append(self.tokenize(text))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        if not self.bm25 or not query:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_objects[idx], float(scores[idx])))
        return results


class GPUReranker:
    """Singleton GPU/CPU Cross-Encoder Reranker."""

    _instance: Optional["GPUReranker"] = None

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda:0",
        threshold: float = 0.20
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = self._resolve_device(device)
        self.model: Optional[CrossEncoder] = None
        self._lock = threading.Lock()
        self._init_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if torch and torch.cuda.is_available() and "cuda" in device:
            return device
        return "cpu"

    def _init_model(self):
        # Using HuggingFace Transformers AutoModel directly to completely bypass sentence_transformers meta-tensor bug
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            print(f"[GPUReranker] Initializing CrossEncoder '{self.model_name}' on '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            load_kwargs = {"low_cpu_mem_usage": False}
            if torch and torch.cuda.is_available() and "cuda" in str(self.device):
                load_kwargs["type"] = torch.float16
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **load_kwargs).to(self.device)
            else:
                self.device = "cpu"
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, **load_kwargs).to("cpu")
            self.model.eval()
            print(f"[GPUReranker] Successfully loaded reranker on '{self.device}'.")
        except Exception as e:
            print(f"[GPUReranker] Failed to load native reranker: {e}")
            self.model = None
            self.tokenizer = None

    def _predict_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> List[float]:
        if not self.model or not self.tokenizer:
            return [1.0] * len(pairs)
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                logits = self.model(**inputs).logits
                if logits.shape[-1] == 1:
                    batch_scores = logits.view(-1).cpu().tolist()
                else:
                    batch_scores = logits[:, 1].cpu().tolist()
                all_scores.extend(batch_scores)
        return all_scores

    def compute_score(self, query: str, text: str) -> float:
        if not self.model:
            return 1.0
        try:
            with self._lock:
                scores = self._predict_pairs([(query, text[:1000])], batch_size=1)
                return float(scores[0]) if scores else 1.0
        except Exception:
            return 1.0

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidates and applies relevance threshold gate.
        Safely predicts in batches of 8 with GPU memory cleanup to prevent CUDA OOM on 4GB VRAM.
        """
        if not candidates:
            return []

        gate_threshold = threshold if threshold is not None else self.threshold

        if not self.model:
            return candidates[:top_k]

        pairs = []
        valid_candidates = []
        for c in candidates:
            text = c.get("text") or c.get("description") or c.get("fact") or ""
            if text:
                pairs.append((query, str(text)[:1200]))
                valid_candidates.append(c)

        if not pairs:
            return candidates[:top_k]

        try:
            with self._lock:
                scores = self._predict_pairs(pairs, batch_size=8)
                if torch and torch.cuda.is_available() and "cuda" in str(self.device):
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"[GPUReranker] Error during reranking: {e}")
            return candidates[:top_k]

        reranked = []
        for cand, score in zip(valid_candidates, scores):
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            if float(score) >= gate_threshold:
                reranked.append(cand_copy)

        # Sort by rerank score descending
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Fallback: if all scores fell below threshold, keep at least the top candidate
        if not reranked and valid_candidates:
            best_idx = int(np.argmax(scores))
            fallback_cand = dict(valid_candidates[best_idx])
            fallback_cand["rerank_score"] = float(scores[best_idx])
            reranked.append(fallback_cand)

        return reranked[:top_k]


# Global Manager
_global_bm25_index: Optional[ThaiBM25Index] = None
_global_reranker: Optional[GPUReranker] = None
_reranker_init_lock = threading.Lock()
_bm25_init_lock = threading.Lock()


def get_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cuda:0",
    threshold: float = 0.20
) -> Optional[GPUReranker]:
    global _global_reranker
    if _global_reranker is None:
        with _reranker_init_lock:
            if _global_reranker is None:
                _global_reranker = GPUReranker(
                    model_name=model_name,
                    device=device,
                    threshold=threshold
                )
    return _global_reranker


def get_bm25_index() -> ThaiBM25Index:
    global _global_bm25_index
    if _global_bm25_index is None:
        with _bm25_init_lock:
            if _global_bm25_index is None:
                _global_bm25_index = ThaiBM25Index()
    return _global_bm25_index


def weighted_rrf(
    dense_results: List[Tuple[Dict[str, Any], float]],
    sparse_results: List[Tuple[Dict[str, Any], float]],
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Weighted Reciprocal Rank Fusion (RRF).
    Combines dense cosine ranking and sparse BM25 ranking.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Process Dense results
    for rank, (doc, _) in enumerate(dense_results):
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + (dense_weight / (rrf_k + rank + 1))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Process Sparse results
    for rank, (doc, _) in enumerate(sparse_results):
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + (sparse_weight / (rrf_k + rank + 1))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort documents by fused RRF score
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused_docs = []
    for doc_id in sorted_doc_ids:
        doc = dict(doc_map[doc_id])
        doc["rrf_score"] = scores[doc_id]
        fused_docs.append(doc)

    return fused_docs
