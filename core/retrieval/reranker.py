# -*- coding: utf-8 -*-
"""
hybrid_reranker.py

Implements Hybrid Retrieval (BM25 with PyThaiNLP + Dense Embedding)
coupled with GPU-accelerated Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
and relevance gating for ProcurementQA Agent.
"""

import os
import re
import sys
import json
import threading
from typing import List, Dict, Any, Tuple, Optional

import requests
import numpy as np
from pythainlp.tokenize import word_tokenize
from core.utils.settings import env

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
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            print(f"[GPUReranker] Initializing CrossEncoder '{self.model_name}' on '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            is_cuda = bool(torch and torch.cuda.is_available() and "cuda" in str(self.device))
            target_device = self.device if is_cuda else "cpu"
            target_dtype = torch.float16 if is_cuda else torch.float32

            # Use device_map directly to prevent PyTorch meta tensor copy exception
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                device_map=target_device,
                dtype=target_dtype,
                low_cpu_mem_usage=True
            )
            self.device = target_device
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
            text = c.get("text") or c.get("description") or ""
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


class OpperAPIReranker:
    """
    API-first Reranker using Opper AI Gateway (/v3/rerank).
    Zero local GPU/CPU memory footprint, EU-hosted, fast and scalable.
    Supports models like 'berget/bge-reranker-v2-m3' and 'aws/cohere/rerank-v3.5'.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "berget/bge-reranker-v2-m3",
        endpoint: str = "https://api.opper.ai/v3/rerank",
        threshold: float = 0.20,
        timeout: int = 15
    ):
        self.api_key = api_key or os.getenv("OPPER_API_KEY", "").strip()
        # Normalize model names to Opper's registered catalog identifiers
        clean_model = model_name.strip() if model_name else ""
        if not clean_model or clean_model in ["BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"]:
            self.model_name = "berget/bge-reranker-v2-m3"
        elif clean_model in ["cohere/rerank-v3.5", "rerank-v3.5"]:
            self.model_name = "aws/cohere/rerank-v3.5"
        else:
            self.model_name = clean_model

        self.endpoint = endpoint
        self.threshold = threshold
        self.timeout = timeout
        # Required for compatibility with `if reranker and reranker.model is not None` checks
        self.model = self.model_name
        self._warned_402 = False
        print(f"[OpperReranker] Initialized API-first reranker with model '{self.model_name}' on '{self.endpoint}'")

    def compute_score(self, query: str, text: str) -> float:
        res = self.rerank(query=query, candidates=[{"text": text}], top_k=1)
        if res and "rerank_score" in res[0]:
            return float(res[0]["rerank_score"])
        return 1.0

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidates using Opper's /v3/rerank API endpoint.
        Falls back gracefully without raising exceptions if the API is unreachable
        or if billing requires a card on file (HTTP 402).
        """
        if not candidates:
            return []

        gate_threshold = threshold if threshold is not None else self.threshold

        if not self.api_key:
            print("[OpperReranker] Warning: OPPER_API_KEY not found in environment. Falling back to unranked candidates.")
            return candidates[:top_k]

        valid_candidates = []
        documents = []
        for c in candidates:
            text = c.get("text") or c.get("description") or ""
            if text:
                valid_candidates.append(c)
                # Cap each document preview to 2,000 characters for optimal latency and token economy
                documents.append(str(text)[:2000])

        if not valid_candidates:
            return candidates[:top_k]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(len(documents), max(top_k * 2, 10)),
            "return_documents": False
        }

        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                reranked = []
                for item in results:
                    idx = item.get("index")
                    score = float(item.get("relevance_score", 0.0))
                    if idx is not None and idx < len(valid_candidates):
                        cand_copy = dict(valid_candidates[idx])
                        cand_copy["rerank_score"] = score
                        if score >= gate_threshold:
                            reranked.append(cand_copy)

                if not reranked and results:
                    # Fallback to top result if all fell below threshold
                    best_idx = results[0].get("index", 0)
                    best_cand = dict(valid_candidates[best_idx])
                    best_cand["rerank_score"] = float(results[0].get("relevance_score", 0.0))
                    reranked.append(best_cand)

                return reranked[:top_k]

            elif resp.status_code == 402:
                if not self._warned_402:
                    print(
                        f"[OpperReranker] HTTP 402 Payment Required: Please add a card at "
                        f"https://platform.opper.ai/go?to=billing to enable Opper Rerank API. "
                        f"Falling back gracefully to retriever ordering."
                    )
                    self._warned_402 = True
                return candidates[:top_k]
            else:
                print(f"[OpperReranker] API Error {resp.status_code}: {resp.text[:200]}. Falling back gracefully.")
                return candidates[:top_k]

        except Exception as e:
            print(f"[OpperReranker] Request failed: {e}. Falling back gracefully.")
            return candidates[:top_k]


# Global Manager
_global_bm25_index: Optional[ThaiBM25Index] = None
_global_reranker: Optional[Any] = None
_reranker_init_lock = threading.Lock()
_bm25_init_lock = threading.Lock()


def is_reranker_enabled() -> bool:
    """Check if cross-encoder reranker is enabled via environment variables."""
    val = env("RERANKER_ENABLED", "true").strip().lower()
    return val in ("true", "1", "yes")


def get_reranker(
    model_name: Optional[str] = None,
    device: str = "cuda:0",
    threshold: Optional[float] = None
) -> Optional[Any]:
    if not is_reranker_enabled():
        return None
    global _global_reranker
    if _global_reranker is None:
        with _reranker_init_lock:
            if _global_reranker is None:
                provider = env("RERANKER_PROVIDER", "").strip().lower()
                opper_key = os.getenv("OPPER_API_KEY", "").strip()

                # Default to Opper API if OPPER_API_KEY is present or reranker_provider is 'opper' / 'api'
                if provider in ("opper", "api") or (not provider and opper_key):
                    resolved_model = os.getenv("OPPER_RERANKER_MODEL") or model_name or "berget/bge-reranker-v2-m3"
                    thresh = float(env("RERANKER_THRESHOLD", threshold if threshold is not None else 0.20))
                    endpoint = os.getenv("OPPER_RERANK_URL", "https://api.opper.ai/v3/rerank")
                    _global_reranker = OpperAPIReranker(
                        api_key=opper_key,
                        model_name=resolved_model,
                        endpoint=endpoint,
                        threshold=thresh
                    )
                else:
                    resolved_model = model_name or env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
                    thresh = float(env("RERANKER_THRESHOLD", threshold if threshold is not None else 0.20))
                    dev = env("RERANKER_DEVICE", device)
                    _global_reranker = GPUReranker(
                        model_name=resolved_model,
                        device=dev,
                        threshold=thresh
                    )
    return _global_reranker



def get_bm25_index() -> ThaiBM25Index:
    global _global_bm25_index
    if _global_bm25_index is None:
        with _bm25_init_lock:
            if _global_bm25_index is None:
                _global_bm25_index = ThaiBM25Index()
    return _global_bm25_index


def expand_numeric_query(query: str, chatbot: Optional[Any] = None) -> str:
    """
    Generalized legal query expansion (Lexical & Concept Normalization).
    Strictly NO hardcoded section numbers or answer values.
    Transforms colloquial phrases, abbreviations, and procedural queries into formal legal terminology:
    - Acronym/Colloquial normalization: MiT / Made in Thailand -> พัสดุที่ผลิตภายในประเทศ, ส่งเสริมหรือสนับสนุน
    - Procedural concept normalization: ลงนามสัญญา -> การทำสัญญา, อุทธรณ์; บอกเลิกสัญญา -> การบอกเลิกสัญญา, ค่าปรับ
    - Quantitative inquiry indicators: กี่, เท่าใด -> เกณฑ์, อัตรา, สัดส่วน, กำหนด
    """
    if not query:
        return query
    
    boost_terms = []
    q_lower = query.lower()

    # 1. Acronym & Trade terms normalization (e.g. Made in Thailand / MiT / สินค้าไทย)
    if any(k in q_lower for k in ["made in thailand", "mit", "ผลิตในประเทศ", "ผลิตภายในประเทศ", "สินค้าไทย"]):
        boost_terms.extend(["พัสดุที่ผลิตภายในประเทศ", "ส่งเสริมหรือสนับสนุน", "แต้มต่อ"])

    # 2. Contract execution & Post-award concepts (ลงนาม / ทำสัญญา)
    if any(k in q_lower for k in ["ลงนาม", "ทำสัญญา", "ชนะการเสนอราคา", "ผู้ชนะ"]):
        boost_terms.extend(["การทำสัญญา", "การลงนามในสัญญา", "การอุทธรณ์"])

    # 3. Electronic Market / Procurement methods (e-market / ตลาดอิเล็กทรอนิกส์)
    if any(k in q_lower for k in ["e-market", "ตลาดอิเล็กทรอนิกส์"]):
        boost_terms.extend(["การจัดซื้อจัดจ้าง", "วิธีตลาดอิเล็กทรอนิกส์", "e-catalog"])

    # 4. Liquidated damages & Contract default (ค่าปรับ / ล่าช้า / ผิดสัญญา)
    if any(k in q_lower for k in ["ปรับ", "ล่าช้า", "ไม่ปฏิบัติตามสัญญา", "ทิ้งงาน"]):
        boost_terms.extend(["อัตราค่าปรับ", "การคิดค่าปรับ", "ค่าปรับรายวัน"])

    # 5. Contract termination (บอกเลิกสัญญา)
    if any(k in q_lower for k in ["บอกเลิก", "เลิกสัญญา"]):
        boost_terms.extend(["การบอกเลิกสัญญา", "การแก้ไขสัญญา", "สัญญาสิ้นสุด"])

    # 6. Selection method concepts (วิธีคัดเลือก)
    if any(k in q_lower for k in ["คัดเลือก", "เชิญชวน"]):
        boost_terms.extend(["วิธีคัดเลือก", "หนังสือเชิญชวน"])

    # 7. Committee formation concepts (คณะกรรมการ / องค์ประกอบ)
    if any(k in q_lower for k in ["คณะกรรมการ", "แต่งตั้ง", "ประธาน", "องค์ประกอบ"]):
        boost_terms.extend(["คณะกรรมการซื้อหรือจ้าง", "การแต่งตั้งคณะกรรมการ", "ประธานกรรมการ"])

    # 8. Standard cost & Financial factors (ราคากลาง / Factor F / ดอกเบี้ย)
    if any(k in q_lower for k in ["ราคากลาง", "factor f", "ดอกเบี้ย"]):
        boost_terms.extend(["หลักเกณฑ์ราคากลาง", "อัตราดอกเบี้ย", "การคำนวณราคากลาง"])

    # 9. Quantitative & constraint indicators (General terms only, no hardcoded values)
    num_triggers = [
        "กี่", "เท่าใด", "เท่าไหร่", "ร้อยละ", "เปอร์เซ็นต์", "%", "บาท", "วงเงิน",
        "อัตรา", "สัดส่วน", "วันทำการ", "กำหนดเวลา", "ปัดเศษ", "ไม่เกิน", "ไม่น้อยกว่า",
        "ขั้นต่ำ", "สูงสุด"
    ]
    if any(trig in query for trig in num_triggers):
        boost_terms.extend(["เกณฑ์", "อัตรา", "หลักเกณฑ์", "กำหนดไว้"])

    # 10. Specifications & Terms of Reference concepts (TOR / ขอบเขตของงาน / สเปก / ยี่ห้อ -> คุณลักษณะเฉพาะ)
    if any(k in q_lower for k in ["tor", "ขอบเขตของงาน", "สเปก", "สเปค", "ยี่ห้อ"]):
        boost_terms.extend(["คุณลักษณะเฉพาะ", "การกำหนดคุณลักษณะเฉพาะ"])

    if boost_terms:
        unique_terms = list(dict.fromkeys(boost_terms))
        return f"{query} {' '.join(unique_terms)}"
    return query


def extract_document_year(text: str) -> Optional[int]:
    """
    Extracts Buddhist Era year (พ.ศ. 25XX) from document title, entry or text dynamically.
    Works for any past, present, or future year (e.g. 2535, 2560, 2563, 2569, 2570+).
    """
    if not text:
        return None
    # Matches patterns like พ.ศ. 2560, พ.ศ.2563, ปี 2569, or standalone 25XX
    m = re.search(r"(?:พ\.ศ\.|ปี|\b)(25\d{2})\b", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def get_legal_hierarchy_multiplier(doc_id: str, entry: str, text: str) -> float:
    """
    Evaluates the legal authority hierarchy dynamically:
    - Primary Statutes / Regulations (พ.ร.บ., ระเบียบ, กฎกระทรวง, ประกาศคณะกรรมการ): Highest authority (1.20)
    - Administrative Guidelines / FAQ / Circulars: Standard authority (1.00)
    - Contract Templates / Blank Sample Forms: Lower priority for substantive statutory queries (0.75)
    """
    combined = f"{doc_id} {entry} {text[:300]}".lower()

    # If it is purely a contract template or form, it's illustrative rather than governing law
    if any(k in combined for k in ["แบบสัญญา", "แบบฟอร์ม", "สัญญาสำเร็จรูป", "ตัวอย่างสัญญา"]):
        return 0.75

    # Primary enacted statutes and ministerial regulations
    if any(k in combined for k in ["พระราชบัญญัติ", "ระเบียบกระทรวงการคลัง", "กฎกระทรวง", "ประกาศคณะกรรมการ"]):
        return 1.25

    return 1.0


def compute_dynamic_boost(
    doc: Dict[str, Any],
    max_corpus_year: int = 2569,
    base_statute_year: int = 2560
) -> float:
    """
    Computes a fully dynamic weight multiplier based on:
    1. Recency relative to the latest legal corpus year (e.g. 2569, 2570+)
    2. Primary legal authority hierarchy (Statute vs Guideline vs Template)
    """
    did = str(doc.get("id", ""))
    entry = str(doc.get("entry", ""))
    text = str(doc.get("text", "") or doc.get("description", ""))

    doc_year = extract_document_year(f"{did} {entry}")
    if doc_year is None:
        doc_year = extract_document_year(text[:400])

    # Explicit exclusion for repealed regulations prior to 2560 (e.g. 2535, 2549)
    if "2535" in did or "2535" in entry:
        return 0.0

    hierarchy_mult = get_legal_hierarchy_multiplier(did, entry, text)

    # Dynamic Recency Multiplier
    if doc_year is not None:
        if doc_year >= base_statute_year:
            # Active legal framework (e.g. 2560 up to max_corpus_year like 2569+)
            # The closer to max_corpus_year, the higher the weight
            year_diff = max(0, max_corpus_year - doc_year)
            recency_mult = max(1.05, 1.30 - (year_diff * 0.02))
        else:
            # Older repealed regulations prior to 2560 reform (e.g. 2535, 2549)
            # Section 3 of 2560 Act strictly repealed these; exclude completely from active QA
            return 0.0
    else:
        recency_mult = 1.0

    return hierarchy_mult * recency_mult


def weighted_rrf(
    dense_results: List[Tuple[Dict[str, Any], float]],
    sparse_results: List[Tuple[Dict[str, Any], float]],
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Weighted Reciprocal Rank Fusion (RRF) with Dynamic Recency & Legal Hierarchy Boosting.
    Combines dense cosine ranking and sparse BM25 ranking, dynamically prioritizing
    modern governing statutes over legacy regulations and blank contract templates.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Detect maximum year dynamically across the candidate set
    detected_years = []
    for cand_list in [dense_results, sparse_results]:
        for doc, _ in cand_list:
            y = extract_document_year(f"{doc.get('id', '')} {doc.get('entry', '')}")
            if y and 2500 <= y <= 2650:
                detected_years.append(y)
    max_year = max(detected_years) if detected_years else 2569

    # Process Dense results
    for rank, (doc, _) in enumerate(dense_results):
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            continue
        boost = compute_dynamic_boost(doc, max_corpus_year=max_year)
        if boost <= 0.0:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + (dense_weight * boost / (rrf_k + rank + 1))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Process Sparse results
    for rank, (doc, _) in enumerate(sparse_results):
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            continue
        boost = compute_dynamic_boost(doc, max_corpus_year=max_year)
        if boost <= 0.0:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + (sparse_weight * boost / (rrf_k + rank + 1))
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



def reranker_status() -> dict:
    """
    Whether the configured reranker is actually usable. A reranker that silently fails to load
    (e.g. a missing runtime dependency) makes retrieval fall back to unreranked order, which
    degrades answer quality without any error, so readiness probes must surface it.
    """
    enabled = is_reranker_enabled()
    reranker = _global_reranker
    return {
        "enabled": enabled,
        "provider": type(reranker).__name__ if reranker is not None else None,
        "loaded": (not enabled) or (reranker is not None and getattr(reranker, "model", None) is not None),
    }
