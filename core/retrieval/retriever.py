# -*- coding: utf-8 -*-
"""
core/retrieval/retriever.py

One retrieval pipeline for every caller (agent workflow, /search, MCP), over either store:

    1. Recall    every query variant (question, sub-questions, refined query) -> store.search
                 (dense + sparse fused by the store); lists are fused across variants with RRF.
                 No cross-encoder here.
    2. Seeds     the top `seeds` fused candidates.
    3. Expand    one batched graph call: clauses the seeds cite or derive authority from.
    4. Rerank    one cross-encoder pass over (variant, seed) pairs; a seed's score is its best
                 score over all variants, so a chunk answering a secondary sub-question is not
                 judged against the main question only.
    5. Assemble  the best `per_query` seeds for each variant (every sub-question gets room),
                 then the remaining seeds by score, then each kept seed's graph neighbours
                 (at most `graph_per_seed`). Graph neighbours are carried by their seed instead of
                 being reranked: they are relevant through the citation, not through wording.

Stores: TriStore (Qdrant + PostgreSQL + Neo4j) and InMemoryStore (outputs/corpus, BM25 + optional
dense, citation graph from the shared linker) for local PoC runs without Docker.
"""

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.utils.settings import env

logger = logging.getLogger(__name__)

SEARCH = "search"
GRAPH = "graph"


@dataclass
class RetrieverConfig:
    search_top_k: int = 40       # per query variant, from the store
    seeds: int = 40              # fused candidates that are reranked and expanded
    per_query: int = 4           # guaranteed slots per query variant
    top_k: int = 15              # reranked seeds returned (the generator reads 15)
    graph_per_seed: int = 2      # graph neighbours carried per kept seed
    rrf_k: int = 60
    rerank_threshold: float = 0.20
    rerank_chars: int = 1200

    @classmethod
    def from_env(cls) -> "RetrieverConfig":
        return cls(
            search_top_k=int(env("RETRIEVE_SEARCH_TOP_K", "40")),
            seeds=int(env("RETRIEVE_SEEDS", "40")),
            per_query=int(env("RETRIEVE_PER_QUERY", "4")),
            top_k=int(env("RETRIEVE_TOP_K", "15")),
            graph_per_seed=int(env("RETRIEVE_GRAPH_PER_SEED", "2")),
            rrf_k=int(env("RRF_K", "60")),
            rerank_threshold=float(env("RERANKER_THRESHOLD", "0.20")),
        )


@dataclass
class Candidate:
    chunk_id: str
    entry: str                       # '<document> | <label>'
    text: str
    source_type: str = "statute"     # statute | tenant_document
    origin: str = SEARCH             # search | graph
    fused_score: float = 0.0
    rerank_score: Optional[float] = None
    best_query: Optional[int] = None
    seed_id: Optional[str] = None    # for graph neighbours: the seed that cites them
    relation: Optional[str] = None   # CITES_CLAUSE | EMPOWERED_BY
    data: Dict[str, Any] = field(default_factory=dict)

    def to_law(self) -> Dict[str, Any]:
        """Dict shape the workflow, synthesizer, guardrail and service already consume."""
        d = self.data
        return {
            "id": self.chunk_id,
            "clause_id": self.chunk_id,
            "entry": self.entry,
            "description": self.text,
            "text": f"[{self.entry}]\n{self.text}",
            "crimes": d.get("topics", []),
            "judge_dep": d.get("judge_dep", []),
            "related_laws": d.get("related_laws", []),
            "insights": "",
            "rerank_score": float(self.rerank_score if self.rerank_score is not None else self.fused_score),
            "source_type": self.source_type,
            "retrieval_origin": self.origin,
            "seed_id": self.seed_id,
            "relation": self.relation,
            "data": d,
        }


@dataclass
class RetrievalResult:
    candidates: List[Candidate]
    queries: List[str]
    stats: Dict[str, Any] = field(default_factory=dict)

    def laws(self) -> List[Dict[str, Any]]:
        return [c.to_law() for c in self.candidates]


# ----------------------------------------------------------------------------- stores

class TriStore:
    """Qdrant hybrid search hydrated from PostgreSQL; citation expansion from Neo4j."""

    def __init__(self, storage=None):
        if storage is None:
            from core.database import StorageManager
            storage = StorageManager.get_instance()
        self.storage = storage

    def search(self, query: str, vector: List[float], org_id: str, top_k: int) -> List[Candidate]:
        out = []
        for r in self.storage.hybrid_search_clauses(query_text=query, query_dense=vector, top_k=top_k, org_id=org_id):
            out.append(Candidate(chunk_id=r["clause_id"], entry=r.get("entry", ""), text=r.get("content_thai", ""),
                                 fused_score=float(r.get("score", 0.0)), data=r))
        try:
            for t in self.storage.hybrid_search_tenant_docs(query_text=query, query_dense=vector, org_id=org_id,
                                                            top_k=max(top_k // 3, 5)):
                page = t.get("page_start")
                entry = f"{t.get('title', '')} | หน้า {page}" if page else t.get("title", "")
                out.append(Candidate(chunk_id=t["chunk_id"], entry=entry, text=t.get("content", ""),
                                     source_type="tenant_document", fused_score=float(t.get("score", 0.0)), data=t))
        except Exception as e:  # tenant collection may not exist yet; statutes still answer
            logger.warning("Tenant document search failed for %s: %s", org_id, e)
        out.sort(key=lambda c: c.fused_score, reverse=True)
        return out

    def expand(self, seeds: Sequence[Candidate], org_id: str) -> Dict[str, List[Candidate]]:
        clause_ids = [s.chunk_id for s in seeds if s.source_type == "statute"]
        tenant_ids = [s.chunk_id for s in seeds if s.source_type == "tenant_document"]
        edges = self.storage.neo4j.get_cited_clauses_batch(clause_ids, org_id=org_id) if clause_ids else []
        if tenant_ids:
            edges += self.storage.neo4j.get_tenant_chunk_citations_batch(tenant_ids, org_id=org_id)
        targets = list({e["target_id"] for e in edges})
        rows = {r["clause_id"]: r for r in self.storage.pg.get_clauses_by_ids(targets, org_id=org_id)} if targets else {}
        by_seed: Dict[str, List[Candidate]] = {}
        for e in edges:
            r = rows.get(e["target_id"])
            if r:
                by_seed.setdefault(e["source_id"], []).append(Candidate(
                    chunk_id=r["clause_id"], entry=r.get("entry", ""), text=r.get("content_thai", ""),
                    origin=GRAPH, seed_id=e["source_id"], relation=e.get("rel_type"), data=r))
        return by_seed


class InMemoryStore:
    """Local PoC store over outputs/corpus: BM25 (+ BGE-M3 dense when an embedder is available),
    fused with RRF; citation edges from the same linker the migration uses."""

    def __init__(self, laws_path: Optional[str] = None, embed_batch: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 cache_dir: str = os.path.join("outputs", "corpus")):
        import hashlib
        import json
        from rank_bm25 import BM25Okapi
        from core.chunking.corpus import ensure_corpus
        from core.graph.citation_linker import ACT_TITLE, extract_legal_edges
        from core.retrieval.reranker import ThaiBM25Index

        laws_path = laws_path or env("LAW_TO_CRIME_PATH", os.path.join(cache_dir, "law_to_crime.json"))
        if not os.path.exists(laws_path):
            ensure_corpus(env("OCR_DIR", "data_ocr"), os.path.dirname(laws_path) or ".")
        with open(laws_path, encoding="utf-8") as f:
            records = json.load(f)

        self.items: List[Candidate] = []
        units = []
        for r in records:
            entry = str(r.get("id", ""))
            text = str((r.get("items") or [{}])[0].get("text", ""))
            cid = "clause_" + hashlib.md5(entry.encode("utf-8")).hexdigest()[:16]
            self.items.append(Candidate(chunk_id=cid, entry=entry, text=text,
                                        data={"entry": entry, "content_thai": text, "page_start": r.get("page_start"),
                                              "page_end": r.get("page_end"), "source_file": r.get("source_file"),
                                              "total_pages": r.get("total_pages")}))
            units.append({"id": cid, "doc_name": r.get("doc_name", entry.split("|")[0].strip()),
                          "label": entry.partition("|")[2].strip(), "text": text})
        self.index = {c.chunk_id: i for i, c in enumerate(self.items)}

        self.tokenize = ThaiBM25Index.tokenize
        self.bm25 = BM25Okapi([self.tokenize(c.text) for c in self.items])

        self.matrix = None
        self.embed_batch = embed_batch
        if embed_batch is not None:
            texts = [c.text[:2500] for c in self.items]
            key = hashlib.md5("\x00".join(texts).encode("utf-8")).hexdigest()[:12]
            path = os.path.join(cache_dir, f"dense_{key}.npy")
            if os.path.exists(path):
                self.matrix = np.load(path)
            else:
                m = np.asarray(embed_batch(texts), dtype=np.float32)
                self.matrix = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
                np.save(path, self.matrix)

        self.cites: Dict[str, List[Tuple[str, str]]] = {}
        for e in extract_legal_edges(units, act_title=ACT_TITLE):
            if e["rel_type"] in ("CITES", "EMPOWERED_BY"):
                rel = "CITES_CLAUSE" if e["rel_type"] == "CITES" else "EMPOWERED_BY"
                self.cites.setdefault(e["source_id"], []).append((e["target_id"], rel))

    def search(self, query: str, vector: Optional[List[float]], org_id: str, top_k: int) -> List[Candidate]:
        sparse = np.asarray(self.bm25.get_scores(self.tokenize(query)))
        lists = [np.argsort(-sparse)[:max(top_k * 3, 30)]]
        if self.matrix is not None and vector is not None:
            v = np.asarray(vector, dtype=np.float32)
            dense = self.matrix @ (v / max(float(np.linalg.norm(v)), 1e-9))
            lists.append(np.argsort(-dense)[:max(top_k * 3, 30)])
        fused: Dict[int, float] = {}
        for ranked in lists:
            for rank, i in enumerate(ranked, 1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + rank)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for i, score in top:
            c = self.items[i]
            out.append(Candidate(chunk_id=c.chunk_id, entry=c.entry, text=c.text, fused_score=score, data=c.data))
        return out

    def expand(self, seeds: Sequence[Candidate], org_id: str) -> Dict[str, List[Candidate]]:
        by_seed: Dict[str, List[Candidate]] = {}
        for s in seeds:
            for target, rel in self.cites.get(s.chunk_id, []):
                t = self.items[self.index[target]]
                by_seed.setdefault(s.chunk_id, []).append(Candidate(
                    chunk_id=t.chunk_id, entry=t.entry, text=t.text, origin=GRAPH,
                    seed_id=s.chunk_id, relation=rel, data=t.data))
        return by_seed


# ----------------------------------------------------------------------------- reranking

class PairScorer:
    """Cross-encoder scores for (query, text) pairs in one batched pass, whichever backend is set."""

    def __init__(self, reranker):
        self.reranker = reranker
        self._lock = getattr(reranker, "_lock", None) or threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.reranker is not None and getattr(self.reranker, "model", True) is not None

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if hasattr(self.reranker, "_predict_pairs"):
            with self._lock:
                scores = self.reranker._predict_pairs(pairs, batch_size=8)
                if "cuda" in str(getattr(self.reranker, "device", "")):
                    import torch
                    torch.cuda.empty_cache()  # 4 GB laptop GPUs run out of memory between requests otherwise
            return [float(s) for s in scores]
        # API rerankers only expose rerank(query, candidates): group the pairs by query
        out = [0.0] * len(pairs)
        by_query: Dict[str, List[int]] = {}
        for i, (q, _) in enumerate(pairs):
            by_query.setdefault(q, []).append(i)
        for q, idxs in by_query.items():
            cands = [{"text": pairs[i][1], "_i": i} for i in idxs]
            for r in self.reranker.rerank(q, cands, top_k=len(cands), threshold=-math.inf):
                out[r["_i"]] = float(r["rerank_score"])
        return out


# ----------------------------------------------------------------------------- retriever

class Retriever:
    def __init__(self, store, embed: Callable[[str], Optional[List[float]]], reranker=None,
                 config: Optional[RetrieverConfig] = None):
        self.store = store
        self.embed = embed
        self.scorer = PairScorer(reranker) if reranker is not None else None
        self.config = config or RetrieverConfig.from_env()

    def retrieve(self, queries: Sequence[str], org_id: str, top_k: Optional[int] = None) -> RetrievalResult:
        cfg = self.config
        top_k = top_k or cfg.top_k
        queries = [q.strip() for q in dict.fromkeys(queries) if q and q.strip()]
        if not queries:
            return RetrievalResult([], [], {})

        # 1. Recall per variant, fused across variants
        fused: Dict[str, Candidate] = {}
        for qi, q in enumerate(queries):
            vector = self.embed(q)
            for rank, c in enumerate(self.store.search(q, vector, org_id, cfg.search_top_k), 1):
                hit = fused.setdefault(c.chunk_id, c)
                if hit is not c:
                    hit.fused_score += 1.0 / (cfg.rrf_k + rank)
                else:
                    c.fused_score = 1.0 / (cfg.rrf_k + rank)
                    c.best_query = qi
        ranked = sorted(fused.values(), key=lambda c: c.fused_score, reverse=True)

        # 2. Seeds
        seeds = ranked[:cfg.seeds]

        # 3. Graph expansion (one batched call)
        try:
            neighbours = self.store.expand(seeds, org_id)
        except Exception as e:
            logger.warning("Graph expansion failed; continuing with search results only: %s", e)
            neighbours = {}

        # 4. One cross-encoder pass over (variant, seed)
        per_query_scores = None
        reranked = False
        if self.scorer is not None and self.scorer.enabled and seeds:
            # Chunk text already opens with its '[document | section]' header
            pairs = [(q, s.text[:cfg.rerank_chars]) for s in seeds for q in queries]
            try:
                flat = self.scorer.score(pairs)
                per_query_scores = np.asarray(flat, dtype=np.float64).reshape(len(seeds), len(queries))
                for s, row in zip(seeds, per_query_scores):
                    s.rerank_score = float(row.max())
                    s.best_query = int(row.argmax())
                reranked = True
            except Exception as e:
                logger.warning("Reranking failed; falling back to fused order: %s", e)

        # 5. Assemble: guaranteed slots per variant, then best remaining, then carried neighbours
        if reranked:
            passing = [i for i, s in enumerate(seeds) if s.rerank_score >= cfg.rerank_threshold]
            if not passing:  # keep the single best rather than return nothing
                passing = [int(np.argmax([s.rerank_score for s in seeds]))]
            chosen: List[int] = []
            for qi in range(len(queries)):
                order = sorted(passing, key=lambda i: per_query_scores[i, qi], reverse=True)
                for i in order[:cfg.per_query]:
                    if i not in chosen:
                        chosen.append(i)
            for i in sorted(passing, key=lambda i: seeds[i].rerank_score, reverse=True):
                if i not in chosen:
                    chosen.append(i)
            chosen = sorted(chosen[:max(top_k, cfg.per_query * len(queries))],
                            key=lambda i: seeds[i].rerank_score, reverse=True)
            kept = [seeds[i] for i in chosen]
        else:
            kept = seeds[:top_k]

        # Each seed is followed directly by the clauses it cites, so an empowering section stays
        # next to the regulation clause that invokes it instead of sinking below every other seed
        result: List[Candidate] = []
        seen = {s.chunk_id for s in kept}
        carried = 0
        for s in kept:
            result.append(s)
            for n in neighbours.get(s.chunk_id, [])[:cfg.graph_per_seed]:
                if n.chunk_id not in seen:
                    # Ranked just below its seed's score so the generator sees it as supporting context
                    n.rerank_score = (s.rerank_score if s.rerank_score is not None else s.fused_score) - 1e-3
                    result.append(n)
                    seen.add(n.chunk_id)
                    carried += 1

        stats = {
            "queries": len(queries), "recalled": len(ranked), "seeds": len(seeds), "kept": len(kept),
            "graph_carried": carried, "reranked": reranked,
        }
        return RetrievalResult(result, queries, stats)


# ----------------------------------------------------------------------------- factory

_default: Optional[Retriever] = None
_default_lock = threading.Lock()


def tri_store_enabled() -> bool:
    return env("USE_TRI_STORE", "false").lower() in ("true", "1", "yes")


def get_retriever() -> Retriever:
    """Process-wide retriever: tri-store when USE_TRI_STORE is set, else the in-memory store."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                from core.retrieval.reranker import get_reranker, is_reranker_enabled
                from core.retrieval.search import get_embedding, batch_embed_tokenmind

                if tri_store_enabled():
                    store = TriStore()
                else:
                    use_dense = env("EMBEDDING_PROVIDER", "local").lower() == "tokenmind" and bool(env("TOKENMIND_API_KEY", ""))
                    store = InMemoryStore(embed_batch=batch_embed_tokenmind if use_dense else None)
                reranker = None
                if is_reranker_enabled():
                    reranker = get_reranker(model_name=env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                                            device=env("RERANKER_DEVICE", "cuda:0"))
                _default = Retriever(store, embed=get_embedding, reranker=reranker)
    return _default
