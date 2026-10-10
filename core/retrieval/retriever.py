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

def _unit_record(chunk_id: str, entry: str, text: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Store-independent record for an exact มาตรา/ข้อ lookup."""
    return {"chunk_id": chunk_id, "entry": entry, "content": text, "doc_title": row.get("doc_title"),
            "source_file": row.get("source_file"), "page_start": row.get("page_start"),
            "page_end": row.get("page_end"), "total_pages": row.get("total_pages")}


class TriStore:
    """Qdrant hybrid search hydrated from PostgreSQL; citation expansion from Neo4j."""

    def __init__(self, storage=None):
        if storage is None:
            from core.database import StorageManager
            storage = StorageManager.get_instance()
        self.storage = storage

    def search(self, query: str, vector: Optional[List[float]], org_id: str, top_k: int) -> List[Candidate]:
        if vector is None:
            raise RuntimeError("No query embedding (check EMBEDDING_PROVIDER / TOKENMIND_API_KEY); "
                               "the tri-store hybrid search needs the dense vector")
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

    def lookup_unit(self, kind: Optional[str], num: int, doc_hint: str, org_id: str) -> List[Dict[str, Any]]:
        """Chunks labelled 'มาตรา N' / 'ข้อ N', best match first (see PostgresRepository ordering)."""
        pg = self.storage.pg
        if kind == "clause":
            rows = pg.lookup_clause(doc_hint, num, org_id=org_id)
        else:
            rows = pg.lookup_section(doc_hint, num, org_id=org_id)
            if not rows and kind is None:
                rows = pg.lookup_clause(doc_hint, num, org_id=org_id)
        return [_unit_record(r.get("clause_id"), r.get("entry", ""), r.get("content_thai", ""), r) for r in rows]

    def related(self, chunk_id: str, org_id: str) -> List[Dict[str, Any]]:
        ctx = self.storage.traverse_clause_graph(chunk_id, org_id=org_id)
        out = [{"node_id": a.get("clause_id"), "relation": "ADJACENT_SECTION", "entry": a.get("entry", "")}
               for a in ctx.get("adjacent_sections", [])]
        out += [{"node_id": c.get("clause_id"), "relation": "CITES_CLAUSE", "entry": c.get("entry", "")}
                for c in ctx.get("cited_clauses", [])]
        out += [{"node_id": s.get("clause_id"), "relation": "SUBORDINATE_RULE", "entry": s.get("entry", "")}
                for s in ctx.get("subordinate_laws", [])]
        out += [{"node_id": c.get("case_id"), "relation": "RELATES_TO_LAW", "entry": c.get("question", "")}
                for c in ctx.get("related_cases", [])]
        return out

    def search_faq(self, query: str, vector: List[float], org_id: str, top_k: int) -> List[Dict[str, Any]]:
        hits = self.storage.hybrid_search_cases(query_text=query, query_dense=vector, top_k=top_k, org_id=org_id)
        rows = {r["case_id"]: r for r in self.storage.pg.get_faq_cases_by_ids([h["case_id"] for h in hits], org_id=org_id)}
        return [{"faq_id": h["case_id"], "question": rows.get(h["case_id"], {}).get("question", h.get("question", "")),
                 "answer": rows.get(h["case_id"], {}).get("answer", ""),
                 "cited": rows.get(h["case_id"], {}).get("cited_laws", []), "score": round(float(h["score"]), 4)}
                for h in hits]

    def catalog(self, org_id: str) -> Dict[str, Any]:
        stats = self.storage.get_stats()
        return {"documents": self.storage.pg.list_document_titles(org_id),
                "chunks": stats["postgres"].get("statute_clauses", 0),
                "faq_pairs": stats["postgres"].get("faq_cases", 0)}

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
    """Local PoC store over outputs/corpus/chunks.jsonl: BM25 (+ BGE-M3 dense when an embedder is
    available) fused with RRF; citation and reading-order edges from the same linker the migration
    uses. Implements the same interface as TriStore."""

    def __init__(self, chunks_path: Optional[str] = None,
                 embed_batch: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 cache_dir: Optional[str] = None):
        import hashlib
        import json
        from rank_bm25 import BM25Okapi
        from core.chunking.corpus import DEFAULT_CORPUS_DIR, ensure_corpus
        from core.graph.citation_linker import ACT_TITLE, extract_legal_edges
        from core.retrieval.reranker import ThaiBM25Index

        corpus_dir = cache_dir or env("CORPUS_DIR", DEFAULT_CORPUS_DIR)
        chunks_path = chunks_path or os.path.join(corpus_dir, "chunks.jsonl")
        ensure_corpus(env("OCR_DIR", "data_ocr"), os.path.dirname(chunks_path) or ".")
        with open(chunks_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        self.items: List[Candidate] = []
        self.meta: List[Dict[str, Any]] = []
        units = []
        for r in records:
            text = r.get("embed_text") or r["content"]
            data = {"entry": r["entry"], "content_thai": text, "page_start": r.get("page_start"),
                    "page_end": r.get("page_end"), "source_file": r.get("source_file"),
                    "total_pages": r.get("total_pages"), "doc_title": r.get("doc_title"), "kind": r.get("kind")}
            self.items.append(Candidate(chunk_id=r["chunk_id"], entry=r["entry"], text=text, data=data))
            self.meta.append(r)
            units.append({"id": r["chunk_id"], "doc_name": r["doc_title"], "label": r["label"], "text": text})
        self.index = {c.chunk_id: i for i, c in enumerate(self.items)}

        self.tokenize = ThaiBM25Index.tokenize
        self.bm25 = BM25Okapi([self.tokenize(c.text) for c in self.items])
        self.faq_idx = [i for i, m in enumerate(self.meta) if m.get("kind") == "faq"]

        self.matrix = None
        if embed_batch is not None:
            texts = [c.text[:2500] for c in self.items]
            key = hashlib.md5("\x00".join(texts).encode("utf-8")).hexdigest()[:12]
            path = os.path.join(os.path.dirname(chunks_path) or ".", f"dense_{key}.npy")
            if os.path.exists(path):
                self.matrix = np.load(path)
            else:
                m = np.asarray(embed_batch(texts), dtype=np.float32)
                self.matrix = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
                np.save(path, self.matrix)

        # Same relations as Neo4j: CITES_CLAUSE / EMPOWERED_BY (outgoing), ADJACENT_SECTION (both ways)
        self.cites: Dict[str, List[Tuple[str, str]]] = {}
        self.cited_by: Dict[str, List[Tuple[str, str]]] = {}
        self.adjacent: Dict[str, List[str]] = {}
        for e in extract_legal_edges(units, act_title=ACT_TITLE):
            src, tgt = e["source_id"], e["target_id"]
            if e["rel_type"] == "NEXT_SECTION":
                self.adjacent.setdefault(src, []).append(tgt)
                self.adjacent.setdefault(tgt, []).append(src)
            else:
                rel = "CITES_CLAUSE" if e["rel_type"] == "CITES" else "EMPOWERED_BY"
                self.cites.setdefault(src, []).append((tgt, rel))
                self.cited_by.setdefault(tgt, []).append((src, rel))

    def lookup_unit(self, kind: Optional[str], num: int, doc_hint: str, org_id: str) -> List[Dict[str, Any]]:
        from core.graph.citation_linker import parse_unit_label
        hint = (doc_hint or "").strip().lower()
        units_per_doc: Dict[str, int] = {}
        matches = []
        for i, m in enumerate(self.meta):
            k, n, part, _ = parse_unit_label(m.get("label", ""))
            if k:
                units_per_doc[m["doc_title"]] = units_per_doc.get(m["doc_title"], 0) + 1
            if n == num and (kind is None or k == kind) and k and (not hint or hint in m["doc_title"].lower()):
                matches.append((i, k, part))

        def rank(t):
            i, k, part = t
            title = self.meta[i]["doc_title"]
            # Same preferences as the PostgreSQL lookup: the Act for มาตรา, regulations for ข้อ,
            # then the most comprehensive document, then reading order
            doc_pref = 0 if (k == "section" and title.startswith("พระราชบัญญัติ")) or                             (k == "clause" and title.startswith("ระเบียบ")) else 1
            return (doc_pref, -units_per_doc.get(title, 0), part)

        out = []
        for i, _, _ in sorted(matches, key=rank):
            c, m = self.items[i], self.meta[i]
            out.append(_unit_record(c.chunk_id, c.entry, c.text, {**c.data, "doc_title": m["doc_title"]}))
        return out

    def related(self, chunk_id: str, org_id: str) -> List[Dict[str, Any]]:
        own_doc = self.meta[self.index[chunk_id]]["doc_title"] if chunk_id in self.index else None
        entry = lambda cid: self.items[self.index[cid]].entry
        out = [{"node_id": c, "relation": "ADJACENT_SECTION", "entry": entry(c)} for c in self.adjacent.get(chunk_id, [])]
        out += [{"node_id": c, "relation": "CITES_CLAUSE", "entry": entry(c)} for c, _ in self.cites.get(chunk_id, [])]
        for c, _ in self.cited_by.get(chunk_id, []):
            m = self.meta[self.index[c]]
            if m["doc_title"] == own_doc:
                continue
            # Same relation names as Neo4j: FAQ cases relate to the law, other documents derive from it
            out.append({"node_id": c, "relation": "RELATES_TO_LAW" if m.get("kind") == "faq" else "SUBORDINATE_RULE",
                        "entry": entry(c)})
        return out

    def search_faq(self, query: str, vector: Optional[List[float]], org_id: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.faq_idx:
            return []
        scores = np.asarray(self.bm25.get_scores(self.tokenize(query)))[self.faq_idx]
        out = []
        for j in np.argsort(-scores)[:top_k]:
            m = self.meta[self.faq_idx[j]]
            md = m.get("metadata", {})
            out.append({"faq_id": m["chunk_id"], "question": md.get("question", ""), "answer": md.get("answer", ""),
                        "cited": [], "score": round(float(scores[j]), 4)})
        return out

    def catalog(self, org_id: str) -> Dict[str, Any]:
        return {"documents": sorted({m["doc_title"] for m in self.meta}),
                "chunks": len(self.items), "faq_pairs": len(self.faq_idx)}

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
                from core.retrieval.embedding import batch_embed_tokenmind, get_embedding

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
