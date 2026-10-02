#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluation/evaluate_retriever.py

Standalone Benchmark & Evaluation Suite for Legal Retrieval Layer:
- Evaluates purely the retrieval engine (Zero LLM Generation cost, fast execution)
- Supports flexible ablation switches:
    * Search Mode: Hybrid (BM25 + Dense + Reranker), BM25-only, Dense-only, Graph-only
    * Graph Traversal: --enable-graph / --no-graph (CITES, EMPOWERS, NEXT_SECTION, PREV_SECTION)
    * Cross-Encoder Reranker: --enable-reranker / --no-reranker
    * Candidate Pool: --dense-top-k, --bm25-top-k, --rerank-pool-size
    * Configurable Cutoffs: --k (e.g., 5, 10)
"""

import os
import sys
import argparse
import json
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv(".env", override=False)

from core.graph_construct.feature_graph import (
    GraphDBManager,
    get_embedding,
    _ensure_bm25_index
)
from core.graph_construct.hybrid_reranker import (
    get_bm25_index,
    get_reranker,
    weighted_rrf,
    expand_numeric_query
)
from evaluation.evaluate_rag_triad import (
    compute_retrieval_metrics,
    match_legal_section,
    match_document
)


def retrieve_candidates(
    query_text: str,
    db: Any,
    mode: str = "hybrid",
    enable_graph: bool = True,
    enable_reranker: bool = True,
    bm25_top_k: int = 30,
    dense_top_k: int = 30,
    rerank_pool_size: int = 40,
    rerank_threshold: float = 0.20,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_device: str = "cuda:0",
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Executes single query retrieval with customizable pipeline switches."""
    _ensure_bm25_index(db)
    
    sparse_results = []
    dense_results = []

    # 1. Sparse BM25 Search
    if mode in ("hybrid", "bm25", "graph"):
        bm25_idx = get_bm25_index()
        bm25_query = expand_numeric_query(query_text)
        sparse_raw = bm25_idx.search(bm25_query, top_k=bm25_top_k)
        for doc, score in sparse_raw:
            sparse_results.append(({
                'id': doc['id'],
                'type': doc['type'],
                'description': doc.get('text', ''),
                'data': doc.get('data', {})
            }, score))

    # 2. Dense Vector Search (BGE-M3)
    if mode in ("hybrid", "dense", "graph"):
        query_embedding = get_embedding(query_text)
        if query_embedding is not None:
            law_records = db.find_similar_nodes(query_embedding, 'Laws', top_k=dense_top_k)
            for rec in law_records:
                dense_results.append(({
                    'id': rec['id'],
                    'type': 'Laws',
                    'description': rec.get('description', ''),
                    'data': rec
                }, rec.get('similarity', 0.0)))

            case_records = db.find_similar_nodes(query_embedding, 'Cases', top_k=dense_top_k)
            for rec in case_records:
                dense_results.append(({
                    'id': rec['id'],
                    'type': 'Cases',
                    'description': rec.get('description', ''),
                    'data': rec
                }, rec.get('similarity', 0.0)))

    # 3. Fusion / Candidate Ranking
    if mode == "bm25":
        fused_candidates = [doc for doc, _ in sparse_results]
    elif mode == "dense":
        dense_results.sort(key=lambda x: x[1], reverse=True)
        fused_candidates = [doc for doc, _ in dense_results]
    else:  # hybrid or graph
        fused_candidates = weighted_rrf(
            dense_results,
            sparse_results,
            dense_weight=1.0,
            sparse_weight=1.0,
            rrf_k=60
        )

    # 4. Cross-Encoder Reranking
    if enable_reranker and mode != "bm25":
        reranker = get_reranker(model_name=rerank_model, device=rerank_device, threshold=rerank_threshold)
        if reranker and reranker.model is not None:
            rerank_pool = fused_candidates[:rerank_pool_size]
            top_candidates = reranker.rerank(
                query_text,
                rerank_pool,
                top_k=max(top_k * 2, 20),
                threshold=rerank_threshold
            )
        else:
            top_candidates = fused_candidates[:max(top_k * 2, 20)]
    else:
        top_candidates = fused_candidates[:max(top_k * 2, 20)]

    # 5. Extract Candidate Laws & Optional Knowledge Graph Traversal
    laws = []
    seen_law_ids = set()
    legal_relations = ['CITES', 'EMPOWERS', 'CITED_BY', 'EMPOWERED_BY', 'PREV_SECTION', 'NEXT_SECTION', 'RELATED_TO']

    for cand in top_candidates:
        node_id = cand.get('id')
        node_type = cand.get('type')
        data = cand.get('data', {})

        if node_type == 'Laws':
            if node_id not in seen_law_ids and len(laws) < top_k:
                laws.append({
                    'id': node_id,
                    'law_entry': data.get('entry', ''),
                    'entry': data.get('entry', ''),
                    'description': data.get('description', ''),
                    'text': data.get('description', ''),
                    'snippet': data.get('description', '')[:250],
                    'rerank_score': cand.get('rerank_score', 1.0)
                })
                seen_law_ids.add(node_id)

            # Knowledge Graph Multi-Relationship Expansion
            if enable_graph and len(laws) < top_k + 4:
                connected_ids = []
                for rel in legal_relations:
                    connected_ids.extend(db.get_neighbors(node_id, rel))
                for n_id in connected_ids[:2]:
                    if n_id not in seen_law_ids and db.nodes_data.get(n_id, {}).get('type') == 'Laws':
                        n_data = db.get_node(n_id)
                        if n_data:
                            laws.append({
                                'id': n_id,
                                'law_entry': n_data.get('entry', ''),
                                'entry': n_data.get('entry', ''),
                                'description': n_data.get('description', ''),
                                'text': n_data.get('description', ''),
                                'snippet': n_data.get('description', '')[:250],
                                'rerank_score': cand.get('rerank_score', 0.5) * 0.9
                            })
                            seen_law_ids.add(n_id)

        elif node_type == 'Cases' and enable_graph:
            # Case to Laws via RELATES_TO_LAW
            law_neighbors = db.get_neighbors(node_id, 'RELATES_TO_LAW')
            for law_id in law_neighbors:
                if law_id not in seen_law_ids and len(laws) < top_k + 4:
                    law_data = db.get_node(law_id)
                    if law_data:
                        laws.append({
                            'id': law_id,
                            'law_entry': law_data.get('entry', ''),
                            'entry': law_data.get('entry', ''),
                            'description': law_data.get('description', ''),
                            'text': law_data.get('description', ''),
                            'snippet': law_data.get('description', '')[:250],
                            'rerank_score': cand.get('rerank_score', 0.5) * 0.85
                        })
                        seen_law_ids.add(law_id)

    return laws


def run_retriever_evaluation(
    dataset_file: str,
    graph_db_path: str,
    mode: str = "hybrid",
    enable_graph: bool = True,
    enable_reranker: bool = True,
    k_cutoffs: List[int] = [5, 10],
    sample_limit: Optional[int] = None,
    output_file: Optional[str] = None,
    dense_top_k: int = 30,
    bm25_top_k: int = 30,
    rerank_pool_size: int = 40,
    rerank_threshold: float = 0.20
):
    print("=" * 65)
    print("🚀 LEGAL RETRIEVER BENCHMARK & ABLATION EVALUATION")
    print("=" * 65)
    print(f"[*] Mode:            {mode.upper()}")
    print(f"[*] Graph Traversal: {'ENABLED' if enable_graph else 'DISABLED'}")
    print(f"[*] Reranker:        {'ENABLED (Cross-Encoder)' if enable_reranker else 'DISABLED'}")
    print(f"[*] K Cutoffs:       {k_cutoffs}")
    print(f"[*] Dataset:         {dataset_file}")

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if sample_limit and sample_limit > 0:
        test_cases = test_cases[:sample_limit]
        print(f"[*] Limited to first {sample_limit} test cases")

    # Load Graph
    if not os.path.exists(graph_db_path):
        raise FileNotFoundError(f"Graph database not found: {graph_db_path}")
    print(f"[*] Loading Graph DB from {graph_db_path}...")
    GraphDBManager.load(graph_db_path)
    db = GraphDBManager.get_db()
    print(f"[+] Loaded Graph with {len(db.nodes_data)} nodes.")

    max_k = max(k_cutoffs)
    results_by_case = []
    latencies = []

    metrics_by_k = {
        k: {
            "strict_hits": 0,
            "doc_hits": 0,
            "sec_hits": 0,
            "recalls": [],
            "precisions": [],
            "adj_precisions": [],
            "mrrs": []
        }
        for k in k_cutoffs
    }

    print("\n[*] Evaluating retrieval across test cases...")
    for idx, case in enumerate(tqdm(test_cases, desc="Evaluating")):
        cid = case.get("id", idx)
        question = case.get("fact", "")
        true_section = case.get("laws", [])
        ground_truth = case.get("ground_truth", "")

        expected_docs = case.get("source_files", [])
        if not expected_docs and case.get("source_file"):
            expected_docs = [s.strip() for s in case["source_file"].split(";") if s.strip()]

        expected_pairs = case.get("expected_pairs", [])
        if not expected_pairs:
            if len(expected_docs) == len(true_section) and len(expected_docs) > 0:
                expected_pairs = [{"doc": d, "section": s} for d, s in zip(expected_docs, true_section)]
            else:
                expected_pairs = [{"doc": d, "section": s} for d in expected_docs for s in true_section]

        t0 = time.time()
        retrieved_laws = retrieve_candidates(
            query_text=question,
            db=db,
            mode=mode,
            enable_graph=enable_graph,
            enable_reranker=enable_reranker,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            rerank_pool_size=rerank_pool_size,
            rerank_threshold=rerank_threshold,
            top_k=max_k
        )
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        case_k_metrics = {}
        for k in k_cutoffs:
            sub_chunks = retrieved_laws[:k]
            ret_metrics = compute_retrieval_metrics(
                retrieved_items=sub_chunks,
                expected_pairs=expected_pairs,
                ground_truth_sections=true_section,
                k=k
            )

            is_both_hit = (ret_metrics.get("hit_at_k", 0.0) > 0)
            if is_both_hit:
                metrics_by_k[k]["strict_hits"] += 1

            # Document Hit
            is_doc_hit = False
            for ed in expected_docs:
                for item in sub_chunks:
                    chunk_text = f"{item.get('law_entry', '')} {item.get('text', '')}".strip()
                    if match_document(ed, chunk_text):
                        is_doc_hit = True
                        break
                if is_doc_hit:
                    break
            if is_doc_hit:
                metrics_by_k[k]["doc_hits"] += 1

            # Section Hit
            is_sec_hit = False
            for ts in true_section:
                ts_clean = str(ts).strip()
                if not ts_clean:
                    continue
                for item in sub_chunks:
                    chunk_text = f"{item.get('law_entry', '')} {item.get('text', '')}".strip()
                    if match_legal_section(ts_clean, chunk_text):
                        is_sec_hit = True
                        break
                if is_sec_hit:
                    break
            if is_sec_hit:
                metrics_by_k[k]["sec_hits"] += 1

            metrics_by_k[k]["recalls"].append(ret_metrics.get("recall_at_k", 0.0))
            metrics_by_k[k]["precisions"].append(ret_metrics.get("precision_at_k", 0.0))
            metrics_by_k[k]["adj_precisions"].append(ret_metrics.get("adjusted_precision_at_k", 0.0))
            metrics_by_k[k]["mrrs"].append(ret_metrics.get("mrr_at_k", 0.0))

            case_k_metrics[f"k_{k}"] = {
                "strict_hit": is_both_hit,
                "doc_hit": is_doc_hit,
                "sec_hit": is_sec_hit,
                **ret_metrics
            }

        results_by_case.append({
            "case_id": cid,
            "question": question[:120],
            "expected_pairs": expected_pairs,
            "latency_ms": round(latency_ms, 2),
            "retrieved_count": len(retrieved_laws),
            "top_candidates": [
                {
                    "rank": r + 1,
                    "entry": l.get("entry", ""),
                    "score": round(float(l.get("rerank_score", 1.0)), 4),
                    "snippet": l.get("snippet", "")
                }
                for r, l in enumerate(retrieved_laws[:max_k])
            ],
            "metrics": case_k_metrics
        })

    total_evaluated = len(test_cases)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    summary_by_k = {}
    for k in k_cutoffs:
        d = metrics_by_k[k]
        strict_rate = (d["strict_hits"] / total_evaluated * 100) if total_evaluated > 0 else 0.0
        doc_rate = (d["doc_hits"] / total_evaluated * 100) if total_evaluated > 0 else 0.0
        sec_rate = (d["sec_hits"] / total_evaluated * 100) if total_evaluated > 0 else 0.0
        mean_recall = (sum(d["recalls"]) / total_evaluated * 100) if total_evaluated > 0 else 0.0
        mean_prec = (sum(d["precisions"]) / total_evaluated * 100) if total_evaluated > 0 else 0.0
        mean_adj_prec = (sum(d["adj_precisions"]) / total_evaluated * 100) if total_evaluated > 0 else 0.0
        mean_mrr = (sum(d["mrrs"]) / total_evaluated * 100) if total_evaluated > 0 else 0.0

        summary_by_k[f"k_{k}"] = {
            "strict_hit_rate": round(strict_rate, 2),
            "doc_hit_rate": round(doc_rate, 2),
            "section_hit_rate": round(sec_rate, 2),
            "mean_recall": round(mean_recall, 2),
            "mean_precision": round(mean_prec, 2),
            "mean_adjusted_precision": round(mean_adj_prec, 2),
            "mean_mrr": round(mean_mrr, 2),
        }

    final_report = {
        "benchmark_config": {
            "mode": mode,
            "enable_graph": enable_graph,
            "enable_reranker": enable_reranker,
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "rerank_pool_size": rerank_pool_size,
            "rerank_threshold": rerank_threshold,
            "total_evaluated": total_evaluated,
            "avg_latency_ms": round(avg_latency, 2)
        },
        "summary": summary_by_k,
        "cases": results_by_case
    }

    # Save Output Report
    if not output_file:
        out_dir = os.path.join(project_root, "outputs", "THAI")
        os.makedirs(out_dir, exist_ok=True)
        graph_tag = "with_graph" if enable_graph else "no_graph"
        rerank_tag = "rerank" if enable_reranker else "no_rerank"
        output_file = os.path.join(out_dir, f"retriever_eval_{mode}_{graph_tag}_{rerank_tag}.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Detailed evaluation report saved to: {output_file}")

    # Console Summary Table
    print("\n" + "=" * 65)
    print("📊 LEGAL RETRIEVER BENCHMARK RESULTS SUMMARY")
    print("=" * 65)
    print(f"Mode: {mode.upper()} | Graph: {enable_graph} | Reranker: {enable_reranker}")
    print(f"Evaluated Cases: {total_evaluated} | Avg Latency: {avg_latency:.1f} ms/query")
    print("-" * 65)
    print(f"{'Metric':<30} | " + " | ".join([f"K={k:<5}" for k in k_cutoffs]))
    print("-" * 65)
    print(f"{'🎯 Strict Hit Rate (Doc & Sec)':<30} | " + " | ".join([f"{summary_by_k[f'k_{k}']['strict_hit_rate']:>6.2f}%" for k in k_cutoffs]))
    print(f"{'📚 Ground Truth Recall':<30} | " + " | ".join([f"{summary_by_k[f'k_{k}']['mean_recall']:>6.2f}%" for k in k_cutoffs]))
    print(f"{'⚡ Mean Reciprocal Rank (MRR)':<30} | " + " | ".join([f"{summary_by_k[f'k_{k}']['mean_mrr']:>6.2f}%" for k in k_cutoffs]))
    print(f"{'🎯 Adjusted Precision':<30} | " + " | ".join([f"{summary_by_k[f'k_{k}']['mean_adjusted_precision']:>6.2f}%" for k in k_cutoffs]))
    print(f"{'📄 Document Hit Rate':<30} | " + " | ".join([f"{summary_by_k[f'k_{k}']['doc_hit_rate']:>6.2f}%" for k in k_cutoffs]))
    print(f"{'⚖️ Section Hit Rate':<30} | " + " | ".join([f"{summary_by_k[f'k_{k}']['section_hit_rate']:>6.2f}%" for k in k_cutoffs]))
    print("=" * 65 + "\n")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Legal Retriever Benchmark with Ablation Switches")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "bm25", "dense", "graph"],
        default="hybrid",
        help="Retrieval search mode (default: hybrid)"
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        default=True,
        help="Enable Knowledge Graph multi-relationship expansion (default: True)"
    )
    parser.add_argument(
        "--no-graph",
        action="store_false",
        dest="enable_graph",
        help="Disable Knowledge Graph expansion"
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        default=True,
        help="Enable Cross-Encoder Reranker (default: True)"
    )
    parser.add_argument(
        "--no-reranker",
        action="store_false",
        dest="enable_reranker",
        help="Disable Cross-Encoder Reranker"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="K cutoff ranks to evaluate (default: 5 10)"
    )
    parser.add_argument(
        "--datasets",
        default="./datasets/crime_data_THAI_small.json",
        help="Path to ground truth dataset JSON"
    )
    parser.add_argument(
        "--graph-db",
        default="./outputs/graph_db.pkl",
        help="Path to prebuilt Graph DB pkl"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Sample limit for quick ablation smoke testing"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save evaluation report JSON"
    )
    parser.add_argument(
        "--dense-top-k",
        type=int,
        default=30,
        help="Dense candidate pool (default: 30)"
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=30,
        help="Sparse BM25 candidate pool (default: 30)"
    )
    parser.add_argument(
        "--rerank-pool-size",
        type=int,
        default=40,
        help="Candidate pool size passed to reranker (default: 40)"
    )
    parser.add_argument(
        "--rerank-threshold",
        type=float,
        default=0.20,
        help="Reranker relevance threshold score (default: 0.20)"
    )

    args = parser.parse_args()

    run_retriever_evaluation(
        dataset_file=args.datasets,
        graph_db_path=args.graph_db,
        mode=args.mode,
        enable_graph=args.enable_graph,
        enable_reranker=args.enable_reranker,
        k_cutoffs=args.k,
        sample_limit=args.limit,
        output_file=args.output,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_pool_size=args.rerank_pool_size,
        rerank_threshold=args.rerank_threshold
    )
