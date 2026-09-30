#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_section_poc.py

Head-to-head retrieval comparison:
- Baseline: Page-Level Chunks (datas/law_to_crime.json)
- Candidate: Section-Level Chunks (datas/law_to_crime_section_level.json)

Evaluates on the 40 test cases from datasets/crime_data_THAI_small.json.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from core.graph_construct.hybrid_reranker import ThaiBM25Index, expand_numeric_query
from evaluation.evaluate_rag_triad import match_doc_and_section, compute_retrieval_metrics


def build_index_for_corpus(corpus_path: str) -> Tuple[ThaiBM25Index, List[Dict[str, Any]]]:
    print(f"[*] Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    docs_for_bm25 = []
    for item in corpus:
        cid = item.get("id", "")
        text = item["items"][0]["text"] if item.get("items") else ""
        docs_for_bm25.append({
            "id": cid,
            "type": "Laws",
            "text": f"{cid}\n{text}",
            "data": {
                "entry": cid,
                "text": text
            }
        })
        
    bm25 = ThaiBM25Index()
    bm25.build_index(docs_for_bm25)
    print(f"[+] BM25 index built with {len(docs_for_bm25)} chunks.")
    return bm25, docs_for_bm25


def run_retrieval_benchmark():
    test_file = "datasets/crime_data_THAI_small.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"[+] Loaded {len(test_cases)} test cases from {test_file}")

    # 1. Build indices
    bm25_page, page_docs = build_index_for_corpus("datas/law_to_crime.json")
    bm25_sec, sec_docs = build_index_for_corpus("datas/law_to_crime_section_level.json")

    page_metrics_list = []
    sec_metrics_list = []

    print(f"\n[*] Evaluating retrieval across {len(test_cases)} test cases...")

    for idx, case in enumerate(test_cases):
        q = case.get("fact") or case.get("question") or case.get("description", "")
        expected_pairs = case.get("expected_pairs", [])
        gt_sections = case.get("laws", case.get("law", []))

        bm25_q = expand_numeric_query(q)

        # Baseline: Page-Level Top-5
        top_page_raw = bm25_page.search(bm25_q, top_k=5)
        top_page_items = [
            {"entry": doc["id"], "text": doc.get("data", {}).get("text", "")}
            for doc, score in top_page_raw
        ]
        m_page = compute_retrieval_metrics(
            retrieved_items=top_page_items,
            expected_pairs=expected_pairs,
            ground_truth_sections=gt_sections,
            k=5
        )
        page_metrics_list.append(m_page)

        # Candidate: Section-Level Top-5
        top_sec_raw = bm25_sec.search(bm25_q, top_k=5)
        top_sec_items = [
            {"entry": doc["id"], "text": doc.get("data", {}).get("text", "")}
            for doc, score in top_sec_raw
        ]
        m_sec = compute_retrieval_metrics(
            retrieved_items=top_sec_items,
            expected_pairs=expected_pairs,
            ground_truth_sections=gt_sections,
            k=5
        )
        sec_metrics_list.append(m_sec)

    # Aggregate
    def avg(lst, key):
        return sum(x[key] for x in lst) / max(len(lst), 1)

    print("\n" + "=" * 65)
    print("🏆 POC RETRIEVAL COMPARISON: PAGE-LEVEL VS SECTION-LEVEL (K=5)")
    print("=" * 65)
    print(f"{'Metric':<28} | {'Page-Level (Baseline)':<20} | {'Section-Level (PoC)':<20}")
    print("-" * 65)
    
    metrics = [
        ("Strict Hit@5", "hit_at_k"),
        ("MRR@5", "mrr_at_k"),
        ("Adjusted Precision@5", "adjusted_precision_at_k"),
        ("Fixed Precision@5 (Raw)", "precision_at_k"),
        ("Ground Truth Recall@5", "recall_at_k")
    ]
    
    for label, key in metrics:
        val_page = avg(page_metrics_list, key) * 100
        val_sec = avg(sec_metrics_list, key) * 100
        diff = val_sec - val_page
        diff_str = f"(+{diff:.1f}%)" if diff >= 0 else f"({diff:.1f}%)"
        print(f"{label:<28} | {val_page:>18.2f}% | {val_sec:>12.2f}% {diff_str:>7}")

    print("=" * 65)

    # Multi-K evaluation for Section-Level
    print("\n📈 Section-Level Hit Rate across K (Scaling Law):")
    for test_k in [5, 7, 10]:
        k_hits = []
        for case in test_cases:
            q = case.get("fact") or case.get("question") or ""
            bm25_q = expand_numeric_query(q)
            res = bm25_sec.search(bm25_q, top_k=test_k)
            items = [{"entry": doc["id"], "text": doc.get("data", {}).get("text", "")} for doc, score in res]
            m = compute_retrieval_metrics(
                retrieved_items=items,
                expected_pairs=case.get("expected_pairs", []),
                ground_truth_sections=case.get("laws", case.get("law", [])),
                k=test_k
            )
            k_hits.append(m["hit_at_k"])
        avg_hit = sum(k_hits) / len(k_hits) * 100
        print(f"  • Section-Level Strict Hit@{test_k}: {avg_hit:.2f}% (Context Footprint: ~{test_k * 2280:,} chars vs Page-Level Top-5: ~{5 * 6200:,} chars)")


if __name__ == "__main__":
    run_retrieval_benchmark()
