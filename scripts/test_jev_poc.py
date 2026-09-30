# -*- coding: utf-8 -*-
"""
scripts/test_jev_poc.py
Standalone PoC: Test Jev 1.13 as a Corrective RAG Retrieval Evaluator / Filter
Takes the top retrieved chunks from outputs/THAI/openrouter_results.json and measures:
- Filtered Precision vs Raw Precision
- Recall retention (did we accidentally drop GT sections?)
- Strict Hit Rate retention
"""

import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from evaluation.evaluate_rag_triad import match_doc_and_section

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment!")

RESULTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "THAI", "openrouter_results.json")
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results_data = json.load(f)


def evaluate_chunk_with_jev(question: str, chunk_entry: str, chunk_snippet: str) -> float:
    """Calls typesafe/jev-1.13 to get relevance probability (0.0 to 1.0)."""
    state_text = f"Inquiry: {question}\n\nCandidate Statutory Text: {chunk_entry} {chunk_snippet}"
    payload = {
        "model": "typesafe/jev-1.13",
        "state": state_text[:2500],
        "questions": {
            "is_directly_relevant": {
                "type": "noul",
                "instructions": (
                    "Does this statutory text contain the specific rules, conditions, "
                    "or authority directly applicable to the scenario in the inquiry?"
                )
            }
        }
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/alpha/decisions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10
        )
        if resp.status_code == 200:
            res_json = resp.json()
            noul_val = res_json.get("answers", {}).get("is_directly_relevant", {}).get("noul", 0.0)
            return float(noul_val)
        else:
            print(f"[Warning] Jev API error ({resp.status_code}): {resp.text[:100]}")
            return 0.5
    except Exception as e:
        print(f"[Error] Jev call failed: {e}")
        return 0.5


def run_poc(sample_limit: int = 5, threshold: float = 0.6):
    print("=" * 65)
    print(f"🔬 POC: Jev 1.13 as Corrective RAG Retrieval Filter (Top-{sample_limit} Samples)")
    print(f"Filter Threshold: {threshold}")
    print("=" * 65)

    cases_to_test = results_data[:sample_limit]

    raw_precisions = []
    filtered_precisions = []
    raw_hits = []
    filtered_hits = []

    for idx, case in enumerate(cases_to_test):
        question = case.get("question", "")
        expected_pairs = case.get("expected_pairs", [])
        evidence = case.get("analysis", {}).get("top_retrieved_evidence", [])[:5]

        print(f"\n[Case {idx+1}/{len(cases_to_test)}] Q: {question[:75]}...")
        gt_summary = [f"{p.get('doc','').split('/')[-1]}: {p.get('section','')}" for p in expected_pairs]
        print(f"  🎯 Ground Truth ({len(expected_pairs)}): {gt_summary}")

        raw_relevant_count = 0
        filtered_chunks = []
        filtered_relevant_count = 0

        for r_idx, chunk in enumerate(evidence, 1):
            entry = chunk.get("law_entry", "")
            snippet = chunk.get("snippet", "")
            chunk_full = f"{entry} {snippet}"

            # Ground Truth Check
            is_gt_match = any(
                match_doc_and_section(p.get("doc", ""), p.get("section", ""), chunk_full)
                for p in expected_pairs
            )
            if is_gt_match:
                raw_relevant_count += 1

            # Jev Evaluator Call
            t0 = time.time()
            jev_score = evaluate_chunk_with_jev(question, entry, snippet)
            dt = time.time() - t0

            passed = jev_score >= threshold
            status_icon = "✅ KEPT" if passed else "❌ FILTERED"
            gt_icon = "⭐ (GT Relevant)" if is_gt_match else "⚪ (Irrelevant)"

            print(f"    Rank {r_idx}: {status_icon} [Score: {jev_score:.2f} in {dt:.2f}s] {gt_icon} | {entry[:60]}")

            if passed:
                filtered_chunks.append(chunk)
                if is_gt_match:
                    filtered_relevant_count += 1

        # Calculate metrics for this case
        raw_p = raw_relevant_count / 5.0
        filt_p = (filtered_relevant_count / len(filtered_chunks)) if filtered_chunks else 0.0

        raw_hit = 1.0 if raw_relevant_count > 0 else 0.0
        filt_hit = 1.0 if filtered_relevant_count > 0 else 0.0

        raw_precisions.append(raw_p)
        filtered_precisions.append(filt_p)
        raw_hits.append(raw_hit)
        filtered_hits.append(filt_hit)

        print(f"  👉 Chunks: {len(evidence)} -> {len(filtered_chunks)} survived | Precision: {raw_p*100:.1f}% -> {filt_p*100:.1f}%")

    print("\n" + "=" * 65)
    print("📊 POC COMPARISON SUMMARY:")
    print("=" * 65)
    print(f"1. Mean Precision:")
    print(f"   - Raw Retrieval:       {sum(raw_precisions)/len(raw_precisions)*100:.2f}%")
    print(f"   - After Jev Filtering: {sum(filtered_precisions)/len(filtered_precisions)*100:.2f}%")
    print(f"2. Strict Hit Rate (Coverage of GT):")
    print(f"   - Raw Retrieval:       {sum(raw_hits)/len(raw_hits)*100:.2f}%")
    print(f"   - After Jev Filtering: {sum(filtered_hits)/len(filtered_hits)*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_poc(sample_limit=5, threshold=0.60)
