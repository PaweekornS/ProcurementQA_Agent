# -*- coding: utf-8 -*-
"""
evaluation/retrieval_benchmark.py

Retrieval-only benchmark of the code paths the agent actually uses (no LLM calls), on both
test sets:
  - statute questions  (expected document + มาตรา/ข้อ pairs, evaluate_rag_triad.match_chunk)
  - general documents  (evidence spans, evaluation/evidence.py)

Systems
  new         core.retrieval.retriever.Retriever (fused recall, one rerank, carried graph neighbours)
  new_nograph Retriever with graph_per_seed=0
  new_norerank Retriever without the cross-encoder (fused order)

Every system gets the single question as its only query, so the comparison isolates retrieval;
multi-query behaviour (sub-questions) is measured end-to-end by run.py + evaluate_rag_triad.py.

Usage:
    python evaluation/retrieval_benchmark.py                               # local, in-memory store
    USE_TRI_STORE=true python evaluation/retrieval_benchmark.py            # tri-store (Docker up)
"""

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

from evaluation.evaluate_rag_triad import compute_retrieval_metrics  # noqa: E402
from evaluation.evidence import evidence_metrics  # noqa: E402

KS = (5, 10, 15, 20)
DATASETS = {
    "statute": "datasets/crime_data_THAI_small.json",
    "general": "datasets/general_docs_qa.json",
}


def build_systems(names: List[str], org_id: str) -> Dict[str, Callable[[str], List[dict]]]:
    from core.retrieval.retriever import get_retriever

    systems = {}
    base = get_retriever()

    def variant(**overrides):
        r = copy.copy(base)
        r.config = dataclasses.replace(base.config, **overrides)
        return r

    for name in names:
        if name == "new":
            systems[name] = lambda q, r=base: r.retrieve([q], org_id).laws()
        elif name == "new_nograph":
            r = variant(graph_per_seed=0)
            systems[name] = lambda q, r=r: r.retrieve([q], org_id).laws()
        elif name == "new_norerank":
            r = copy.copy(base)
            r.scorer = None
            systems[name] = lambda q, r=r: r.retrieve([q], org_id).laws()
    return systems


def score_question(q: dict, laws: List[dict]) -> Dict[str, dict]:
    out = {}
    if q.get("evidence"):
        ranked = [{"doc": l.get("entry", "").split("|")[0].strip(), "text": l.get("description", "")} for l in laws]
        for k in KS:
            m = evidence_metrics(q["evidence"], ranked, k)
            out[k] = {"recall": m["recall"], "hit": m["hit"], "mrr": m["mrr"]}
    else:
        items = [{"law_entry": l.get("entry", ""), "snippet": l.get("description", "")[:300]} for l in laws]
        for k in KS:
            m = compute_retrieval_metrics(items, q.get("expected_pairs", []), q.get("laws"), k=k)
            out[k] = {"recall": m["recall_at_k"], "hit": m["hit_at_k"], "mrr": m["mrr_at_k"]}
    return out


def main():
    ap = argparse.ArgumentParser(description="Retrieval-only benchmark (no LLM)")
    ap.add_argument("--systems", default="new,new_nograph,new_norerank")
    ap.add_argument("--datasets", default="statute,general")
    ap.add_argument("--org-id", default=os.getenv("DEFAULT_ORG_ID", "DGA"))
    ap.add_argument("--out", default="outputs/retrieval_benchmark.json")
    args = ap.parse_args()

    systems = build_systems(args.systems.split(","), args.org_id)
    report = {"store": "tri" if os.getenv("USE_TRI_STORE", "false").lower() in ("true", "1", "yes") else "memory",
              "results": {}}
    for ds in args.datasets.split(","):
        questions = json.load(open(DATASETS[ds], encoding="utf-8"))
        for name, fn in systems.items():
            rows, latencies, sizes = [], [], []
            for q in questions:
                t0 = time.time()
                laws = fn(q["fact"])
                latencies.append(time.time() - t0)
                sizes.append(len(laws))
                rows.append({"id": q.get("id"), "type": q.get("question_type", "statute"), **score_question(q, laws),
                             "top3": [l.get("entry", "")[:90] for l in laws[:3]]})
            summary = {f"{m}@{k}": round(float(np.mean([r[k][m] for r in rows])), 4) for k in KS for m in ("recall", "hit", "mrr")}
            summary.update(n=len(rows), latency_s=round(float(np.mean(latencies)), 2), context=round(float(np.mean(sizes)), 1))
            report["results"].setdefault(ds, {})[name] = {"summary": summary, "per_question": rows}
            print(f"{ds:8} {name:13} recall@15={summary['recall@15']:.3f} hit@15={summary['hit@15']:.3f} "
                  f"mrr@15={summary['mrr@15']:.3f} | recall@5={summary['recall@5']:.3f} ctx={summary['context']:.0f} "
                  f"lat={summary['latency_s']:.2f}s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
