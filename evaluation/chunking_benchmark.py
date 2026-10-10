# -*- coding: utf-8 -*-
"""
evaluation/chunking_benchmark.py

Offline comparison of chunking strategies on the same corpus, retriever and test set.
Needs no database or LLM; dense retrieval is added when TOKENMIND_API_KEY is set.

Strategies
  page_paragraph  previous tenant-document chunker: ~1,200-char paragraph packing that never
                  crosses a page and hard-cuts long paragraphs at a character count
  fixed_overlap   naive baseline: 1,000-char windows with 200-char overlap over the whole text
  structure       core/chunking (statute units / heading sections / standalone tables / FAQ pairs)
                  indexed with its '[document | section]' context header
  structure_nohdr the same chunks indexed without the context header (ablation)

Every strategy indexes '[document title]' + text so the only difference is the chunking itself
(and, for 'structure', the section path in the header).

Metrics (evaluation/evidence.py): evidence recall@k, hit@k, all-evidence@k, MRR and precision@k,
per question type; plus 'evidence intact', the share of evidence spans that sit whole inside at
least one chunk (a property of the chunker alone, independent of the retriever).

Usage:
    python evaluation/chunking_benchmark.py
    python evaluation/chunking_benchmark.py --dense          # + BGE-M3 dense and hybrid RRF
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from core.chunking import chunk_document, doc_title_from_path  # noqa: E402
from core.retrieval.reranker import ThaiBM25Index  # noqa: E402
from evaluation.evidence import evidence_metrics, normalize, same_document  # noqa: E402

KS = (1, 3, 5, 10, 20)
_PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s+of\s+(\d+)\s*-->", re.IGNORECASE)


# ----------------------------------------------------------------------------- chunkers

def page_paragraph_chunks(markdown: str, max_chars: int = 1200, hard_limit: int = 1800) -> List[str]:
    """The chunker tenant uploads used before core/chunking (kept here as the baseline)."""
    markers = list(_PAGE_MARKER.finditer(markdown))
    pages = [markdown] if not markers else [
        markdown[m.end():(markers[i + 1].start() if i + 1 < len(markers) else len(markdown))]
        for i, m in enumerate(markers)
    ]
    chunks = []
    for page in pages:
        buf, pieces = "", []
        for p in [p.strip() for p in re.split(r"\n\s*\n", page) if p.strip()]:
            while len(p) > hard_limit:
                pieces.append(p[:max_chars])
                p = p[max_chars:]
            pieces.append(p)
        for p in pieces:
            if buf and len(buf) + len(p) + 2 > max_chars:
                chunks.append(buf)
                buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            chunks.append(buf)
    return chunks


def fixed_overlap_chunks(markdown: str, size: int = 1000, overlap: int = 200) -> List[str]:
    text = _PAGE_MARKER.sub("", markdown)
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def build_corpora(ocr_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    corpora = defaultdict(list)
    seen = set()
    for path in sorted(ocr_dir.rglob("*.md")):
        rel = path.relative_to(ocr_dir).as_posix()
        md = path.read_text(encoding="utf-8")
        digest = hashlib.md5(md.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        title = doc_title_from_path(rel)
        for t in page_paragraph_chunks(md):
            corpora["page_paragraph"].append({"doc": title, "text": t, "index": f"[{title}]\n{t}"})
        for t in fixed_overlap_chunks(md):
            corpora["fixed_overlap"].append({"doc": title, "text": t, "index": f"[{title}]\n{t}"})
        _, chunks = chunk_document(md, rel, doc_title=title)
        for c in chunks:
            corpora["structure"].append({"doc": title, "text": c.content, "index": c.embed_text, "kind": c.kind})
            corpora["structure_nohdr"].append({"doc": title, "text": c.content, "index": f"[{title}]\n{c.content}", "kind": c.kind})
    return corpora


# ----------------------------------------------------------------------------- retrievers

class BM25Retriever:
    def __init__(self, docs: List[Dict[str, str]]):
        self.bm25 = BM25Okapi([ThaiBM25Index.tokenize(d["index"]) for d in docs])

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(ThaiBM25Index.tokenize(query)))


class DenseRetriever:
    def __init__(self, docs: List[Dict[str, str]], cache_dir: Path, name: str):
        self.matrix = self._embed_cached([d["index"] for d in docs], cache_dir / f"dense_{name}.npy")

    @staticmethod
    def _embed(texts: List[str]) -> np.ndarray:
        from scripts.migrate_to_tri_store import batch_embed_texts
        vecs = batch_embed_texts(
            texts, api_key=os.getenv("TOKENMIND_API_KEY"),
            base_url=os.getenv("TOKENMIND_BASE_URL", "https://tokenmind.abdul.in.th/v1").rstrip("/"),
            model=os.getenv("TOKENMIND_EMBEDDING_MODEL", "BAAI/bge-m3"), batch_size=32,
        )
        m = np.asarray(vecs, dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def _embed_cached(self, texts: List[str], path: Path) -> np.ndarray:
        key = hashlib.md5("\x00".join(texts).encode("utf-8")).hexdigest()[:12]
        path = path.with_name(f"{path.stem}_{key}.npy")
        if path.exists():
            return np.load(path)
        m = self._embed(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, m)
        return m

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self._embed([query])[0]


def rrf(*score_lists: np.ndarray, k: int = 60, depth: int = 100) -> np.ndarray:
    fused = np.zeros_like(score_lists[0], dtype=np.float64)
    for s in score_lists:
        top = np.argsort(-s)[:depth]
        fused[top] += 1.0 / (k + np.arange(1, len(top) + 1))
    return fused


# ----------------------------------------------------------------------------- evaluation

def evidence_intact(corpus: List[Dict[str, str]], questions: List[dict]) -> float:
    by_doc = defaultdict(list)
    for c in corpus:
        by_doc[c["doc"]].append(normalize(c["text"]))
    spans = [ev for q in questions for ev in q["evidence"]]
    ok = 0
    for ev in spans:
        needle = normalize(ev["text"])
        docs = [d for d in by_doc if same_document(ev["doc"], d)]
        ok += any(needle in t for d in docs for t in by_doc[d])
    return ok / len(spans)


def evaluate(corpus, scorer, questions) -> Dict[str, dict]:
    per_q = []
    for q in questions:
        order = np.argsort(-scorer(q["question"]))[:max(KS)]
        ranked = [corpus[i] for i in order]
        row = {"id": q["id"], "type": q["question_type"]}
        for k in KS:
            row[k] = evidence_metrics(q["evidence"], ranked, k)
        row["top3"] = [f"{c['doc'][:40]} :: {c['text'][:60]!r}" for c in ranked[:3]]
        per_q.append(row)
    return per_q


def aggregate(per_q, types=None) -> Dict[str, float]:
    rows = [r for r in per_q if types is None or r["type"] in types]
    if not rows:
        return {}
    out = {"n": len(rows)}
    for k in KS:
        for m in ("recall", "hit", "all", "precision"):
            out[f"{m}@{k}"] = round(float(np.mean([r[k][m] for r in rows])), 4)
    out["mrr@20"] = round(float(np.mean([r[20]["mrr"] for r in rows])), 4)
    return out


GROUPS = {
    "all": None,
    "text": {"text"},
    "faq": {"faq"},
    "table": {"table", "table+text", "table+multi_doc"},
}


def main():
    load_dotenv(override=False)
    ap = argparse.ArgumentParser(description="Offline chunking benchmark (BM25, optional dense/hybrid)")
    ap.add_argument("--ocr-dir", default="data_ocr")
    ap.add_argument("--dataset", default="datasets/general_docs_qa.json")
    ap.add_argument("--out", default="outputs/general_docs")
    ap.add_argument("--dense", action="store_true", help="Add BGE-M3 dense and hybrid RRF (needs TOKENMIND_API_KEY)")
    ap.add_argument("--strategies", default="page_paragraph,fixed_overlap,structure_nohdr,structure")
    args = ap.parse_args()

    questions = json.load(open(args.dataset, encoding="utf-8"))
    t0 = time.time()
    corpora = build_corpora(Path(args.ocr_dir))
    print(f"Chunked corpus in {time.time() - t0:.1f}s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": args.dataset, "questions": len(questions), "strategies": {}}
    for name in args.strategies.split(","):
        corpus = corpora[name]
        lens = [len(c["text"]) for c in corpus]
        entry = {
            "chunks": len(corpus),
            "chars_mean": int(np.mean(lens)), "chars_p95": int(np.percentile(lens, 95)), "chars_max": int(max(lens)),
            "evidence_intact": round(evidence_intact(corpus, questions), 4),
            "retrievers": {},
        }
        t0 = time.time()
        bm25 = BM25Retriever(corpus)
        retrievers = {"bm25": bm25.scores}
        if args.dense:
            dense = DenseRetriever(corpus, out_dir / "cache", name)
            retrievers["dense"] = dense.scores
            retrievers["hybrid_rrf"] = lambda q, b=bm25, d=dense: rrf(b.scores(q), d.scores(q))
        for rname, scorer in retrievers.items():
            per_q = evaluate(corpus, scorer, questions)
            entry["retrievers"][rname] = {
                "summary": {g: aggregate(per_q, t) for g, t in GROUPS.items()},
                "per_question": per_q,
            }
        report["strategies"][name] = entry
        s = entry["retrievers"]["bm25"]["summary"]["all"]
        print(f"{name:16} chunks={entry['chunks']:6} intact={entry['evidence_intact']:.2f} "
              f"bm25 recall@5={s['recall@5']:.3f} hit@5={s['hit@5']:.3f} mrr={s['mrr@20']:.3f} ({time.time() - t0:.0f}s)")

    (out_dir / "chunking_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "chunking_benchmark.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"\nReport: {out_dir / 'chunking_benchmark.md'}")


def render_markdown(report) -> str:
    lines = [f"# Chunking benchmark ({report['questions']} questions, {report['dataset']})", ""]
    lines += ["| strategy | chunks | mean chars | p95 chars | evidence intact |", "|---|---|---|---|---|"]
    for name, e in report["strategies"].items():
        lines.append(f"| {name} | {e['chunks']} | {e['chars_mean']} | {e['chars_p95']} | {e['evidence_intact']:.1%} |")
    retrievers = next(iter(report["strategies"].values()))["retrievers"].keys()
    for r in retrievers:
        for g in GROUPS:
            lines += ["", f"## {r} - {g}", "",
                      "| strategy | n | hit@1 | hit@5 | recall@5 | recall@10 | all@10 | recall@20 | MRR |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for name, e in report["strategies"].items():
                s = e["retrievers"][r]["summary"][g]
                if s:
                    lines.append(f"| {name} | {s['n']} | {s['hit@1']:.3f} | {s['hit@5']:.3f} | {s['recall@5']:.3f} | "
                                 f"{s['recall@10']:.3f} | {s['all@10']:.3f} | {s['recall@20']:.3f} | {s['mrr@20']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
