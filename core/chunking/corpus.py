# -*- coding: utf-8 -*-
"""
Builds the chunked corpus from the OCR markdown directory (data_ocr/), in the format the tri-store
migration and the local pipeline read:

    <out>/law_to_crime.json        every chunk (statute units, sections, tables, FAQ pairs)
    <out>/cases_with_feature.json  FAQ pairs again, as precedent cases for the Neo4j graph
    <out>/chunks.jsonl             full Chunk records (kind, section_path, parent_id, metadata)
    <out>/manifest.json            per-document type and chunk counts

The corpus is derived data (it may carry confidential content) and is never committed:
ensure_corpus() rebuilds it whenever it is missing or older than the OCR files.
"""

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from . import FAQ_KIND, chunk_document, doc_title_from_path

DEFAULT_OCR_DIR = "data_ocr"
DEFAULT_CORPUS_DIR = os.path.join("outputs", "corpus")

_REF = re.compile(r"(มาตรา|ข้อ)\s*([๐-๙\d]+)")


def to_law_record(chunk) -> dict:
    text = chunk.embed_text
    return {
        "id": chunk.entry,
        "doc_name": chunk.doc_title,
        "section": chunk.label,
        "chapter": " > ".join(chunk.section_path),
        "char_length": len(text),
        "items": [{
            "text": text,
            "crime": [p for p in chunk.section_path if p] or [chunk.doc_title],
            "judge_dep": [chunk.doc_title],
            "related_laws": [chunk.doc_title],
        }],
        "source_file": chunk.source_file,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "total_pages": chunk.total_pages,
        "doc_type": chunk.doc_type,
        "chunk_kind": chunk.kind,
        "parent_id": chunk.parent_id,
    }


def to_case_record(idx: int, chunk) -> dict:
    q, a = chunk.metadata.get("question", ""), chunk.metadata.get("answer", "")
    refs = list(dict.fromkeys(f"{k} {n}" for k, n in _REF.findall(a)))
    return {
        "id": idx,
        "name": [chunk.metadata.get("category") or "FAQ"],
        "fact": f"ข้อหารือ/คำถาม: {q}\nแนวทางวินิจฉัย/คำตอบ: {a}",
        "crime": [chunk.metadata.get("category") or "FAQ"],
        "law": refs,
        "source_file": chunk.source_file,
        "page_start": chunk.page_start,
    }


def build(ocr_dir: Path, out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    laws, cases, manifest = [], [], []
    kinds = Counter()
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as fout:
        seen_content, seen_titles = {}, Counter()
        for path in sorted(ocr_dir.rglob("*.md")):
            rel = path.relative_to(ocr_dir).as_posix()
            markdown = path.read_text(encoding="utf-8")
            digest = hashlib.md5(markdown.encode("utf-8")).hexdigest()
            if digest in seen_content:
                # The same file filed under two folders would only duplicate every chunk
                if verbose:
                    print(f"{'skip':8} {'':5}  {rel} (identical to {seen_content[digest]})")
                manifest.append({"source_file": rel, "duplicate_of": seen_content[digest]})
                continue
            seen_content[digest] = rel
            title = doc_title_from_path(rel)
            seen_titles[title] += 1
            if seen_titles[title] > 1:
                # Different documents whose file names differ only by whitespace
                title = f"{title} ({seen_titles[title]})"
            doc_type, chunks = chunk_document(markdown, rel, doc_title=title)
            for c in chunks:
                kinds[c.kind] += 1
                fout.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                laws.append(to_law_record(c))
                if c.kind == FAQ_KIND:
                    cases.append(to_case_record(len(cases), c))
            manifest.append({"source_file": rel, "doc_title": title, "doc_type": doc_type, "chunks": len(chunks),
                             "kinds": dict(Counter(c.kind for c in chunks))})
            if verbose:
                print(f"{doc_type:8} {len(chunks):5}  {rel}")

    entries = Counter(r["id"] for r in laws)
    dupes = [e for e, n in entries.items() if n > 1]
    if dupes:
        raise ValueError(f"Duplicate chunk entries: {dupes[:5]}")

    (out_dir / "law_to_crime.json").write_text(json.dumps(laws, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "cases_with_feature.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {
        "documents": sum("doc_type" in m for m in manifest),
        "doc_types": dict(Counter(m["doc_type"] for m in manifest if "doc_type" in m)),
        "chunks": len(laws),
        "chunk_kinds": dict(kinds),
        "faq_cases": len(cases),
        "files": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def _newest_mtime(ocr_dir: Path) -> float:
    return max((p.stat().st_mtime for p in ocr_dir.rglob("*.md")), default=0.0)


def ensure_corpus(ocr_dir: str = DEFAULT_OCR_DIR, out_dir: str = DEFAULT_CORPUS_DIR) -> Optional[dict]:
    """Build the corpus from ocr_dir when it is missing or stale. Returns the build summary,
    None when the existing corpus is current, and raises when there is neither."""
    ocr, out = Path(ocr_dir), Path(out_dir)
    target = out / "law_to_crime.json"
    if not ocr.is_dir():
        if target.exists():
            return None
        raise FileNotFoundError(
            f"No corpus at {target} and no OCR directory at '{ocr}' to build it from: copy the OCR "
            f"markdown into {DEFAULT_OCR_DIR}/ (or set OCR_DIR)"
        )
    if target.exists() and target.stat().st_mtime >= _newest_mtime(ocr):
        return None
    return build(ocr, out, verbose=False)
