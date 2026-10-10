# -*- coding: utf-8 -*-
"""
Builds the chunked corpus from the OCR markdown directory (data_ocr/):

    <out>/chunks.jsonl    one Chunk record per line (core/chunking/schema.py): the corpus contract
                          read by the tri-store migration and the local in-memory store
    <out>/manifest.json   per-document type and chunk counts

The corpus is derived data (it may carry confidential content) and is never committed:
ensure_corpus() rebuilds it whenever it is missing or older than the OCR files.
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from . import FAQ_KIND, chunk_document, doc_title_from_path

DEFAULT_OCR_DIR = "data_ocr"
DEFAULT_CORPUS_DIR = os.path.join("outputs", "corpus")
CHUNKS_FILE = "chunks.jsonl"


def build(ocr_dir: Path, out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, entries = [], Counter()
    kinds = Counter()
    tmp = out_dir / (CHUNKS_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fout:
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
                entries[c.entry] += 1
                fout.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            manifest.append({"source_file": rel, "doc_title": title, "doc_type": doc_type, "chunks": len(chunks),
                             "kinds": dict(Counter(c.kind for c in chunks))})
            if verbose:
                print(f"{doc_type:8} {len(chunks):5}  {rel}")

    dupes = [e for e, n in entries.items() if n > 1]
    if dupes:
        tmp.unlink()
        raise ValueError(f"Duplicate chunk entries: {dupes[:5]}")
    # Replace atomically so a reader never sees a half-written corpus
    os.replace(tmp, out_dir / CHUNKS_FILE)

    summary = {
        "documents": sum("doc_type" in m for m in manifest),
        "doc_types": dict(Counter(m["doc_type"] for m in manifest if "doc_type" in m)),
        "chunks": sum(entries.values()),
        "chunk_kinds": dict(kinds),
        "faq_pairs": kinds.get(FAQ_KIND, 0),
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
    target = out / CHUNKS_FILE
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
