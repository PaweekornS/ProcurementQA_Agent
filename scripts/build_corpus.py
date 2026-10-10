# -*- coding: utf-8 -*-
"""
scripts/build_corpus.py

Chunks the OCR corpus (data_ocr/) with core/chunking into outputs/corpus/ (see core/chunking/corpus.py
for the files). The migration and the local pipeline call this automatically when the corpus is
missing or older than the OCR files; run it by hand only to inspect the chunks.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --ocr-dir data_ocr --out outputs/corpus

Re-ingest after changing the chunker or the OCR files (Docker stack running):
    python scripts/migrate_to_tri_store.py --reset-corpus
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.chunking.corpus import DEFAULT_CORPUS_DIR, DEFAULT_OCR_DIR, build  # noqa: E402,F401


def main():
    parser = argparse.ArgumentParser(description="Chunk the OCR corpus into tri-store migration input")
    parser.add_argument("--ocr-dir", default=DEFAULT_OCR_DIR)
    parser.add_argument("--out", default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args()
    s = build(Path(args.ocr_dir), Path(args.out))
    print(f"\n{s['documents']} documents {s['doc_types']} -> {s['chunks']} chunks {s['chunk_kinds']}, "
          f"{s['faq_pairs']} FAQ pairs written to {args.out}")


if __name__ == "__main__":
    main()
