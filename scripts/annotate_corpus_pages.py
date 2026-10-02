# -*- coding: utf-8 -*-
"""
scripts/annotate_corpus_pages.py

Embeds source-document page metadata into the committed corpus (datas/law_to_crime.json) so that
deployments do not need the raw Typhoon OCR markdown, which is kept out of git.

For every chunk it adds:
  source_file  - path of the OCR markdown relative to the OCR root, e.g. "พรบ/พระราชบัญญัติ...md"
  page_start   - first page the chunk text appears on (null if it could not be located)
  page_end     - last page
  total_pages  - page count of the source document

Run locally wherever datas/typhoon_ocr/ exists, after any corpus regeneration:
    python scripts/annotate_corpus_pages.py
    python scripts/annotate_corpus_pages.py --laws-path datas/law_to_crime.json --ocr-dir datas/typhoon_ocr
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.preprocess.page_locator import load_ocr_documents

PAGE_FIELDS = ("source_file", "page_start", "page_end", "total_pages")


def annotate(laws_path: str, ocr_dir: str) -> None:
    ocr_docs = load_ocr_documents(ocr_dir)
    if not ocr_docs:
        raise SystemExit(f"No OCR markdown found under '{ocr_dir}'.")

    with open(laws_path, encoding="utf-8") as f:
        records = json.load(f)

    search_pos: Dict[str, int] = {}
    located = no_source = 0
    for rec in records:
        doc_name = str(rec.get("doc_name") or str(rec.get("id", "")).split("|")[0]).strip()
        ocr_doc = ocr_docs.get(doc_name)
        if not ocr_doc or not rec.get("items"):
            no_source += 1
            for key in PAGE_FIELDS:
                rec[key] = None
            continue
        page_start, page_end, search_pos[doc_name] = ocr_doc.locate(
            str(rec["items"][0].get("text", "")), search_pos.get(doc_name, 0)
        )
        rec["source_file"] = ocr_doc.rel_path
        rec["page_start"] = page_start
        rec["page_end"] = page_end
        rec["total_pages"] = ocr_doc.total_pages
        located += page_start is not None

    with open(laws_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(
        f"Annotated {laws_path}: {located}/{len(records)} chunks with page ranges; "
        f"{no_source} chunks have no OCR source (e.g. FAQ-derived)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed OCR page metadata into the corpus JSON")
    parser.add_argument("--laws-path", default="datas/law_to_crime.json")
    parser.add_argument("--ocr-dir", default="datas/typhoon_ocr")
    args = parser.parse_args()
    annotate(args.laws_path, args.ocr_dir)
