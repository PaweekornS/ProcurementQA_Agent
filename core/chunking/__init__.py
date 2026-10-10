# -*- coding: utf-8 -*-
"""
Document chunking for the procurement corpus and tenant uploads.

    chunk_document(markdown, rel_path) -> (doc_type, [Chunk])

The document is parsed once into page-tagged blocks (core/chunking/blocks.py), classified, and
handed to the chunker that matches its structure:

    statute  - numbered มาตรา / ข้อ units          -> StatuteChunker  (one chunk per unit)
    faq      - question/answer table               -> FAQChunker      (one chunk per pair)
    general  - everything else (headings, tables)  -> StructuredChunker

All chunkers return the same Chunk schema, so retrieval and the stores do not depend on the
strategy that produced a chunk.
"""

import os
import re
from typing import List, Optional, Tuple

from .blocks import TABLE, ParsedDocument, parse_document
from .faq import FAQChunker, is_faq_table
from .schema import FAQ as FAQ_KIND, Chunk, dedupe_labels
from .statute import StatuteChunker, dominant_unit_kind
from .structured import StructuredChunker

MIN_STATUTE_UNITS = 2
# Label of a document that fits in one chunk: the chunk holds every clause of it
WHOLE_DOCUMENT_LABEL = "ทั้งฉบับ"

__all__ = [
    "Chunk", "FAQ_KIND", "WHOLE_DOCUMENT_LABEL", "ParsedDocument", "parse_document", "classify_document", "chunk_document",
    "doc_title_from_path", "StatuteChunker", "StructuredChunker", "FAQChunker",
]


def doc_title_from_path(rel_path: str) -> str:
    base = os.path.basename(str(rel_path))
    base = re.sub(r"\.md$", "", base, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", base).strip()


def classify_document(doc: ParsedDocument, rel_path: str = "") -> str:
    tables = [b.table for b in doc.blocks if b.type == TABLE]
    if tables and sum(is_faq_table(t) for t in tables) >= max(1, 0.3 * len(tables)):
        return "faq"
    _, n_units = dominant_unit_kind(doc)
    if n_units >= MIN_STATUTE_UNITS:
        return "statute"
    return "general"


def chunk_document(markdown: str, rel_path: str, doc_title: Optional[str] = None,
                   doc_type: Optional[str] = None) -> Tuple[str, List[Chunk]]:
    doc = parse_document(markdown)
    title = doc_title or doc_title_from_path(rel_path)
    doc_type = doc_type or classify_document(doc, rel_path)
    chunker = {"statute": StatuteChunker, "faq": FAQChunker}.get(doc_type, StructuredChunker)()
    chunks = chunker.chunk(doc, title, rel_path.replace("\\", "/"))
    if not chunks and doc_type != "general":
        # A misclassified document must still be indexed
        doc_type, chunks = "general", StructuredChunker().chunk(doc, title, rel_path.replace("\\", "/"))
    if len(chunks) == 1 and chunks[0].kind != "table":
        # Short amendments often carry no 'ข้อ 1' at all; the one chunk is the whole instrument
        chunks[0].label = WHOLE_DOCUMENT_LABEL
    return doc_type, dedupe_labels(chunks)
