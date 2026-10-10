# -*- coding: utf-8 -*-
"""Chunk schema shared by every chunker, so retrieval and the stores never depend on the strategy."""

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Chunk kinds
STATUTE_UNIT = "statute_unit"   # one มาตรา / ข้อ (or a part of a long one)
PREAMBLE = "preamble"           # text before the first unit of a statute
SECTION = "section"             # text under a heading of a general document
TABLE = "table"                 # a table (or a row group of a long one) with its header repeated
FAQ = "faq"                     # one question/answer pair


def clean_label(text: str, limit: int = 120) -> str:
    """Single-line label safe for 'doc | label' entries (no pipes, no markdown)."""
    t = re.sub(r"[|#*_`]+", " ", str(text or ""))
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit].rstrip()


@dataclass
class Chunk:
    doc_title: str
    source_file: str              # path relative to the OCR root
    doc_type: str                 # statute | general | faq
    kind: str                     # one of the kinds above
    label: str                    # 'มาตรา 56', 'ข้อ 7 (ตอนที่ 2)', '2. ข้อกำหนด... > 2.1 ...'
    content: str                  # text shown to the LLM and cited
    section_path: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    total_pages: Optional[int] = None
    parent_id: Optional[str] = None   # section (or unit) the chunk was cut from, for small-to-big expansion
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def entry(self) -> str:
        return f"{self.doc_title} | {self.label}"

    @property
    def chunk_id(self) -> str:
        return "chunk_" + hashlib.md5(self.entry.encode("utf-8")).hexdigest()[:16]

    @property
    def embed_text(self) -> str:
        """Content prefixed with where it comes from; used for dense and BM25 indexing."""
        header = f"[{self.doc_title} | {self.label}]"
        path = " > ".join(p for p in self.section_path if p)
        if path and path not in self.label:
            header += f"\n[หัวข้อ: {path}]"
        return f"{header}\n{self.content}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(entry=self.entry, chunk_id=self.chunk_id)
        return d


def make_parent_id(doc_title: str, key: str) -> str:
    return "parent_" + hashlib.md5(f"{doc_title}|{key}".encode("utf-8")).hexdigest()[:16]


def dedupe_labels(chunks: List[Chunk]) -> List[Chunk]:
    """Entries must be unique per document: repeated labels get a '(ตอนที่ N)' suffix,
    the convention citation_linker.parse_unit_label already understands."""
    groups: Dict[str, List[Chunk]] = {}
    for c in chunks:
        groups.setdefault(c.entry, []).append(c)
    for same in groups.values():
        if len(same) > 1:
            for i, c in enumerate(same, 1):
                c.label = f"{c.label} (ตอนที่ {i})"
    # A suffixed label can still collide with an existing one; fall back to a counter
    seen: Dict[str, int] = {}
    for c in chunks:
        n = seen.get(c.entry, 0)
        if n:
            c.label = f"{c.label} #{n + 1}"
        seen[c.entry] = n + 1
    return chunks
