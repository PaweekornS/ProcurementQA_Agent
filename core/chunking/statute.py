# -*- coding: utf-8 -*-
"""
StatuteChunker: one chunk per มาตรา / ข้อ.

The unit is the legal meaning unit, so it is never merged with its neighbours. A unit longer
than max_unit_chars is split at paragraph boundaries into '(ตอนที่ N)' parts sharing a parent.
Text before the first unit becomes 'คำนำ'; headings after the units that are not chapters
(attached forms, schedules) become ordinary sections.

Numbering is validated as a sequence so that quoted text inside an amendment
("ให้ใช้ความต่อไปนี้แทน ... ข้อ ๗๙ ...") is not mistaken for a new unit.
"""

import re
from typing import List, Optional, Tuple

from .blocks import HEADING, PARAGRAPH, TABLE, UNIT_START, Block, ParsedDocument
from .schema import PREAMBLE, SECTION, STATUTE_UNIT, TABLE as TABLE_KIND, TH_TO_AR, Chunk, clean_label, make_parent_id
from .tables import split_table, to_markdown
from .text import pack, page_range

_CHAPTER = re.compile(r"^\s*(หมวด|ส่วนที่|บทเฉพาะกาล|บทกำหนดโทษ|บททั่วไป)")
_SUFFIX = re.compile(r"(ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว)")


def unit_of(block: Block, kind: Optional[str] = None) -> Optional[Tuple[str, int, str]]:
    if block.type not in (PARAGRAPH, HEADING):
        return None
    m = UNIT_START.match(block.text)
    if not m or (kind and m.group(1) != kind):
        return None
    return m.group(1), int(m.group(2).translate(TH_TO_AR)), m.group(3) or ""


def dominant_unit_kind(doc: ParsedDocument) -> Tuple[Optional[str], int]:
    """The unit kind (มาตรา/ข้อ) with the longest valid numbering sequence, and its unit count."""
    best: Tuple[Optional[str], int] = (None, 0)
    for kind in ("มาตรา", "ข้อ"):
        n = len(_accepted_units(doc.blocks, kind))
        if n > best[1]:
            best = (kind, n)
    return best


def _accepted_units(blocks: List[Block], kind: str) -> List[int]:
    """Indices of blocks that start a unit, under the sequence rule:
    next number (or skip one), same number with a suffix (ทวิ), or a restart at 1 after a heading."""
    accepted, last, heading_since = [], 0, True
    for i, b in enumerate(blocks):
        if b.type == HEADING and not unit_of(b, kind):
            heading_since = True
            continue
        u = unit_of(b, kind)
        if not u:
            continue
        _, num, suffix = u
        ok = (
            (last == 0 and num <= 3)
            or last < num <= last + 2
            or (suffix and num == last)
            or (num == 1 and heading_since and last > 0)
        )
        if ok:
            accepted.append(i)
            last, heading_since = num, False
    return accepted


class StatuteChunker:
    doc_type = "statute"

    def __init__(self, max_unit_chars: int = 2000, max_table_chars: int = 2000):
        self.max_unit_chars = max_unit_chars
        self.max_table_chars = max_table_chars

    def chunk(self, doc: ParsedDocument, doc_title: str, source_file: str) -> List[Chunk]:
        kind, _ = dominant_unit_kind(doc)
        starts = set(_accepted_units(doc.blocks, kind)) if kind else set()
        chunks: List[Chunk] = []
        chapter: List[str] = []
        context = ""            # heading that introduced a restarted numbering (e.g. one contract template)
        restarted = False
        label, body, unit_kind = "คำนำ", [], PREAMBLE
        last_num = 0

        def flush():
            nonlocal body
            if body:
                chunks.extend(self._emit(doc, doc_title, source_file, unit_kind, label, list(chapter), body))
            body = []

        for i, b in enumerate(doc.blocks):
            if i in starts:
                flush()
                _, num, suffix = unit_of(b, kind)
                if num == 1 and last_num > 0:
                    restarted = True
                last_num = num
                label = f"{kind} {num}{(' ' + suffix) if suffix else ''}"
                if restarted and context:
                    label += f" ({clean_label(context, 60)})"
                unit_kind = STATUTE_UNIT
                body = [b]
                continue
            if b.type == HEADING or (b.type == PARAGRAPH and _CHAPTER.match(b.text) and len(b.text) < 150):
                if _CHAPTER.match(b.text):
                    flush()
                    level = 0 if b.text.startswith(("หมวด", "บท")) else 1
                    chapter = chapter[:level] + [clean_label(b.text, 80)]
                    label, unit_kind = clean_label(b.text, 80), SECTION
                    continue
                if unit_kind != PREAMBLE and b.level and b.level <= 2:
                    # A titled block after the units: an attachment, form or schedule
                    flush()
                    context = b.text
                    label, unit_kind = clean_label(b.text, 100), SECTION
                    body = [b]
                    continue
            body.append(b)
        flush()
        return chunks

    def _emit(self, doc, doc_title, source_file, unit_kind, label, chapter, blocks) -> List[Chunk]:
        parent = make_parent_id(doc_title, f"{label}|{blocks[0].page_start}")
        text_pieces, out = [], []
        for b in blocks:
            if b.type == TABLE:
                md = to_markdown(b.table)
                if len(md) <= self.max_table_chars:
                    text_pieces.append((md, b.page_start, b.page_end))
                else:
                    # Long schedules inside a unit become their own table chunks
                    for k, (part, ps, pe) in enumerate(split_table(b.table, self.max_table_chars), 1):
                        out.append(Chunk(
                            doc_title=doc_title, source_file=source_file, doc_type=self.doc_type,
                            kind=TABLE_KIND, label=f"{label} ตาราง (ส่วน {k})", content=part,
                            section_path=chapter + [label], page_start=ps or b.page_start, page_end=pe or b.page_end,
                            total_pages=doc.total_pages, parent_id=parent,
                        ))
            else:
                prefix = "#" * b.level + " " if b.type == HEADING else ""
                text_pieces.append((prefix + b.text, b.page_start, b.page_end))

        if text_pieces:
            groups = pack(text_pieces, self.max_unit_chars) if sum(len(t) for t, _, _ in text_pieces) > self.max_unit_chars \
                else [("\n\n".join(t for t, _, _ in text_pieces), *page_range(text_pieces))]
            for n, (text, ps, pe) in enumerate(groups, 1):
                part_label = f"{label} (ตอนที่ {n})" if len(groups) > 1 else label
                out.insert(n - 1, Chunk(
                    doc_title=doc_title, source_file=source_file, doc_type=self.doc_type,
                    kind=unit_kind, label=part_label, content=text, section_path=chapter,
                    page_start=ps, page_end=pe, total_pages=doc.total_pages,
                    parent_id=parent if len(groups) > 1 or len(out) > 0 else None,
                ))
        return out
