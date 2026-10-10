# -*- coding: utf-8 -*-
"""
StructuredChunker: heading-aware chunking for general documents (announcements, guidelines,
price-calculation rules, TOR, contracts without clause numbering).

  * The heading tree gives each chunk its section path; a section is the unit of meaning.
  * A section's text is packed up to max_chars at paragraph boundaries; small sibling sections
    are merged so chunks are not one-liners.
  * Every table becomes its own chunk (row groups for long tables, header repeated) captioned
    with its section path and the sentence introducing it.
  * Documents without markdown headings fall back to numbered lines ('1.', '๒.๑') as headings,
    and to plain paragraph packing when there are none: the recursive case.
"""

import re
from typing import List, Optional

from .blocks import HEADING, ITEM_START, PARAGRAPH, TABLE, UNIT_START, Block, ParsedDocument
from .schema import SECTION, TABLE as TABLE_KIND, TH_TO_AR, Chunk, clean_label, make_parent_id
from .tables import split_table
from .text import pack

_NUMBERED = re.compile(r"^(?:\*\*)?\s*([\d๐-๙]+(?:\.[\d๐-๙]+)*)[.)]?\s+\S")
_SENTENCE_END = re.compile(r"[:;,]\s*$|ดังนี้\s*$|และ\s*$|หรือ\s*$")


def _promote_numbered_lines(blocks: List[Block]) -> List[Block]:
    """In heading-less documents, short single-line numbered paragraphs act as headings."""
    if sum(b.type == HEADING for b in blocks) >= 3:
        return blocks
    out = []
    for b in blocks:
        m = _NUMBERED.match(b.text) if b.type == PARAGRAPH else None
        if m and "\n" not in b.text and len(b.text) <= 80 and not _SENTENCE_END.search(b.text):
            depth = m.group(1).count(".") + 1
            out.append(Block(HEADING, text=b.text.strip("* "), level=min(depth + 1, 6),
                             page_start=b.page_start, page_end=b.page_end))
        else:
            out.append(b)
    return out


def _item_headings(blocks: List[Block], level: int = 4) -> List[Block]:
    """Give every 'ข้อ N' / 'N)' item its own heading 'ข้อ N <opening words>' so each item is a
    section that can be cited by number (attachments number their items this way)."""
    out = []
    for b in blocks:
        m = (UNIT_START.match(b.text) or ITEM_START.match(b.text)) if b.type == PARAGRAPH else None
        if m:
            num = (m.group(2) if m.re is UNIT_START else m.group(1)).translate(TH_TO_AR)
            opening = clean_label(b.text[m.end(0) if m.re is UNIT_START else m.end(1) + 1:], 50)
            out.append(Block(HEADING, text=f"ข้อ {num} {opening}".strip(), level=level,
                             page_start=b.page_start, page_end=b.page_end))
        out.append(b)
    return out


class StructuredChunker:
    doc_type = "general"

    def __init__(self, max_chars: int = 1500, min_chars: int = 300, max_table_chars: int = 2000,
                 caption_chars: int = 250, item_headings: bool = False):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.max_table_chars = max_table_chars
        self.caption_chars = caption_chars
        self.item_headings = item_headings

    def chunk(self, doc: ParsedDocument, doc_title: str, source_file: str,
              path_prefix: Optional[List[str]] = None) -> List[Chunk]:
        blocks = _item_headings(doc.blocks) if self.item_headings else _promote_numbered_lines(doc.blocks)
        chunks: List[Chunk] = []
        stack: List[tuple] = [(0, p) for p in (path_prefix or [])]  # (level, title)
        section: List[Block] = []
        held: Optional[dict] = None      # small section waiting to be merged with the next one
        table_no = 0

        def path() -> List[str]:
            return [clean_label(t, 80) for _, t in stack]

        def flush_section(final: bool = False):
            nonlocal section, held, table_no
            sec_path = path()
            label = self._label(sec_path, doc_title)
            pieces = []
            for idx, b in enumerate(section):
                if b.type == TABLE:
                    table_no += 1
                    caption = self._caption(section, idx)
                    chunks.extend(self._table_chunks(doc, doc_title, source_file, b, sec_path, label, table_no, caption))
                else:
                    pieces.append((b.text, b.page_start, b.page_end))
            section = []
            if pieces and sec_path:
                # The heading stays in the text so merged sections remain distinguishable
                pieces.insert(0, (f"## {sec_path[-1]}", pieces[0][1], pieces[0][1]))
            size = sum(len(p[0]) for p in pieces)

            if held:
                if pieces and held["size"] + size <= self.max_chars:
                    pieces = held["pieces"] + pieces
                    size += held["size"]
                    label, sec_path = held["label"], held["path"]
                else:
                    self._emit_text(chunks, doc, doc_title, source_file, held["pieces"], held["path"], held["label"])
                held = None

            if not pieces:
                return
            if size < self.min_chars and not final:
                held = {"pieces": pieces, "path": sec_path, "label": label, "size": size}
                return
            self._emit_text(chunks, doc, doc_title, source_file, pieces, sec_path, label)

        for b in blocks:
            if b.type == HEADING:
                flush_section()
                while stack and stack[-1][0] >= b.level and stack[-1][0] > 0:
                    stack.pop()
                stack.append((b.level, b.text))
            else:
                section.append(b)
        flush_section(final=True)
        if held:
            self._emit_text(chunks, doc, doc_title, source_file, held["pieces"], held["path"], held["label"])
        return chunks

    @staticmethod
    def _label(sec_path: List[str], doc_title: str) -> str:
        if not sec_path:
            return "เนื้อหา"
        return clean_label(" > ".join(sec_path[-2:]), 120) or "เนื้อหา"

    def _caption(self, section: List[Block], idx: int) -> str:
        """The paragraph right before a table usually names it ('ตารางที่ 1 ...', 'อัตรา ... ดังนี้')."""
        for j in range(idx - 1, -1, -1):
            b = section[j]
            if b.type == TABLE:
                break
            if b.type == PARAGRAPH:
                return b.text[-self.caption_chars:]
        return ""

    def _table_chunks(self, doc, doc_title, source_file, b, sec_path, label, table_no, caption) -> List[Chunk]:
        cols = set(b.table.columns())
        values = {c for r in b.table.rows for c in r if c and c not in cols}
        if sum(len(v) for v in values) < 15:
            return []   # an empty form (header and blank rows only) carries nothing to retrieve
        parts = split_table(b.table, self.max_table_chars)
        parent = make_parent_id(doc_title, f"table{table_no}")
        out = []
        for k, (md, ps, pe) in enumerate(parts, 1):
            suffix = f" ตารางที่ {table_no}" + (f" (ส่วน {k}/{len(parts)})" if len(parts) > 1 else "")
            content = f"{caption}\n\n{md}" if caption else md
            out.append(Chunk(
                doc_title=doc_title, source_file=source_file, doc_type=self.doc_type, kind=TABLE_KIND,
                label=clean_label(label + suffix, 160), content=content, section_path=sec_path,
                page_start=ps or b.page_start, page_end=pe or b.page_end, total_pages=doc.total_pages,
                parent_id=parent if len(parts) > 1 else None,
                metadata={"table_no": table_no, "columns": b.table.drop_empty_columns().columns()},
            ))
        return out

    def _emit_text(self, chunks, doc, doc_title, source_file, pieces, sec_path, label):
        groups = pack(pieces, self.max_chars)
        parent = make_parent_id(doc_title, "|".join(sec_path) + f"|{groups[0][1]}") if len(groups) > 1 else None
        for n, (text, ps, pe) in enumerate(groups, 1):
            chunks.append(Chunk(
                doc_title=doc_title, source_file=source_file, doc_type=self.doc_type, kind=SECTION,
                label=label if len(groups) == 1 else f"{label} (ตอนที่ {n})", content=text,
                section_path=sec_path, page_start=ps, page_end=pe, total_pages=doc.total_pages,
                parent_id=parent,
            ))
