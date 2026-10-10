# -*- coding: utf-8 -*-
"""
OCR markdown -> ordered blocks (heading / paragraph / table), each tagged with its page range.

Page markers ('<!-- Page N of M -->') are removed from the text but kept as block metadata, so
chunkers can cut by meaning across page breaks and still report correct pages. Two OCR artifacts
are repaired here, once, for every chunker:
  * running headers/footers repeated on most pages are dropped;
  * a table that continues on the next page (same columns, header repeated) is merged back.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from .tables import Table, is_pipe_separator, parse_html_table, parse_pipe_table

_PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s+of\s+(\d+)\s*-->", re.IGNORECASE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_HR = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_SIMPLE_HTML = re.compile(r"</?(p|b|strong|i|em|u|ul|ol|div|span)(\s[^>]*)?>", re.IGNORECASE)
_LI = re.compile(r"<li[^>]*>", re.IGNORECASE)
# A line that starts a statutory unit; such lines always start a new paragraph block
UNIT_START = re.compile(
    r"^\s*(?:\*\*)?\s*(มาตรา|ข้อ)\s*([๐-๙\d]+)\s*(ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว)?(?:\*\*)?(?=\s|$|/)"
)

HEADING, PARAGRAPH, TABLE = "heading", "paragraph", "table"


@dataclass
class Block:
    type: str
    text: str = ""
    level: int = 0                    # heading level (1-6)
    table: Optional[Table] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    first_on_page: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    blocks: List[Block]
    total_pages: Optional[int]


def _norm(text: str) -> str:
    # Digits are masked so page-numbered headers ('หน้า ๑๓ เล่ม ๑๓๗ ...') compare equal across pages
    return re.sub(r"[\d๐-๙]+", "0", re.sub(r"[\s#*|]+", "", text or ""))


def _split_pages(markdown: str):
    markers = list(_PAGE_MARKER.finditer(markdown))
    if not markers:
        return [(None, None, markdown)]
    pages = []
    head = markdown[:markers[0].start()]
    if head.strip():
        pages.append((None, None, head))
    for i, m in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(markdown)
        pages.append((int(m.group(1)), int(m.group(2)), markdown[m.end():end]))
    return pages


def _parse_page(text: str, page: Optional[int]) -> List[Block]:
    text = _COMMENT.sub("", text)
    text = _LI.sub("\n- ", text)
    lines = text.splitlines()
    blocks: List[Block] = []
    para: List[str] = []

    def flush():
        if para:
            body = _SIMPLE_HTML.sub("", "\n".join(para)).strip()
            if body:
                blocks.append(Block(PARAGRAPH, text=body, page_start=page, page_end=page))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if "<table" in stripped.lower():
            flush()
            buf = [line]
            while "</table>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                buf.append(lines[i])
            html = "\n".join(buf)
            start = html.lower().find("<table")
            before = _SIMPLE_HTML.sub("", html[:start]).strip()
            if before:
                blocks.append(Block(PARAGRAPH, text=before, page_start=page, page_end=page))
            blocks.append(Block(TABLE, table=parse_html_table(html[start:]), page_start=page, page_end=page))
        elif stripped.startswith("|") and (
            (i + 1 < len(lines) and lines[i + 1].strip().startswith("|")) or is_pipe_separator(stripped)
        ):
            flush()
            buf = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i])
                i += 1
            i -= 1
            blocks.append(Block(TABLE, table=parse_pipe_table(buf), page_start=page, page_end=page))
        elif _HR.match(stripped):
            flush()
        elif _HEADING.match(stripped):
            flush()
            m = _HEADING.match(stripped)
            title = m.group(2).strip().strip("*").strip()
            letters = len(re.findall(r"[ก-ฮะ-ฺเ-๎A-Za-z]", title))
            if title and letters < 0.4 * len(title.replace(" ", "")):
                # Number rows of a table that the OCR rendered as '#' lines are not headings
                para.append(title)
            elif title:
                blocks.append(Block(HEADING, text=title, level=len(m.group(1)), page_start=page, page_end=page))
        elif not stripped:
            flush()
        else:
            if UNIT_START.match(stripped):
                flush()
            para.append(stripped)
        i += 1
    flush()
    return blocks


def _drop_running_headers(pages: List[List[Block]]) -> None:
    """Remove blocks that open (or close) most pages with identical text: page headers/footers."""
    n = len(pages)
    if n < 4:
        return
    edge = Counter()
    for blocks in pages:
        keys = {_norm(b.text) for b in blocks[:2] + blocks[-1:] if b.type != TABLE and len(b.text) <= 150}
        edge.update(k for k in keys if k)
    running = {k for k, c in edge.items() if c >= max(3, 0.3 * n)}
    if not running:
        return
    seen = set()
    for blocks in pages:
        keep = []
        for j, b in enumerate(blocks):
            k = _norm(b.text)
            at_edge = j < 2 or j == len(blocks) - 1
            if at_edge and k in running and b.type != TABLE:
                if k in seen:
                    continue
                seen.add(k)  # keep the first occurrence: it is usually the real title
            keep.append(b)
        blocks[:] = keep


def _merge_headings(blocks: List[Block]) -> List[Block]:
    """'# 1. ภาพรวม...' immediately followed by '# งานก่อสร้างอาคาร' is one wrapped heading."""
    out: List[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (
            prev is not None and b.type == HEADING and prev.type == HEADING
            and b.level == prev.level and not re.match(r"^[\d๐-๙]+[.)]|^(หมวด|ส่วนที่|บทที่|มาตรา|ข้อ)\s", b.text)
            and len(prev.text) + len(b.text) < 200
        ):
            prev.text = f"{prev.text} {b.text}"
            prev.page_end = b.page_end
            continue
        out.append(b)
    return out


def _merge_split_tables(blocks: List[Block]) -> List[Block]:
    """A table that opens a page and has the same columns as the table closing the previous
    page is its continuation; header rows repeated by the OCR are not duplicated."""
    out: List[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        if (
            prev is not None and b.type == TABLE and prev.type == TABLE and b.first_on_page
            and prev.page_end is not None and b.page_start is not None
            and b.page_start == prev.page_end + 1 and prev.table.same_shape(b.table)
        ):
            prev.table.rows.extend(b.table.rows)
            prev.table.row_pages.extend(b.table.row_pages)
            prev.page_end = b.page_end
            prev.meta["continued"] = True
            continue
        out.append(b)
    return out


def parse_document(markdown: str) -> ParsedDocument:
    pages = _split_pages(markdown)
    per_page: List[List[Block]] = []
    total = None
    for num, tot, text in pages:
        total = tot or total
        page_blocks = _parse_page(text, num)
        for b in page_blocks:
            if b.type == TABLE:
                b.table.row_pages = [num] * len(b.table.rows)
        per_page.append(page_blocks)
    _drop_running_headers(per_page)
    blocks: List[Block] = []
    for page_blocks in per_page:
        if page_blocks:
            page_blocks[0].first_on_page = True
        blocks.extend(page_blocks)
    blocks = _merge_split_tables(blocks)
    blocks = _merge_headings(blocks)
    return ParsedDocument(blocks=blocks, total_pages=total)
