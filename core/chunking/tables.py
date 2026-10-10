# -*- coding: utf-8 -*-
"""
Table parsing and serialization.

The OCR emits two table syntaxes: markdown pipe tables and HTML <table> with rowspan/colspan.
Both are parsed into a rectangular grid (spans are expanded by repeating the cell), header rows
are collapsed into one column name per column, and the table is re-serialized as a markdown
pipe table. Long tables are split into row groups that each repeat the header, so every chunk
can be read on its own.
"""

import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_PIPE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _clean_cell(text: str) -> str:
    t = _BR.sub(" ", text or "")
    t = _TAG.sub(" ", t)
    t = unescape(t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Table:
    header: List[List[str]] = field(default_factory=list)   # header rows (possibly several)
    rows: List[List[str]] = field(default_factory=list)
    row_pages: List[Optional[int]] = field(default_factory=list)   # page of each body row

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.header + self.rows), default=0)

    def columns(self) -> List[str]:
        """One name per column: the distinct header cells of that column joined with ' / '."""
        names = []
        for c in range(self.n_cols):
            parts: List[str] = []
            for row in self.header:
                v = row[c] if c < len(row) else ""
                if v and v not in parts:
                    parts.append(v)
            names.append(" / ".join(parts))
        return names

    def drop_empty_columns(self) -> "Table":
        keep = [c for c in range(self.n_cols)
                if any(c < len(r) and r[c] for r in self.header + self.rows)]
        pick = lambda r: [r[c] if c < len(r) else "" for c in keep]
        return Table([pick(r) for r in self.header], [pick(r) for r in self.rows], list(self.row_pages))

    def same_shape(self, other: "Table") -> bool:
        if self.n_cols != other.n_cols:
            return False
        return not self.header or not other.header or self.columns() == other.columns()


class _HtmlTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[dict]] = []
        self._row: Optional[List[dict]] = None
        self._cell: Optional[dict] = None
        self._in_thead = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            if self._row is None:
                self._row = []
            self._cell = {
                "text": "",
                "rowspan": _int(a.get("rowspan")),
                "colspan": _int(a.get("colspan")),
                "header": tag == "th" or self._in_thead,
            }
        elif tag == "br" and self._cell is not None:
            self._cell["text"] += " "

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data


def _int(v) -> int:
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return 1


def parse_html_table(html: str) -> Table:
    p = _HtmlTableParser()
    p.feed(html)
    if p._row:
        p.rows.append(p._row)

    grid: List[List[Optional[str]]] = []
    is_header: List[bool] = []
    pending = {}  # (row, col) -> text carried down by rowspan
    for r, row in enumerate(p.rows):
        out: List[Optional[str]] = []
        c = 0
        cells = iter(row)
        header_row = bool(row) and all(cell["header"] for cell in row)
        while True:
            while (r, c) in pending:
                out.append(pending.pop((r, c)))
                c += 1
            cell = next(cells, None)
            if cell is None:
                break
            text = _clean_cell(cell["text"])
            for dc in range(cell["colspan"]):
                out.append(text)
                for dr in range(1, cell["rowspan"]):
                    pending[(r + dr, c + dc)] = text
            c += cell["colspan"]
        # Trailing carried cells beyond the last real cell
        while (r, c) in pending:
            out.append(pending.pop((r, c)))
            c += 1
        grid.append(out)
        is_header.append(header_row)

    n_header = 0
    if any(is_header):
        while n_header < len(is_header) and is_header[n_header]:
            n_header += 1
    elif p.rows:
        # No <th>: the first row is a header when it spans several rows (multi-level header)
        # or when its cells are all short, non-numeric labels.
        first = p.rows[0]
        n_header = max(cell["rowspan"] for cell in first) if first else 0
        if n_header == 1 and not _looks_like_header([_clean_cell(c["text"]) for c in first]):
            n_header = 0

    width = max((len(r) for r in grid), default=0)
    grid = [[(v or "") for v in r] + [""] * (width - len(r)) for r in grid]
    return promote_subheader_rows(Table(header=grid[:n_header], rows=grid[n_header:]))


def _looks_like_header(cells: List[str]) -> bool:
    filled = [c for c in cells if c]
    if len(filled) < 2:
        return False
    numeric = sum(bool(re.fullmatch(r"[\d๐-๙.,\-%() ]+", c)) for c in filled)
    return numeric == 0 and all(len(c) <= 40 for c in filled)


def parse_pipe_table(lines: List[str]) -> Table:
    def split(line: str) -> List[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [_clean_cell(c) for c in s.split("|")]

    sep_idx = next((i for i, l in enumerate(lines) if _PIPE_SEP.match(l.strip())), None)
    rows = [split(l) for i, l in enumerate(lines) if i != sep_idx]
    n_header = sep_idx if sep_idx is not None else 0
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return promote_subheader_rows(Table(header=rows[:n_header], rows=rows[n_header:]))


def is_pipe_separator(line: str) -> bool:
    return bool(_PIPE_SEP.match(line.strip()))


def promote_subheader_rows(table: Table, max_rows: int = 2) -> Table:
    """OCR often leaves the second level of a two-level header ('| | | จำนวนตามชั้น | ลักษณะโครงการ |')
    as the first body row. Such a row starts with empty cells and holds only short labels."""
    if not table.header:
        return table
    moved = 0
    while moved < max_rows and table.rows:
        row = table.rows[0]
        filled = [c for c in row if c]
        if (
            not row or row[0] or not filled or len(filled) == len(row)
            or any(len(c) > 40 or re.search(r"[0-9]", c) or re.fullmatch(r"[๐-๙.,\s]+", c) for c in filled)
        ):
            break
        table.header.append(table.rows.pop(0))
        if table.row_pages:
            table.row_pages.pop(0)
        moved += 1
    return table


def to_markdown(table: Table, rows: Optional[List[List[str]]] = None) -> str:
    t = table.drop_empty_columns() if rows is None else table
    body = t.rows if rows is None else rows
    cols = t.columns()
    lines = []
    if any(cols):
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
    for r in body:
        if any(r):
            lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def split_table(table: Table, max_chars: int) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Serialize a table into (markdown, page_start, page_end) pieces of at most ~max_chars,
    each repeating the column header. A single over-long row is kept whole."""
    t = table.drop_empty_columns()
    header_md = to_markdown(t, rows=[])
    pages = t.row_pages if len(t.row_pages) == len(t.rows) else [None] * len(t.rows)
    pieces, buf, buf_pages, size = [], [], [], len(header_md)

    def emit():
        known = [p for p in buf_pages if p is not None]
        pieces.append((to_markdown(t, rows=buf), min(known) if known else None, max(known) if known else None))

    for r, page in zip(t.rows, pages):
        if not any(r):
            continue
        line_len = sum(len(c) for c in r) + 3 * len(r) + 2
        if buf and size + line_len > max_chars:
            emit()
            buf, buf_pages, size = [], [], len(header_md)
        buf.append(r)
        buf_pages.append(page)
        size += line_len
    if buf or not pieces:
        emit()
    return pieces
