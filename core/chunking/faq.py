# -*- coding: utf-8 -*-
"""
FAQChunker: one chunk per question/answer pair of a Q&A table (e.g. the CGD procurement FAQ).

The FAQ is a single table spanning many pages, so rows have to be stitched back together:
  * columns are located by header names (ข้อ / คำถาม / คำตอบ / หมายเหตุ) on every page;
  * a row without a number at the top of a page continues the previous pair;
  * a row with a single short cell and no number is a category heading;
  * numbering restarts per category, so the label is '<category> ข้อ N'.
"""

import re
from typing import Dict, List, Optional

from .blocks import TABLE, ParsedDocument
from .schema import FAQ, Chunk, TH_TO_AR, clean_label

_COLS = {"num": ("ข้อ", "ลำดับ", "ที่"), "q": ("คำถาม", "ข้อหารือ"), "a": ("คำตอบ", "แนวทาง"), "note": ("หมายเหตุ",)}
_NUM = re.compile(r"^[\d๐-๙]+[.)]?$")


def is_faq_table(table) -> bool:
    cols = " ".join(table.columns()) + " " + " ".join(" ".join(r) for r in table.rows[:1])
    return "คำถาม" in cols and "คำตอบ" in cols


def _column_map(names: List[str], rows: List[List[str]]) -> Optional[Dict[str, int]]:
    found: Dict[str, int] = {}
    for key, words in _COLS.items():
        for i, n in enumerate(names):
            if any(n.strip() == w or n.strip().startswith(w) for w in words) and i not in found.values():
                found[key] = i
                break
    if "q" not in found or "a" not in found:
        return None
    # OCR pipe tables can shift the body one column off the header ('||ข้อ|||คำถาม' over
    # '|๕|||...'): snap every header column to the nearest column that actually holds data.
    occupied = [c for c in range(max((len(r) for r in rows), default=0))
                if sum(1 for r in rows if c < len(r) and r[c].strip()) >= max(1, 0.1 * len(rows))]
    if occupied:
        found = {k: min(occupied, key=lambda c: (abs(c - i), c)) for k, i in found.items()}
    return found


class FAQChunker:
    doc_type = "faq"

    def chunk(self, doc: ParsedDocument, doc_title: str, source_file: str) -> List[Chunk]:
        records: List[dict] = []
        category, cmap = "", None
        for b in doc.blocks:
            if b.type != TABLE:
                continue
            t = b.table
            header_map = _column_map(t.columns(), t.rows) if t.header else None
            rows = t.rows
            pages = t.row_pages if len(t.row_pages) == len(rows) else [b.page_start] * len(rows)
            if header_map is None and rows:
                # HTML pages sometimes carry the header as the first body row
                header_map = _column_map(rows[0], rows[1:])
                if header_map:
                    rows, pages = rows[1:], pages[1:]
            if header_map:
                cmap = header_map
            if cmap is None:
                continue
            prev_page = None
            for row, page in zip(rows, pages):
                get = lambda k: row[cmap[k]].strip() if k in cmap and cmap[k] < len(row) else ""
                num, q, a, note = get("num"), get("q"), get("a"), get("note")
                if note == a or note == "หมายเหตุ":
                    note = ""   # colspan duplicated the answer into the note column
                filled = list(dict.fromkeys(c.strip() for c in row if c.strip()))
                page_top = page != prev_page
                prev_page = page
                if not filled:
                    continue
                if _NUM.match(num):
                    records.append({"category": category, "num": int(num.rstrip(".)").translate(TH_TO_AR)),
                                    "q": q, "a": a, "note": note, "ps": page, "pe": page})
                elif records and (page_top or a or note):
                    # Text running over from the previous row (page break or wrapped cell)
                    self._append(records[-1], q, a, note, page)
                elif len(filled) == 1 and len(filled[0]) <= 100:
                    category = filled[0]
                elif records:
                    self._append(records[-1], q, a, note, page)

        chunks = []
        for r in records:
            body = f"คำถาม: {r['q']}\nคำตอบ: {r['a']}"
            if r["note"]:
                body += f"\nหมายเหตุ: {r['note']}"
            label = clean_label(f"{r['category']} ข้อ {r['num']}" if r["category"] else f"ข้อ {r['num']}", 120)
            chunks.append(Chunk(
                doc_title=doc_title, source_file=source_file, doc_type=self.doc_type, kind=FAQ,
                label=label, content=body, section_path=[r["category"]] if r["category"] else [],
                page_start=r["ps"], page_end=r["pe"], total_pages=doc.total_pages,
                metadata={"question": r["q"], "answer": r["a"], "category": r["category"]},
            ))
        return chunks

    @staticmethod
    def _append(rec: dict, q: str, a: str, note: str, page: Optional[int]):
        for key, val in (("q", q), ("a", a), ("note", note)):
            if val:
                rec[key] = f"{rec[key]} {val}".strip()
        if page is not None:
            rec["pe"] = page
