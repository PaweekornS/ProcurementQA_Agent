# -*- coding: utf-8 -*-
"""
core/preprocess/page_locator.py

Recovers page numbers for corpus chunks from the Typhoon OCR markdown they were cut from.
The OCR files carry '<!-- Page N of M -->' markers; a chunk's page range is found by locating
its opening and closing text in the page-annotated source. Whitespace and markdown punctuation
are ignored when matching because chunking reflows them.
"""

import bisect
import os
import re
from typing import Dict, List, Optional, Tuple

_PAGE_MARKER = re.compile(r"<!--\s*Page\s+(\d+)\s+of\s+(\d+)\s*-->", re.IGNORECASE)
_NOISE = re.compile(r"[\s|#*\-]+")
_LEADING_HEADERS = re.compile(r"^(?:\s*\[[^\]]*\]+)+")
PROBE_LEN = 60


def _squash(text: str) -> str:
    return _NOISE.sub("", text)


class OcrDocument:
    """One OCR markdown file flattened for substring search, with page boundaries kept."""

    def __init__(self, path: str, rel_path: str):
        self.rel_path = rel_path
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        self.total_pages: Optional[int] = None
        parts: List[str] = []
        self._offsets: List[int] = []  # squashed offset where each page starts
        self._pages: List[int] = []
        length = 0
        markers = list(_PAGE_MARKER.finditer(raw))
        for i, m in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
            chunk = _squash(raw[m.end():end])
            self._offsets.append(length)
            self._pages.append(int(m.group(1)))
            self.total_pages = int(m.group(2))
            parts.append(chunk)
            length += len(chunk)
        self.text = "".join(parts) if markers else _squash(raw)

    def _page_at(self, offset: int) -> Optional[int]:
        if not self._pages:
            return None
        return self._pages[max(0, bisect.bisect_right(self._offsets, offset) - 1)]

    def locate(self, chunk_text: str, search_from: int = 0) -> Tuple[Optional[int], Optional[int], int]:
        """
        Return (page_start, page_end, next_search_offset) for a chunk, or (None, None, search_from)
        when its text cannot be found. Searching forward from the previous chunk keeps repeated
        boilerplate (e.g. identical headers) attached to the right occurrence.
        """
        body = _squash(_LEADING_HEADERS.sub("", chunk_text).lstrip("] \n"))
        if not body:
            return None, None, search_from
        head = body[:PROBE_LEN]
        start = self.text.find(head, search_from)
        if start == -1:
            start = self.text.find(head)
        if start == -1:
            return None, None, search_from
        tail = body[-PROBE_LEN:]
        end = self.text.find(tail, start)
        end = start + len(body) - 1 if end == -1 else end + len(tail) - 1
        return self._page_at(start), self._page_at(min(end, len(self.text) - 1)), start + 1


def load_ocr_documents(ocr_dir: str) -> Dict[str, OcrDocument]:
    """Index OCR markdown files by document title (the filename without .md)."""
    docs: Dict[str, OcrDocument] = {}
    if not ocr_dir or not os.path.isdir(ocr_dir):
        return docs
    for root, _, files in os.walk(ocr_dir):
        for name in files:
            if name.lower().endswith(".md"):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, ocr_dir).replace(os.sep, "/")
                docs[os.path.splitext(name)[0].strip()] = OcrDocument(path, rel)
    return docs


def format_page_range(page_start: Optional[int], page_end: Optional[int], total_pages: Optional[int]) -> Optional[str]:
    """'4-6/42', '4/42', '4-6' or None, matching the citation format the frontend displays."""
    if page_start is None:
        return None
    span = str(page_start) if page_end in (None, page_start) else f"{page_start}-{page_end}"
    return f"{span}/{total_pages}" if total_pages else span
