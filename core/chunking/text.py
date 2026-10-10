# -*- coding: utf-8 -*-
"""Size-bounded packing of text pieces that never cuts inside a Thai word."""

import re
from typing import List, Optional, Tuple

# (text, page_start, page_end)
Piece = Tuple[str, Optional[int], Optional[int]]


def split_long_text(text: str, max_chars: int) -> List[str]:
    """Split one over-long paragraph at line breaks, then at spaces (Thai marks phrase and
    sentence boundaries with a space), never inside a word. A single unbroken run longer
    than max_chars is kept whole rather than cut mid-word."""
    if len(text) <= max_chars:
        return [text]
    out: List[str] = []
    buf = ""
    for unit in _units(text):
        if buf and len(buf) + len(unit) > max_chars:
            out.append(buf.strip())
            buf = ""
        buf += unit
    if buf.strip():
        out.append(buf.strip())
    return out


def _units(text: str) -> List[str]:
    lines = text.splitlines(keepends=True)
    if len(lines) > 1 and max(len(l) for l in lines) < len(text):
        units = []
        for l in lines:
            units.extend(re.findall(r"\S+\s*", l) if len(l) > 400 else [l])
        return units
    return re.findall(r"\S+\s*", text)


def pack(pieces: List[Piece], max_chars: int, sep: str = "\n\n") -> List[Piece]:
    """Greedily pack pieces into groups of at most ~max_chars, keeping page ranges."""
    groups: List[Piece] = []
    buf, p0, p1 = "", None, None
    for text, ps, pe in pieces:
        for part in split_long_text(text, max_chars):
            if buf and len(buf) + len(sep) + len(part) > max_chars:
                groups.append((buf, p0, p1))
                buf, p0, p1 = "", None, None
            buf = f"{buf}{sep}{part}" if buf else part
            p0 = p0 if p0 is not None else ps
            p1 = pe if pe is not None else p1
    if buf:
        groups.append((buf, p0, p1))
    return groups


def page_range(pieces: List[Piece]) -> Tuple[Optional[int], Optional[int]]:
    starts = [p for _, p, _ in pieces if p is not None]
    ends = [p for _, _, p in pieces if p is not None]
    return (min(starts) if starts else None, max(ends) if ends else None)
