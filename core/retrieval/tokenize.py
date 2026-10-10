# -*- coding: utf-8 -*-
"""Thai word tokenization shared by every lexical index (in-memory BM25 and Qdrant sparse vectors),
so a term means the same thing in local and tri-store mode."""

import re
from typing import List

try:
    from pythainlp.tokenize import word_tokenize as _word_tokenize
except ImportError:  # pragma: no cover - PyThaiNLP is a hard dependency in requirements.txt
    _word_tokenize = None


def thai_tokens(text: str) -> List[str]:
    """Lower-cased newmm tokens of length > 1 (single characters are mostly punctuation and
    Thai vowel/tone fragments)."""
    if not text:
        return []
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if _word_tokenize is not None:
        tokens = _word_tokenize(clean, engine="newmm", keep_whitespace=False)
    else:
        tokens = re.findall(r"\w+", clean)
    return [t.strip().lower() for t in tokens if t.strip() and len(t.strip()) > 1]
