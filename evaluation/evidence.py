# -*- coding: utf-8 -*-
"""
Chunker-agnostic relevance for retrieval evaluation.

Section labels ('มาตรา 56') only exist for statutes, and differ between chunking strategies, so
datasets for general documents and tables mark relevance with evidence spans instead: a short
verbatim quote from the source document. A retrieved chunk is relevant to an evidence span when
it comes from the same document and contains the span after normalization (markup, whitespace,
quotes and table pipes removed, Thai digits mapped to Arabic). The same test set can therefore
score any chunker, including the raw OCR markdown.
"""

import os
import re
from html import unescape
from typing import Dict, List, Optional

TH_TO_AR = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_TAGS = re.compile(r"</?(?:table|thead|tbody|tr|td|th|br|p|b|i|u|strong|em|span|div|ul|ol|li)(?:\s[^>]*)?/?>", re.IGNORECASE)
_NOISE = re.compile(r"[\s|#*`\"'“”‘’]+")


def normalize(text: str) -> str:
    t = _TAGS.sub(" ", str(text or ""))
    t = unescape(t).translate(TH_TO_AR)
    return _NOISE.sub("", t).lower()


def doc_key(path_or_title: str) -> str:
    """Comparable document identity from an OCR path or a chunk's document title."""
    base = os.path.basename(str(path_or_title or "").replace("\\", "/"))
    base = re.sub(r"\.md$", "", base, flags=re.IGNORECASE)
    return normalize(base)


def same_document(expected_doc: str, chunk_doc: str) -> bool:
    exp, got = doc_key(expected_doc), doc_key(chunk_doc)
    if not exp or not got:
        return False
    return exp == got or exp in got or (len(got) >= 20 and got in exp)


def chunk_matches(evidence: Dict[str, str], chunk_doc: str, chunk_text: str,
                  normalized_text: Optional[str] = None) -> bool:
    if not same_document(evidence.get("doc", ""), chunk_doc):
        return False
    needle = normalize(evidence.get("text", ""))
    hay = normalized_text if normalized_text is not None else normalize(chunk_text)
    return bool(needle) and needle in hay


def evidence_metrics(evidence: List[Dict[str, str]], ranked: List[Dict[str, str]], k: int) -> Dict[str, float]:
    """
    ranked: retrieved chunks in rank order, each {'doc': ..., 'text': ...}.
    Returns evidence recall@k (share of spans found), hit@k (any span), all@k (every span),
    MRR (first relevant rank) and precision@k (share of top-k chunks holding any span).
    """
    if not evidence:
        return {}
    top = ranked[:k]
    norm = [normalize(c.get("text", "")) for c in top]
    found, first, relevant = set(), None, 0
    for rank, (c, n) in enumerate(zip(top, norm), 1):
        hit = False
        for i, ev in enumerate(evidence):
            if chunk_matches(ev, c.get("doc", ""), "", normalized_text=n):
                found.add(i)
                hit = True
        if hit:
            relevant += 1
            first = first or rank
    return {
        "recall": len(found) / len(evidence),
        "hit": 1.0 if found else 0.0,
        "all": 1.0 if len(found) == len(evidence) else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "precision": relevant / max(len(top), 1),
    }
