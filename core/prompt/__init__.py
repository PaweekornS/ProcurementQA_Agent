"""Unified import interface for prompt module (Thai Procurement LegalGraphRAG)"""

from typing import Optional

# Preprocess prompts
from .preprocess import (
    GET_PROCUREMENT_FEATURES_PROMPT,
)

# Judge prompts
from .judge import (
    ANSWER_LEGAL_QA_PROMPT,
)

# CRAG prompts
from .crag import (
    INTENT_DECOMPOSE_PROMPT,
    AUDIT_COMPLETENESS_PROMPT,
    QUERY_REFINE_PROMPT,
)

_PROMPTS = {
    "th": {
        "GET_FEATURES_PROMPT": GET_PROCUREMENT_FEATURES_PROMPT,
        "GET_FEATURES_INPUT_TEMPLATE": "\nคำถาม/ข้อหารือ: {fact}",
        "CASE_SEG_PROMPT": "คำถาม/ข้อหารือ:\n{fact}\n\nจงสรุปข้อเท็จจริงและประเด็นคำถามให้กระชับ ชัดเจน:",
        "JUDGE_CRIME_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "JUDGE_CRIME_ALL_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "ANSWER_LEGAL_QA_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "INTENT_DECOMPOSE_PROMPT": INTENT_DECOMPOSE_PROMPT,
        "AUDIT_COMPLETENESS_PROMPT": AUDIT_COMPLETENESS_PROMPT,
        "QUERY_REFINE_PROMPT": QUERY_REFINE_PROMPT,
        "JUDGE_CRIME_ALL_INPUT_TEMPLATE": (
            "ข้อกฎหมายและระเบียบ:\n-----\n{law}\n-----\n"
            "คำถาม/ข้อหารือ:\n-----\n{case}\n-----\nคำตอบ (JSON):"
        ),
        "SUMMARIZE_TEXTS_PROMPT": "จงสรุปสาระสำคัญของเอกสารและประเด็นกฎหมายต่อไปนี้:\n",
        "RERANK_CLUSTERS_PROMPT_TEMPLATE": "จงจัดลำดับความเกี่ยวข้องของกลุ่มกฎหมายต่อไปนี้:\n{cluster_summaries}\nคำถาม: {query_text}\nลำดับ:",
        "RERANK_PROMPT_TEMPLATE": "จงจัดลำดับความเกี่ยวข้องของข้อกฎหมายต่อไปนี้:\n{neighbor_summaries}\nคำถาม: {query_text}\nลำดับ:",
    },
}

_LANGUAGE_ALIASES = {
    "th": "th",
    "thai": "th",
    "default": "th",
}

_current_language = "th"


def set_prompt_language(language: str) -> None:
    """Set runtime prompt language."""
    global _current_language
    normalized = _LANGUAGE_ALIASES.get(language.strip().lower())
    if normalized is None:
        raise ValueError("prompt_language must be one of: th, thai")
    _current_language = normalized


def get_prompt(name: str, language: Optional[str] = None) -> str:
    """Return a prompt by name for the configured language."""
    selected_language = _current_language
    if language is not None:
        selected_language = _LANGUAGE_ALIASES.get(language.strip().lower())
        if selected_language is None:
            raise ValueError("prompt_language must be one of: th, thai")
    try:
        return _PROMPTS[selected_language][name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt: {name}") from exc


__all__ = [
    "set_prompt_language",
    "get_prompt",
    "GET_PROCUREMENT_FEATURES_PROMPT",
    "ANSWER_LEGAL_QA_PROMPT",
    "INTENT_DECOMPOSE_PROMPT",
    "AUDIT_COMPLETENESS_PROMPT",
    "QUERY_REFINE_PROMPT",
]
