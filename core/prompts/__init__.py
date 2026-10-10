"""Unified import interface for prompt module (ProcurementQA Agent)"""

from typing import Optional

# Answer generation prompts
from .answer import (
    ANSWER_LEGAL_QA_PROMPT,
)

# Agent (decompose / audit / refine) prompts
from .agent import (
    INTENT_DECOMPOSE_PROMPT,
    AUDIT_COMPLETENESS_PROMPT,
    QUERY_REFINE_PROMPT,
)

_PROMPTS = {
    "th": {
        "ANSWER_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "ANSWER_LEGAL_QA_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "INTENT_DECOMPOSE_PROMPT": INTENT_DECOMPOSE_PROMPT,
        "AUDIT_COMPLETENESS_PROMPT": AUDIT_COMPLETENESS_PROMPT,
        "QUERY_REFINE_PROMPT": QUERY_REFINE_PROMPT,
        "ANSWER_INPUT_TEMPLATE": (
            "ข้อกฎหมายและระเบียบ:\n-----\n{law}\n-----\n"
            "คำถาม/ข้อหารือ:\n-----\n{case}\n-----\nคำตอบ (JSON):"
        ),
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
    "ANSWER_LEGAL_QA_PROMPT",
    "INTENT_DECOMPOSE_PROMPT",
    "AUDIT_COMPLETENESS_PROMPT",
    "QUERY_REFINE_PROMPT",
]
