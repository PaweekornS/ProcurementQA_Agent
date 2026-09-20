"""Unified import interface for prompt module"""

import os
from typing import Optional

# Preprocess prompts
from .preprocess import (
    GET_FEATURES_PROMPT,
    GET_PROCUREMENT_FEATURES_PROMPT,
    CASE_SEG_PROMPT,
    PRE_JUDGE_PROMPT,
)

# Judge prompts
from .judge import (
    JUDGE_LAW_PROMPT,
    JUDGE_LAW_PROMPT0,
    JUDGE_LAW_PROMPT1,
    JUDGE_CRIME_PROMPT,
    JUDGE_CRIME_ALL_PROMPT,
    ANSWER_LEGAL_QA_PROMPT,
)

# Retrieval prompts
from .retrieval import (
    RETRIEVE_LAW_PROMPT,
)

# Graph prompts
from .graph import (
    SUMMARIZE_TEXTS_PROMPT,
    RERANK_CLUSTERS_PROMPT_TEMPLATE,
    RERANK_PROMPT_TEMPLATE,
)

_PROMPTS = {
    "en": {
        "GET_FEATURES_PROMPT": GET_FEATURES_PROMPT,
        "CASE_SEG_PROMPT": CASE_SEG_PROMPT,
        "PRE_JUDGE_PROMPT": PRE_JUDGE_PROMPT,
        "JUDGE_LAW_PROMPT": JUDGE_LAW_PROMPT,
        "JUDGE_LAW_PROMPT0": JUDGE_LAW_PROMPT0,
        "JUDGE_LAW_PROMPT1": JUDGE_LAW_PROMPT1,
        "JUDGE_CRIME_PROMPT": JUDGE_CRIME_PROMPT,
        "JUDGE_CRIME_ALL_PROMPT": JUDGE_CRIME_ALL_PROMPT,
        "ANSWER_LEGAL_QA_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "RETRIEVE_LAW_PROMPT": RETRIEVE_LAW_PROMPT,
        "SUMMARIZE_TEXTS_PROMPT": SUMMARIZE_TEXTS_PROMPT,
        "SUMMARIZE_TEXTS_INPUT_PREFIX": "\n**Now process the following input data**: \n",
        "RERANK_CLUSTERS_PROMPT_TEMPLATE": RERANK_CLUSTERS_PROMPT_TEMPLATE,
        "RERANK_PROMPT_TEMPLATE": RERANK_PROMPT_TEMPLATE,
        "GET_FEATURES_INPUT_TEMPLATE": "\nDefendant name: {name}\nCase facts: {fact}",
        "JUDGE_CRIME_ALL_INPUT_TEMPLATE": (
            "Input:\nLegal provision:\n-----\n{law}\n-----\n"
            "Case to be judged:\n-----\n{case}\n-----\nOutput:"
        ),
    },
    "zh": {
        "GET_FEATURES_PROMPT": GET_FEATURES_PROMPT,
        "CASE_SEG_PROMPT": CASE_SEG_PROMPT,
        "PRE_JUDGE_PROMPT": PRE_JUDGE_PROMPT,
        "JUDGE_LAW_PROMPT": JUDGE_LAW_PROMPT,
        "JUDGE_LAW_PROMPT0": JUDGE_LAW_PROMPT0,
        "JUDGE_LAW_PROMPT1": JUDGE_LAW_PROMPT1,
        "JUDGE_CRIME_PROMPT": JUDGE_CRIME_PROMPT,
        "JUDGE_CRIME_ALL_PROMPT": JUDGE_CRIME_ALL_PROMPT,
        "ANSWER_LEGAL_QA_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "RETRIEVE_LAW_PROMPT": RETRIEVE_LAW_PROMPT,
        "SUMMARIZE_TEXTS_PROMPT": SUMMARIZE_TEXTS_PROMPT,
        "SUMMARIZE_TEXTS_INPUT_PREFIX": "\n**Now process the following input data**: \n",
        "RERANK_CLUSTERS_PROMPT_TEMPLATE": RERANK_CLUSTERS_PROMPT_TEMPLATE,
        "RERANK_PROMPT_TEMPLATE": RERANK_PROMPT_TEMPLATE,
        "GET_FEATURES_INPUT_TEMPLATE": "\nDefendant name: {name}\nCase facts: {fact}",
        "JUDGE_CRIME_ALL_INPUT_TEMPLATE": (
            "Input:\nLegal provision:\n-----\n{law}\n-----\n"
            "Case to be judged:\n-----\n{case}\n-----\nOutput:"
        ),
    },
    "th": {
        "GET_FEATURES_PROMPT": GET_PROCUREMENT_FEATURES_PROMPT,
        "CASE_SEG_PROMPT": "คำถาม/ข้อหารือ:\n{fact}\n\nจงสรุปข้อเท็จจริงและประเด็นคำถามให้กระชับ ชัดเจน:",
        "PRE_JUDGE_PROMPT": PRE_JUDGE_PROMPT,
        "JUDGE_LAW_PROMPT": JUDGE_LAW_PROMPT,
        "JUDGE_LAW_PROMPT0": JUDGE_LAW_PROMPT0,
        "JUDGE_LAW_PROMPT1": JUDGE_LAW_PROMPT1,
        "JUDGE_CRIME_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "JUDGE_CRIME_ALL_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "ANSWER_LEGAL_QA_PROMPT": ANSWER_LEGAL_QA_PROMPT,
        "RETRIEVE_LAW_PROMPT": RETRIEVE_LAW_PROMPT,
        "SUMMARIZE_TEXTS_PROMPT": "จงสรุปประเด็นสำคัญของข้อหารือและกฎหมายต่อไปนี้:\n",
        "SUMMARIZE_TEXTS_INPUT_PREFIX": "\nข้อมูลนำเข้า:\n",
        "RERANK_CLUSTERS_PROMPT_TEMPLATE": RERANK_CLUSTERS_PROMPT_TEMPLATE,
        "RERANK_PROMPT_TEMPLATE": RERANK_PROMPT_TEMPLATE,
        "GET_FEATURES_INPUT_TEMPLATE": "\nคำถาม/ข้อหารือ: {fact}",
        "JUDGE_CRIME_ALL_INPUT_TEMPLATE": (
            "ข้อกฎหมายและระเบียบ:\n-----\n{law}\n-----\n"
            "คำถาม/ข้อหารือ:\n-----\n{case}\n-----\nคำตอบ (JSON):"
        ),
    },
}

_LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "zh": "zh",
    "cn": "zh",
    "chinese": "zh",
    "th": "th",
    "thai": "th",
    "default": "th",
}

_current_language = _LANGUAGE_ALIASES.get(
    os.getenv("prompt_language", "th").strip().lower(),
    "th",
)


def set_prompt_language(language: str) -> None:
    """Set runtime prompt language."""
    global _current_language
    normalized = _LANGUAGE_ALIASES.get(language.strip().lower())
    if normalized is None:
        raise ValueError("prompt_language must be one of: en, english, zh, cn, chinese, th, thai")
    _current_language = normalized


def get_prompt(name: str, language: Optional[str] = None) -> str:
    """Return a prompt by name for the configured language."""
    selected_language = _current_language
    if language is not None:
        selected_language = _LANGUAGE_ALIASES.get(language.strip().lower())
        if selected_language is None:
            raise ValueError("prompt_language must be one of: en, english, zh, cn, chinese, th, thai")
    try:
        return _PROMPTS[selected_language][name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt: {name}") from exc

__all__ = [
    "set_prompt_language",
    "get_prompt",
    # Preprocess
    "GET_FEATURES_PROMPT",
    "GET_FEATURES_PROMPT_ZH",
    "GET_PROCUREMENT_FEATURES_PROMPT",
    "CASE_SEG_PROMPT",
    "CASE_SEG_PROMPT_ZH",
    "PRE_JUDGE_PROMPT",
    "PRE_JUDGE_PROMPT_ZH",
    # Judge
    "JUDGE_LAW_PROMPT",
    "JUDGE_LAW_PROMPT_ZH",
    "JUDGE_LAW_PROMPT0",
    "JUDGE_LAW_PROMPT0_ZH",
    "JUDGE_LAW_PROMPT1",
    "JUDGE_LAW_PROMPT1_ZH",
    "JUDGE_CRIME_PROMPT",
    "JUDGE_CRIME_PROMPT_ZH",
    "JUDGE_CRIME_ALL_PROMPT",
    "JUDGE_CRIME_ALL_PROMPT_ZH",
    "ANSWER_LEGAL_QA_PROMPT",
    # Retrieval
    "RETRIEVE_LAW_PROMPT",
    "RETRIEVE_LAW_PROMPT_ZH",
    # Graph
    "SUMMARIZE_TEXTS_PROMPT",
    "SUMMARIZE_TEXTS_PROMPT_ZH",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE_ZH",
    "RERANK_PROMPT_TEMPLATE",
    "RERANK_PROMPT_TEMPLATE_ZH",
]

