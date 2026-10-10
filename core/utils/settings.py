# -*- coding: utf-8 -*-
"""
core/utils/settings.py

Single place that maps configuration names. Every setting has one canonical UPPER_CASE name;
older spellings (lower_case keys from the original .env, duplicate aliases) still work but log a
one-time deprecation warning so existing deployments keep running while .env files are migrated.
"""

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger("settings")

# canonical name -> legacy names still accepted (checked in order)
LEGACY_ALIASES: Dict[str, Tuple[str, ...]] = {
    # LLM
    "MODEL_NAME": ("model_name",),
    "OPENROUTER_API_KEY": ("api_key", "OPENAI_API_KEY", "﻿OPENROUTER_API_KEY"),
    "LLM_BASE_URL": ("base_url", "OPENROUTER_BASE_URL"),
    "LLM_TEMPERATURE": ("temperature",),
    "LLM_MAX_TOKENS": ("max_length",),
    "PROMPT_LANGUAGE": ("prompt_language",),
    "DEVICE": ("device",),
    # Embedding
    "EMBEDDING_PROVIDER": ("embedding_provider",),
    "EMBEDDING_API_URL": ("embedding_api_url",),
    "EMBEDDING_MODEL": ("embedding_model",),
    "TOKENMIND_API_KEY": ("tokenmind_api_key",),
    "TOKENMIND_BASE_URL": ("tokenmind_base_url",),
    "TOKENMIND_EMBEDDING_MODEL": ("tokenmind_embedding_model",),
    # Reranker
    "RERANKER_ENABLED": ("enable_reranker", "reranker_enabled"),
    "RERANKER_PROVIDER": ("reranker_provider",),
    "RERANKER_MODEL": ("reranker_model",),
    "RERANKER_DEVICE": ("reranker_device",),
    "RERANKER_THRESHOLD": ("reranker_threshold",),
    "RERANK_POOL_SIZE": ("rerank_pool_size",),
    # Retrieval
    "RRF_K": ("rrf_k",),
    # Corpus & datasets
    "DATASETS_PATH": ("datasets_path",),
    "OUTPUT_DIR": ("output_dir",),
    # Serving
    "EXPOSE_INTERNAL_TOOLS": ("expose_internal_tools",),
}

_warned = set()


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting by its canonical name, falling back to legacy spellings, then `default`."""
    value = os.environ.get(name)
    if value not in (None, ""):
        return value
    for legacy in LEGACY_ALIASES.get(name, ()):
        value = os.environ.get(legacy)
        if value not in (None, ""):
            if legacy not in _warned:
                _warned.add(legacy)
                logger.warning("Config '%s' is deprecated; rename it to '%s' in .env", legacy.lstrip("﻿"), name)
            return value
    return default
