# -*- coding: utf-8 -*-
"""
Query/document embeddings (BGE-M3) with a bounded in-process cache.

Backends, in order: TokenMind API (EMBEDDING_PROVIDER=tokenmind), an HTTP embedding endpoint
(e.g. Ollama), a local SentenceTransformer. When every backend fails get_embedding returns None
and logs the reason: callers fall back to sparse search rather than ranking on a made-up vector.
"""

import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional

import requests

from core.utils.settings import env

logger = logging.getLogger(__name__)

MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "2500"))

_provider: Optional[str] = None      # None -> read EMBEDDING_PROVIDER until configure_embedding() runs
_http_url = "http://localhost:11434/api/embed"
_model = "BAAI/bge-m3"
_tokenmind_api_key: Optional[str] = None
_tokenmind_base_url: Optional[str] = None
_tokenmind_model: Optional[str] = None
_http_available: Optional[bool] = None
_local_embedder = None
_local_lock = threading.Lock()


class _LRUCache:
    """Bounded, thread-safe text -> vector cache."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: "OrderedDict[str, list]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


_cache = _LRUCache(int(os.getenv("EMBEDDING_CACHE_SIZE", "5000")))


def configure_embedding(api_url=None, model=None, provider=None, tokenmind_api_key=None,
                        tokenmind_base_url=None, tokenmind_model=None):
    global _provider, _http_url, _model, _tokenmind_api_key, _tokenmind_base_url, _tokenmind_model
    global _http_available, _local_embedder
    if provider:
        _provider = provider.lower()
    if api_url:
        _http_url = api_url
    if model and model != _model:
        _model, _local_embedder, _http_available = model, None, None
    if tokenmind_api_key:
        _tokenmind_api_key = tokenmind_api_key
    if tokenmind_base_url:
        _tokenmind_base_url = tokenmind_base_url.rstrip("/")
    if tokenmind_model:
        _tokenmind_model = tokenmind_model


def _tokenmind_settings():
    return (
        _tokenmind_api_key or env("TOKENMIND_API_KEY"),
        (_tokenmind_base_url or env("TOKENMIND_BASE_URL") or "https://tokenmind.abdul.in.th/v1").rstrip("/"),
        _tokenmind_model or env("TOKENMIND_EMBEDDING_MODEL") or "BAAI/bge-m3",
    )


def _tokenmind(texts: List[str], timeout: float) -> List[List[float]]:
    api_key, base_url, model = _tokenmind_settings()
    resp = requests.post(
        f"{base_url}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": [t[:MAX_CHARS] for t in texts]},
        timeout=timeout,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def get_embedding(text: str) -> Optional[List[float]]:
    global _http_available, _local_embedder
    if not text:
        return None
    cached = _cache.get(text)
    if cached is not None:
        return cached

    vec = None
    if (_provider or env("EMBEDDING_PROVIDER", "local")).lower() == "tokenmind" and _tokenmind_settings()[0]:
        try:
            vec = _tokenmind([text], timeout=15.0)[0]
        except Exception as e:
            logger.warning("TokenMind embedding failed: %s", e)

    if vec is None and _http_available is not False and _http_url:
        try:
            resp = requests.post(_http_url, json={"model": _model, "input": text[:MAX_CHARS]}, timeout=2.0)
            resp.raise_for_status()
            data = resp.json().get("embeddings") or resp.json().get("data")
            first = data[0]
            vec = first["embedding"] if isinstance(first, dict) else first
            _http_available = True
        except Exception:
            _http_available = False

    if vec is None:
        if _local_embedder is None:
            with _local_lock:
                if _local_embedder is None:
                    try:
                        import torch
                        from sentence_transformers import SentenceTransformer
                        device = "cuda:0" if torch.cuda.is_available() else "cpu"
                        _local_embedder = SentenceTransformer(env("EMBEDDING_MODEL") or _model, device=device)
                    except Exception as e:
                        logger.warning("Local embedder unavailable: %s", e)
                        _local_embedder = False
        if _local_embedder:
            vec = _local_embedder.encode(text[:MAX_CHARS], normalize_embeddings=True).tolist()

    if vec is None:
        logger.error("No embedding backend produced a vector; dense retrieval is skipped for this text")
        return None
    _cache.put(text, vec)
    return vec


def batch_embed_tokenmind(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a corpus through TokenMind. Raises instead of padding with placeholders, so a partial
    failure never ends up in a cached index."""
    from tqdm import tqdm
    out: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", disable=len(texts) <= batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                out.extend(_tokenmind(batch, timeout=60.0))
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"TokenMind embedding failed at offset {i}: {e}") from e
                logger.warning("TokenMind batch at offset %d failed (%s); retrying", i, e)
    return out
