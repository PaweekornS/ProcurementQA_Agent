# -*- coding: utf-8 -*-
"""
core/crag/jev_filter.py

Retrieval Evaluator & Chunk Filter powered by TypeSafe Jev 1.13 on OpenRouter.
Acts as the Corrective Gatekeeper in CRAG:
- Evaluates retrieved statutory chunks against the user inquiry.
- Discards irrelevant or distracting laws before passing context to the Synthesizer.
- Prevents Context Poisoning and Hallucination.
"""

import os
import time
import requests
import concurrent.futures
from typing import List, Dict, Any, Optional


class JevChunkFilter:
    """
    Ultra-fast (<500ms) statutory chunk relevance evaluator using TypeSafe Jev 1.13.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        threshold: float = 0.45,
        timeout: float = 8.0,
        enabled: Optional[bool] = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.threshold = float(os.getenv("JEV_FILTER_THRESHOLD", str(threshold)))
        self.timeout = timeout

        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv("USE_JEV_FILTER", "true").strip().lower() in ("true", "1", "yes")

        self.endpoint = "https://openrouter.ai/api/alpha/decisions"
        self.model = os.getenv("JEV_MODEL", "typesafe/jev-1.13")

    def evaluate_chunk(self, question: str, chunk: Dict[str, Any]) -> float:
        """
        Calls Jev 1.13 to evaluate a single chunk's relevance to the inquiry.
        Returns a probability score between 0.0 and 1.0.
        """
        if not self.api_key:
            return 0.5

        entry = str(chunk.get("law_entry", chunk.get("entry", "")))
        snippet = str(chunk.get("description", chunk.get("text", chunk.get("snippet", chunk.get("content_thai", "")))))
        state_text = f"Inquiry: {question}\n\nCandidate Statutory Text: {entry} {snippet}"

        payload = {
            "model": self.model,
            "state": state_text[:2500],
            "questions": {
                "is_directly_relevant": {
                    "type": "noul",
                    "instructions": (
                        "Does this statutory text contain the specific rules, conditions, "
                        "thresholds, or authority directly applicable to the scenario in the inquiry?"
                    )
                }
            }
        }

        try:
            resp = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                res_json = resp.json()
                noul_val = res_json.get("answers", {}).get("is_directly_relevant", {}).get("noul", 0.0)
                return float(noul_val)
            else:
                return 0.5
        except Exception:
            return 0.5

    def filter_chunks(
        self,
        question: str,
        candidate_laws: List[Dict[str, Any]],
        max_keep: int = 6,
        min_keep: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Evaluates candidate chunks concurrently and filters out those below threshold.
        Guarantees at least `min_keep` chunks are preserved to avoid context starvation.
        """
        if not self.enabled or not candidate_laws or not self.api_key:
            return candidate_laws[:max_keep]

        t0 = time.time()
        chunks_to_eval = candidate_laws[:10]  # Focus evaluation on top candidates

        scored_chunks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks_to_eval), 8)) as executor:
            future_to_chunk = {
                executor.submit(self.evaluate_chunk, question, chunk): chunk
                for chunk in chunks_to_eval
            }
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    score = future.result()
                except Exception:
                    score = 0.5
                c_copy = dict(chunk)
                c_copy["jev_score"] = round(score, 4)
                scored_chunks.append(c_copy)

        # Sort by Jev score descending
        scored_chunks.sort(key=lambda x: x.get("jev_score", 0.0), reverse=True)

        # Filter chunks that meet the relevance threshold
        kept_chunks = [c for c in scored_chunks if c.get("jev_score", 0.0) >= self.threshold]

        # Safety Fallback: If Jev filtered everything out, keep the top `min_keep`
        if len(kept_chunks) < min_keep and scored_chunks:
            kept_chunks = scored_chunks[:min_keep]

        dt = time.time() - t0
        # print(f"[JevGatekeeper] Filtered {len(chunks_to_eval)} -> {len(kept_chunks)} chunks in {dt:.2f}s (Threshold: {self.threshold})")

        return kept_chunks[:max_keep]
