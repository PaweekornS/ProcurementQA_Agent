# -*- coding: utf-8 -*-
"""
core/utils/agent_logger.py

Observability and execution tracing for Agentic RAG.
Provides real-time, clean, color-coded console logs and persists structured JSONL trace logs.
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Optional


class AgentTraceLogger:
    """Thread-safe trace logger for Agentic RAG execution steps."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, log_dir: str = "./outputs/THAI", log_file: str = "agentic_traces.jsonl"):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_file)
        os.makedirs(self.log_dir, exist_ok=True)
        self._write_lock = threading.Lock()

    @classmethod
    def get_instance(cls, log_dir: str = "./outputs/THAI") -> "AgentTraceLogger":
        with cls._lock:
            if cls._instance is None or cls._instance.log_dir != log_dir:
                cls._instance = cls(log_dir=log_dir)
            return cls._instance

    def log_step(
        self,
        case_id: Any,
        step_num: int,
        step_name: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        emoji: str = "📌"
    ):
        """Prints a structured, user-friendly step message to console."""
        prefix = f"[Agentic-RAG] [Q#{case_id}] {emoji} Step {step_num}: {step_name}"
        print(f"{prefix}")
        if details:
            for k, v in details.items():
                print(f"              ├─ {k}: {v}")
        if message:
            print(f"              └─ {message}")

    def append_trace(self, trace_record: Dict[str, Any]):
        """Persists a single case execution trace to the JSONL trace log file."""
        trace_record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._write_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
