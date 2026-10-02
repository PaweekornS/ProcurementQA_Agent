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
        self._console_lock = threading.Lock()
        self.verbose = os.getenv("VERBOSE_AGENT_LOG", "0").lower() in ("1", "true", "yes")

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
        """Prints a structured, user-friendly step message only when VERBOSE_AGENT_LOG=1."""
        if not self.verbose:
            return

        prefix = f"[Agentic-RAG] [Q#{case_id}] {emoji} Step {step_num}: {step_name}"
        with self._console_lock:
            print(f"{prefix}")
            if details:
                for k, v in details.items():
                    print(f"              ├─ {k}: {v}")
            if message:
                print(f"              └─ {message}")

    def log_summary(
        self,
        case_id: Any,
        status: str,
        retries: int,
        chunks_count: int,
        tools_used: List[str],
        duration: Optional[float] = None
    ):
        """Prints a concise single-line summary for the query execution."""
        tools_str = ", ".join(tools_used) if tools_used else "None"
        time_str = f" ({duration:.2f}s)" if duration is not None else ""
        summary_line = (
            f"[Agentic-RAG] [Q#{case_id}] {status} | Chunks: {chunks_count} | "
            f"Retries: {retries} | Tools: [{tools_str}]{time_str}"
        )
        with self._console_lock:
            print(summary_line)

    def append_trace(self, trace_record: Dict[str, Any]):
        """Persists a single case execution trace to the JSONL trace log file."""
        trace_record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._write_lock:
            self._rotate_if_needed()
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

    def _rotate_if_needed(self):
        """
        Size-based rotation (traces.jsonl -> .1 -> .2 ...) so a long-running server does not grow
        the trace file without bound. Tunable via TRACE_LOG_MAX_MB / TRACE_LOG_BACKUPS.
        Caller must hold self._write_lock.
        """
        max_bytes = int(float(os.getenv("TRACE_LOG_MAX_MB", "50")) * 1024 * 1024)
        backups = int(os.getenv("TRACE_LOG_BACKUPS", "5"))
        try:
            if os.path.getsize(self.log_path) < max_bytes:
                return
        except OSError:
            return
        for i in range(backups - 1, 0, -1):
            src = f"{self.log_path}.{i}"
            if os.path.exists(src):
                os.replace(src, f"{self.log_path}.{i + 1}")
        if backups > 0:
            os.replace(self.log_path, f"{self.log_path}.1")
        else:
            os.remove(self.log_path)
