"""
Method 2: Session-scoped semantic retrieval memory for lower-latency chat.

This strategy does not attempt per-query model-weight paging. Instead it keeps a
small in-memory store of recent query/response turns per page session, retrieves
the most relevant turns for each new query, and returns a compact context block.
"""

from __future__ import annotations

import time
import re
import math
from dataclasses import dataclass
from collections import Counter
from threading import Lock

import psutil


@dataclass
class SessionTransferState:
    query_count: int = 0
    memory_turns: list[dict] = None
    last_retrieval_ms: int = 0
    last_retrieved_count: int = 0
    last_similarity: float = 0.0
    last_query_started_at: float = 0.0

    def __post_init__(self):
        if self.memory_turns is None:
            self.memory_turns = []


class SessionRamTransferStrategy:
    def __init__(self, max_turns_per_session: int = 40, top_k: int = 2):
        self.max_turns_per_session = max_turns_per_session
        self.top_k = top_k
        self._states: dict[str, SessionTransferState] = {}
        self._lock = Lock()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _counter_cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        common = set(a.keys()) & set(b.keys())
        num = sum(a[k] * b[k] for k in common)
        den_a = math.sqrt(sum(v * v for v in a.values()))
        den_b = math.sqrt(sum(v * v for v in b.values()))
        if den_a == 0.0 or den_b == 0.0:
            return 0.0
        return float(num / (den_a * den_b))

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        t = (text or "").strip()
        if len(t) <= limit:
            return t
        return t[: limit - 1].rstrip() + "…"

    def _get_state(self, page_session_id: str) -> SessionTransferState:
        with self._lock:
            if page_session_id not in self._states:
                self._states[page_session_id] = SessionTransferState()
            return self._states[page_session_id]

    def reset_session(self, page_session_id: str) -> None:
        with self._lock:
            self._states.pop(page_session_id, None)

    def begin_query(self, page_session_id: str, prompt_text: str) -> dict:
        state = self._get_state(page_session_id)
        state.query_count += 1
        state.last_query_started_at = time.perf_counter()
        retrieval_started = time.perf_counter()

        vm = psutil.virtual_memory()
        ram_available_mb = int(vm.available / (1024 * 1024))
        ram_percent = float(vm.percent)

        prompt_counter = Counter(self._tokenize(prompt_text))
        ranked = []
        for idx, turn in enumerate(state.memory_turns):
            sim = self._counter_cosine(prompt_counter, turn["counter"])
            recency_boost = 0.015 * (idx + 1) / max(1, len(state.memory_turns))
            score = sim + recency_boost
            ranked.append((score, sim, turn))

        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = [r[2] for r in ranked[: self.top_k] if r[1] >= 0.1]

        context_lines = []
        for i, t in enumerate(selected, 1):
            q = self._clip(t["query"], 180)
            a = self._clip(t["response"], 220)
            context_lines.append(f"{i}. Prior user: {q}\n   Prior assistant: {a}")

        retrieval_context = ""
        if context_lines:
            retrieval_context = (
                "Use the following prior session context only if relevant. "
                "Do not repeat it verbatim.\n"
                + "\n".join(context_lines)
            )

        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        state.last_retrieval_ms = retrieval_ms
        state.last_retrieved_count = len(selected)
        state.last_similarity = round(ranked[0][1], 3) if ranked else 0.0

        return {
            "page_session_id": page_session_id,
            "query_index": state.query_count,
            "retrieval_performed": len(selected) > 0,
            "retrieved_count": len(selected),
            "retrieval_ms": retrieval_ms,
            "top_similarity": state.last_similarity,
            "semantic_context": retrieval_context,
            "ram_available_mb": ram_available_mb,
            "ram_percent": round(ram_percent, 1),
            # Backward-compatible keys for existing frontend labels.
            "transfer_performed": len(selected) > 0,
            "transfer_delay_ms": retrieval_ms,
            "transfer_mode": "semantic-retrieval-memory",
            "queued_transfer_count": 0,
            "cached_context_hint": len(selected) > 0,
            "skipped_reason": "",
        }

    def end_query(self, page_session_id: str, generation_ms: int, user_query: str, assistant_response: str) -> dict:
        state = self._get_state(page_session_id)
        state.memory_turns.append(
            {
                "query": (user_query or "").strip(),
                "response": (assistant_response or "").strip(),
                "counter": Counter(self._tokenize(user_query + " " + assistant_response)),
            }
        )
        if len(state.memory_turns) > self.max_turns_per_session:
            state.memory_turns = state.memory_turns[-self.max_turns_per_session :]

        return {
            "page_session_id": page_session_id,
            "query_index": state.query_count,
            "generation_ms": generation_ms,
            "last_retrieval_ms": state.last_retrieval_ms,
            "retrieved_count": state.last_retrieved_count,
            "stored_turns": len(state.memory_turns),
            "top_similarity": state.last_similarity,
            # Backward-compatible keys.
            "last_transfer_ms": state.last_retrieval_ms,
            "queued_transfer_count": 0,
        }
