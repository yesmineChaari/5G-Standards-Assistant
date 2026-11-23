"""Recorded LLM calls: identical (model, messages, settings) calls are answered from disk.

The recording ``eval/cache/generation/llm_calls.jsonl`` is versioned with the repository, so the
evaluation can be replayed without an API key or quota. Retrieval and prompts are always recomputed;
a call is answered from the recording only if its prompt is byte-identical to a recorded one, so a
complete replay also shows that chunking, indexing and retrieval reproduce exactly. A call that is
not recorded needs the provider: without an API key it raises :class:`LLMUnavailable`.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.config import resolve_path
from src.generate.llm import LLM, LLMUnavailable

CALLS_FILE = "eval/cache/generation/llm_calls.jsonl"


def _hash(obj: Any, n: int = 24) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:n]


class CachedLLM:
    """Wraps an :class:`LLM`; identical (model, messages, settings) calls are answered from disk."""

    def __init__(self, llm: LLM, path: Path | None = None) -> None:
        self.llm = llm
        self.path = path or resolve_path(CALLS_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec
        self.hits = self.misses = 0
        self.last_latency = 0.0
        self.replay_only = False  # set by check() when the provider is unavailable

    @property
    def name(self) -> str:
        return self.llm.name

    @property
    def classify_max_tokens(self) -> int:
        return self.llm.classify_max_tokens

    def check(self) -> None:
        """Raise :class:`LLMUnavailable` only if the provider is unavailable *and* nothing is recorded."""
        try:
            self.llm.check()
            self.replay_only = False
        except LLMUnavailable:
            if not self.cache:
                raise
            self.replay_only = True

    def generate(self, messages, temperature=None, max_tokens=None) -> str:
        key = _hash([self.llm.name, self.llm.reasoning_effort, messages, temperature, max_tokens,
                     self.llm.temperature, self.llm.max_tokens])
        if key in self.cache:
            self.hits += 1
            self.last_latency = self.cache[key]["latency_s"]
            return self.cache[key]["text"]
        t0 = time.perf_counter()
        text = self.llm.generate(messages, temperature=temperature, max_tokens=max_tokens)
        # model time only: waiting for the provider's rate limit is not latency of the system
        self.last_latency = time.perf_counter() - t0 - getattr(self.llm, "last_wait_s", 0.0)
        self.misses += 1
        rec = {"key": key, "text": text, "latency_s": self.last_latency, "model": self.llm.name}
        self.cache[key] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return text

    def classify(self, prompt: str) -> str:
        return self.generate([{"role": "user", "content": prompt}], temperature=0.0,
                             max_tokens=self.llm.classify_max_tokens)
