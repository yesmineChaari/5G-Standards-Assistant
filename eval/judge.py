"""RAGAS faithfulness and answer correctness with a cached LLM judge.

The judge is reached through an OpenAI-compatible endpoint (Groq, OpenAI or a local Ollama server)
via ``langchain_openai.ChatOpenAI``; answer correctness also needs embeddings, for which the
project's own embedding model (bge-base, local) is used. Scores are cached in
``eval/cache/ragas.jsonl`` keyed by judge, metric and the exact inputs, so re-runs are free.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Sequence

from src.config import get_api_key, load_config, resolve_path

logger = logging.getLogger(__name__)

CACHE_FILE = "eval/cache/ragas.jsonl"
BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini's OpenAI-compatible API
    "openai": None,  # library default
    "ollama": os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1",
}


class _LocalEmbeddings:
    """LangChain ``Embeddings`` interface over the project's sentence-transformers model."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from src.index.embed import Embedder

        self.embedder = Embedder.from_config(cfg)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self.embedder.embed_documents(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        # Plain similarity between two answers: no query instruction prefix.
        return [float(x) for x in self.embedder.embed_documents([text])[0]]

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class RagasJudge:
    """Faithfulness (answer vs contexts) and answer correctness (answer vs reference)."""

    def __init__(self, cfg: dict[str, Any] | None = None, provider: str | None = None,
                 model: str | None = None, cache_only: bool = False) -> None:
        self.cfg = cfg or load_config()
        ev = self.cfg.get("evaluation", {})
        self.provider = provider or ev.get("judge_provider", "groq")
        self.model = model or ev.get("judge_model", "openai/gpt-oss-20b")
        self.reasoning_effort = ev.get("judge_reasoning_effort") if provider is None else None
        self.cache_only = cache_only  # read cached scores only (e.g. a previous judge, for comparison)
        self.name = f"{self.provider}:{self.model}"
        self.cache_path = resolve_path(CACHE_FILE)
        self.cache = self._load_cache()
        self._metrics: dict[str, Any] | None = None
        self.exhausted = False  # set when the judge's daily quota is reached: no further calls
        self.missing = 0  # scores that could not be computed in this run

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> dict[str, float]:
        if not self.cache_path.exists():
            return {}
        out = {}
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec["score"]
        return out

    def _save(self, key: str, score: float, meta: dict[str, Any]) -> None:
        self.cache[key] = score
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "score": score, **meta}, ensure_ascii=False) + "\n")

    def _key(self, metric: str, question: str, answer: str, contexts: Sequence[str], reference: str) -> str:
        raw = json.dumps([self.name, metric, question, answer, list(contexts), reference], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ------------------------------------------------------------------ ragas

    def check(self) -> None:
        if self.provider in ("groq", "openai", "gemini") and not get_api_key(self.provider):
            from src.generate.llm import LLMUnavailable

            raise LLMUnavailable(f"no API key for the judge provider {self.provider}")

    def _build(self) -> dict[str, Any]:
        from langchain_openai import ChatOpenAI
        from ragas import RunConfig
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerCorrectness, Faithfulness

        from src.generate.llm import is_reasoning_model

        extra = {"reasoning_effort": self.reasoning_effort} if (
            self.reasoning_effort and is_reasoning_model(self.model)) else None
        chat = ChatOpenAI(
            model=self.model, base_url=BASE_URLS.get(self.provider), temperature=0.0, max_tokens=4096,
            api_key=get_api_key(self.provider) or "ollama", max_retries=8, timeout=180, extra_body=extra,
        )
        llm = LangchainLLMWrapper(chat)
        emb = LangchainEmbeddingsWrapper(_LocalEmbeddings(self.cfg))
        run_config = RunConfig(max_workers=1, max_retries=8, max_wait=90, timeout=300, seed=self.cfg.get("seed", 42))
        metrics = {"faithfulness": Faithfulness(llm=llm), "answer_correctness": AnswerCorrectness(llm=llm, embeddings=emb)}
        for m in metrics.values():
            m.init(run_config)
        return metrics

    def score(self, metric: str, question: str, answer: str, contexts: Sequence[str],
              reference: str = "") -> float | None:
        """Cached score in [0, 1], or None if the judge failed (not cached, retried on the next run)."""
        key = self._key(metric, question, answer, contexts, reference)
        if key in self.cache:
            return self.cache[key]
        if self.exhausted or self.cache_only:
            if not self.cache_only:
                self.missing += 1
            return None
        from ragas import SingleTurnSample

        if self._metrics is None:
            self._metrics = self._build()
        sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=list(contexts),
                                  reference=reference or None)
        try:
            value = asyncio.run(self._metrics[metric].single_turn_ascore(sample))
        except Exception as exc:  # rate limit exhausted, parsing failure, timeout
            self.missing += 1
            if is_daily_limit(exc):
                self.exhausted = True
                logger.error("RAGAS judge %s: daily quota reached; remaining scores skipped (re-run later)", self.name)
            else:
                logger.warning("RAGAS %s failed (%s)", metric, str(exc)[:200])
            return None
        if value is None or (isinstance(value, float) and math.isnan(value)):
            logger.warning("RAGAS %s returned no score", metric)
            return None
        value = float(value)
        self._save(key, value, {"metric": metric, "judge": self.name, "question": question[:80]})
        return value


def is_daily_limit(exc: BaseException) -> bool:
    """True for a per-day token/request quota error (Groq "tokens per day (TPD)", Gemini "...PerDay...")."""
    text = str(exc)
    low = text.lower()
    return "per day" in low or "perday" in low or "(TPD)" in text or "(RPD)" in text


def cache_path() -> Path:
    return resolve_path(CACHE_FILE)
