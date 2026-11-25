"""LLM provider abstraction: ``generate(messages) -> str`` for Groq, Gemini, OpenAI and Ollama.

Messages use the chat format ``[{"role": "system"|"user"|"assistant", "content": str}, ...]``.
Provider and model come from ``config.yaml`` (``llm``) or are passed explicitly (UI selector).
API keys are read from the environment (``.env``): GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY.
Calls are retried with exponential backoff; temperature defaults to 0.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable

from src.config import API_KEY_ENV, get_api_key, load_config

logger = logging.getLogger(__name__)

Messages = list[dict[str, str]]
PROVIDERS = ("groq", "gemini", "openai", "ollama")


class LLMUnavailable(RuntimeError):
    """Raised when a provider cannot be used (missing key, package or server)."""


class RateLimitExhausted(RuntimeError):
    """Raised when the provider asks to wait longer than ``max_rate_wait_s`` (e.g. a daily quota)."""


class LLM:
    """A chat LLM behind a single ``generate`` method."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        reasoning_effort: str | None = None,
        classify_max_tokens: int = 512,
        ollama_num_ctx: int = 8192,
    ) -> None:
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider {provider!r}; choose one of {PROVIDERS}")
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.reasoning_effort = reasoning_effort
        self.classify_max_tokens = classify_max_tokens
        # Ollama's default context window (2k-4k tokens, depending on the version) is shorter than
        # the prompt (6 passages of up to 480 tokens): the start of the prompt would be cut silently.
        self.ollama_num_ctx = ollama_num_ctx
        # 429 handling: wait as long as the provider asks (up to max_rate_wait_s, max_rate_waits times);
        # a longer wait (daily limit) raises RateLimitExhausted.
        self.max_rate_wait_s = 120.0
        self.last_wait_s = 0.0
        self.max_rate_waits = 30
        self._client: Any = None
        self._call: Callable[[Messages, float, int], str] = getattr(self, f"_call_{provider}")

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any] | None = None, provider: str | None = None, model: str | None = None
    ) -> "LLM":
        cfg = cfg or load_config()
        c = cfg["llm"]
        return cls(
            provider=provider or c["provider"],
            model=model or c["model"],
            temperature=c.get("temperature", 0.0),
            max_tokens=c.get("max_tokens", 1024),
            seed=cfg.get("seed"),
            max_retries=c.get("max_retries", 3),
            backoff_s=c.get("backoff_s", 2.0),
            reasoning_effort=c.get("reasoning_effort"),
            classify_max_tokens=c.get("classify_max_tokens", 512),
            ollama_num_ctx=c.get("ollama_num_ctx", 8192),
        )

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------------ public API

    def check(self) -> None:
        """Raise :class:`LLMUnavailable` if the provider obviously cannot be used.

        API providers: the key must be set (no network call). Ollama: the local server must answer.
        """
        if self.provider in API_KEY_ENV and not get_api_key(self.provider):
            raise LLMUnavailable(
                f"{API_KEY_ENV[self.provider]} is not set. Add it to the .env file (see .env.example)."
            )
        if self.provider == "ollama":
            import requests

            host = ollama_host()
            try:
                requests.get(f"{host}/api/tags", timeout=2).raise_for_status()
            except requests.RequestException as exc:
                raise LLMUnavailable(
                    f"the Ollama server at {host} is not running. Start Ollama, then `ollama pull {self.model}`."
                ) from exc

    def generate(self, messages: Messages, temperature: float | None = None, max_tokens: int | None = None) -> str:
        """Return the model's reply. Retries with exponential backoff on transient errors."""
        self.check()
        temperature = self.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.max_tokens
        last_exc: Exception | None = None
        attempt, rate_waits = 0, 0
        self.last_wait_s = 0.0  # time spent waiting for rate limits in this call (excluded from latency)
        while attempt < self.max_retries:
            attempt += 1
            try:
                return self._call(messages, temperature, max_tokens).strip()
            except LLMUnavailable:
                raise
            except Exception as exc:
                if _status(exc) == 429:
                    wait = retry_after_seconds(exc)
                    if wait is not None and wait > self.max_rate_wait_s:  # e.g. daily token limit
                        raise RateLimitExhausted(f"{self.name}: rate limit, retry in {wait:.0f} s: {exc}") from exc
                    if rate_waits < self.max_rate_waits:  # per-minute limit: wait as asked, do not count it
                        rate_waits += 1
                        attempt -= 1
                        wait = (wait if wait is not None else self.backoff_s * 2 ** min(rate_waits, 5)) + 0.5
                        logger.info("%s rate limited; waiting %.1f s", self.name, wait)
                        time.sleep(wait)
                        self.last_wait_s += wait
                        continue
                if not _is_transient(exc):  # wrong model name, bad key, bad request: retrying will not help
                    raise RuntimeError(f"{self.name} call failed: {exc}") from exc
                last_exc = exc
                if attempt == self.max_retries:
                    break
                wait = self.backoff_s * 2 ** (attempt - 1)
                logger.warning("%s call failed (%s); retry %d/%d in %.0f s", self.name, exc, attempt,
                               self.max_retries - 1, wait)
                time.sleep(wait)
        raise RuntimeError(f"{self.name} failed after {self.max_retries} attempts: {last_exc}") from last_exc

    def classify(self, prompt: str) -> str:
        """Short single-turn completion (used by the router on ties)."""
        return self.generate([{"role": "user", "content": prompt}], temperature=0.0,
                             max_tokens=self.classify_max_tokens)

    # ------------------------------------------------------------------ providers

    def _extra_body(self) -> dict[str, Any] | None:
        """``reasoning_effort`` for reasoning models only (other models reject the parameter)."""
        if self.reasoning_effort and is_reasoning_model(self.model):
            return {"reasoning_effort": self.reasoning_effort}
        return None

    def _call_groq(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=get_api_key("groq"), max_retries=0)
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=self.seed,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def _call_openai(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=get_api_key("openai"), max_retries=0)
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, seed=self.seed,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def _call_gemini(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        import google.generativeai as genai

        if self._client is None:
            genai.configure(api_key=get_api_key("gemini"))
            self._client = genai
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        history = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
            for m in messages
            if m["role"] != "system"
        ]
        model = genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(
            history,
            generation_config=genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens),
        )
        return resp.text or ""

    def _call_ollama(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        import ollama

        if self._client is None:
            self._client = ollama.Client(host=ollama_host())
        try:
            resp = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens, "seed": self.seed or 0,
                         "num_ctx": self.ollama_num_ctx},
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:  # model not pulled
                raise LLMUnavailable(f"Ollama model {self.model!r} is not installed: `ollama pull {self.model}`") from exc
            raise
        except Exception as exc:
            if "connect" in type(exc).__name__.lower() or isinstance(exc, ConnectionError):
                raise LLMUnavailable(f"Ollama server not reachable at {ollama_host()}: {exc}") from exc
            raise
        return resp["message"]["content"] or ""


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def is_reasoning_model(model: str) -> bool:
    """Models that spend hidden reasoning tokens before answering."""
    name = model.lower().split("/")[-1]
    return name.startswith(("gpt-oss", "o1", "o3", "o4", "gpt-5", "qwen3", "deepseek-r1"))


def _status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient(exc: Exception) -> bool:
    """Network errors, timeouts, rate limits (429) and server errors (5xx) are worth retrying."""
    status = _status(exc)
    if status is not None:
        return status == 429 or status >= 500
    return True


_DURATION = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)(ms|s))?", re.I)


def retry_after_seconds(exc: Exception) -> float | None:
    """Wait requested by a 429: ``retry-after`` header, else "Please try again in 1m2.5s" in the message."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    m = _DURATION.search(str(exc))
    if not m or not any(m.groups()[:3]):
        return None
    h, mnt, num, unit = m.groups()
    secs = int(h or 0) * 3600 + int(mnt or 0) * 60
    if num:
        secs += float(num) / (1000 if unit and unit.lower() == "ms" else 1)
    return float(secs)


def get_llm(provider: str | None = None, model: str | None = None, cfg: dict[str, Any] | None = None) -> LLM:
    """Build an :class:`LLM` from the config, optionally overriding provider/model."""
    return LLM.from_config(cfg, provider=provider, model=model)
