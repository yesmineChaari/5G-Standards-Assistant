"""Configuration loading.

Every module reads its parameters from ``config.yaml`` through :func:`load_config`.
API keys and other secrets come from environment variables (optionally a ``.env`` file).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

REQUIRED_SECTIONS: tuple[str, ...] = (
    "corpus",
    "paths",
    "chunking",
    "embeddings",
    "vector_store",
    "bm25",
    "retrieval",
    "reranker",
    "llm",
    "refusal_message",
    "seed",
)

# Environment variable holding the API key for each LLM provider.
API_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _validate(cfg: dict[str, Any]) -> None:
    """Raise ``ValueError`` if mandatory sections or fields are missing."""
    missing = [key for key in REQUIRED_SECTIONS if key not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing sections: {missing}")
    specs = cfg["corpus"].get("specs") or []
    if not specs:
        raise ValueError("config.yaml: corpus.specs must list at least one specification")
    for spec in specs:
        for field in ("id", "short", "version", "release", "url", "file"):
            if field not in spec:
                raise ValueError(f"config.yaml: spec entry {spec} is missing '{field}'")


@lru_cache(maxsize=None)
def _load_cached(path_str: str) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(path_str, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path_str} does not contain a YAML mapping")
    _validate(cfg)
    logger.debug("Loaded configuration from %s", path_str)
    return cfg


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the project configuration.

    Args:
        path: Optional path to a YAML config file. Defaults to ``config.yaml``
            at the project root.

    Returns:
        A fresh copy of the configuration dictionary (callers may modify it).
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg = _load_cached(str(config_path.resolve()))
    return _deep_copy(cfg)


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def resolve_path(relative: str | Path) -> Path:
    """Return an absolute path for a path given relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_api_key(provider: str) -> str | None:
    """Return the API key for an LLM provider from the environment, if set."""
    load_dotenv(PROJECT_ROOT / ".env")
    env_name = API_KEY_ENV.get(provider.lower())
    return os.getenv(env_name) if env_name else None


def get_spec(cfg: dict[str, Any], spec_id: str) -> dict[str, Any]:
    """Return the corpus entry for a spec id such as ``"TS 24.501"``."""
    for spec in cfg["corpus"]["specs"]:
        if spec["id"] == spec_id or spec["short"] == spec_id:
            return spec
    raise KeyError(f"Unknown spec: {spec_id}")
