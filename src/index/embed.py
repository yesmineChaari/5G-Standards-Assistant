"""Sentence embeddings with an English bge model (sentence-transformers).

Documents are embedded as-is (``text_for_index``, i.e. header line + chunk text).
Queries are prefixed with the bge query instruction from ``config.yaml``.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> str:
    """Return ``"cuda"`` when requested/available, otherwise ``"cpu"``."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        return "cpu"
    return device


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Embedder:
    """Wrapper around a sentence-transformers model with bge query handling.

    The model is loaded lazily on first use.
    """

    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        normalize: bool = True,
        batch_size: int = 32,
        max_seq_length: int | None = 512,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = resolve_device(device)
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Embedder":
        e = cfg["embeddings"]
        return cls(
            model_name=e["model"],
            query_instruction=e.get("query_instruction", ""),
            normalize=e.get("normalize", True),
            batch_size=e.get("batch_size", 32),
            max_seq_length=e.get("max_seq_length", 512),
            device=e.get("device", "auto"),
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def count_truncated(self, texts: Sequence[str]) -> int:
        """Number of texts longer than the model's max sequence length (they get truncated)."""
        tok = self.model.tokenizer
        limit = self.model.max_seq_length
        return sum(1 for t in texts if len(tok(t, add_special_tokens=True)["input_ids"]) > limit)

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Embed passages (no instruction prefix). Returns a float32 array (n, dim)."""
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query, prefixed with the bge query instruction."""
        vec = self.model.encode(
            [self.query_instruction + query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)
