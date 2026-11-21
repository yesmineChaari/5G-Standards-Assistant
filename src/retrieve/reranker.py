"""Cross-encoder reranking of fused candidates (bge reranker).

Scores are the raw cross-encoder logits passed through a sigmoid, so they lie in [0, 1]
(needed for the refusal threshold in the generation step).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

from src.index.embed import resolve_device
from src.retrieve.hybrid import Candidate

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Reranker:
    """Lazy-loaded sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str,
        fallback_model: str | None = None,
        use_fallback_on_cpu: bool = True,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.device = resolve_device(device)
        if self.device == "cpu" and use_fallback_on_cpu and fallback_model:
            logger.info("Reranker on CPU: using fallback model %s instead of %s", fallback_model, model_name)
            model_name = fallback_model
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Reranker":
        r = cfg["reranker"]
        return cls(
            model_name=r["model"],
            fallback_model=r.get("fallback_model"),
            use_fallback_on_cpu=r.get("use_fallback_on_cpu", True),
            device=r.get("device", "auto"),
            batch_size=r.get("batch_size", 8),
            max_length=r.get("max_length", 512),
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s on %s", self.model_name, self.device)
            # Identity activation: we get raw logits and apply the sigmoid ourselves.
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                default_activation_function=torch.nn.Identity(),
            )
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Sigmoid relevance scores in [0, 1] for (query, text) pairs."""
        if not texts:
            return []
        logits = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [sigmoid(float(x)) for x in logits]

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Score candidates on ``(query, text_for_index)`` and return them sorted by score desc."""
        scores = self.score(query, [c.chunk["text_for_index"] for c in candidates])
        for c, s in zip(candidates, scores):
            c.rerank_score = s
        ranked = sorted(candidates, key=lambda c: (-(c.rerank_score or 0.0), c.fused_rank or 0))
        for rank, c in enumerate(ranked, start=1):
            c.rerank_rank = rank
        return ranked
