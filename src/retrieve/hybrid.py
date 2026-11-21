"""Hybrid retrieval: dense (Chroma) + lexical (BM25) fused with Reciprocal Rank Fusion.

RRF score of a chunk = sum over the ranked lists it appears in of ``1 / (rrf_k + rank)``.
When several specs are requested, each spec is retrieved and fused separately and the per-spec
lists are interleaved, so that one spec cannot crowd out the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Sequence

from src.index.bm25_index import BM25Index
from src.index.common import Hit
from src.index.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A chunk with its scores and ranks at every retrieval stage (``None`` = not in that list)."""

    chunk_id: str
    chunk: dict[str, Any]
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_score: float = 0.0
    fused_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    final_rank: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, with_text: bool = True) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("chunk", "extra")}
        chunk = self.chunk if with_text else {k: v for k, v in self.chunk.items() if not k.startswith("text")}
        d["chunk"] = chunk
        return d


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], rrf_k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of ranked id lists. Returns ``(id, score)`` sorted by score desc.

    Ties are broken by first appearance (list order, then rank) for determinism.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            order.setdefault(cid, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def interleave(lists: Sequence[Sequence[Candidate]]) -> list[Candidate]:
    """Round-robin merge of ranked lists (first of each, then second of each, ...), deduplicated."""
    out: list[Candidate] = []
    seen: set[str] = set()
    for group in zip_longest(*lists):
        for cand in group:
            if cand is not None and cand.chunk_id not in seen:
                out.append(cand)
                seen.add(cand.chunk_id)
    return out


class HybridRetriever:
    """Dense + BM25 retrieval with RRF fusion and optional per-spec retrieval."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Index,
        dense_k: int = 30,
        bm25_k: int = 30,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k

    @classmethod
    def from_config(cls, cfg: dict[str, Any], vector_store: VectorStore, bm25: BM25Index) -> "HybridRetriever":
        r = cfg["retrieval"]
        return cls(vector_store, bm25, r["dense_k"], r["bm25_k"], r["rrf_k"])

    def _single(self, query: str, specs: Sequence[str] | None, use_hybrid: bool) -> list[Candidate]:
        dense: list[Hit] = self.vector_store.query(query, k=self.dense_k, specs=specs)
        cands: dict[str, Candidate] = {}
        for h in dense:
            cands[h.chunk_id] = Candidate(h.chunk_id, h.chunk, dense_rank=h.rank, dense_score=h.score)
        if not use_hybrid:
            out = sorted(cands.values(), key=lambda c: c.dense_rank or 0)
            for c in out:
                c.fused_score = 1.0 / (self.rrf_k + (c.dense_rank or 0))
            return out
        lexical: list[Hit] = self.bm25.query(query, k=self.bm25_k, specs=specs)
        for h in lexical:
            c = cands.get(h.chunk_id)
            if c is None:
                c = cands[h.chunk_id] = Candidate(h.chunk_id, h.chunk)
            c.bm25_rank, c.bm25_score = h.rank, h.score
        fused = rrf_fuse([[h.chunk_id for h in dense], [h.chunk_id for h in lexical]], self.rrf_k)
        out = []
        for cid, score in fused:
            cands[cid].fused_score = score
            out.append(cands[cid])
        return out

    def retrieve(
        self, query: str, specs: Sequence[str] | None = None, use_hybrid: bool = True
    ) -> list[Candidate]:
        """Return fused candidates in fused order, with ``fused_rank`` set.

        Args:
            query: The user query (already expanded if needed).
            specs: Restrict to these specs. With several specs, retrieval is done per spec and the
                per-spec lists are interleaved. ``None`` searches the whole corpus at once.
            use_hybrid: If False, dense retrieval only.
        """
        if specs and len(specs) > 1:
            per_spec = [self._single(query, [s], use_hybrid) for s in specs]
            out = interleave(per_spec)
        else:
            out = self._single(query, specs, use_hybrid)
        for rank, c in enumerate(out, start=1):
            c.fused_rank = rank
        return out
