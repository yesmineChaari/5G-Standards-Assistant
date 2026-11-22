"""Exact dense search: correct order, spec filter, deterministic ties (no Chroma, no model)."""

from __future__ import annotations

import numpy as np

from src.index.vector_store import VectorStore


def _store() -> VectorStore:
    vs = VectorStore.__new__(VectorStore)  # no Chroma client, no embedder
    ids = ["a", "b", "c", "d"]
    emb = np.array([[1, 0], [0.6, 0.8], [0, 1], [1, 0]], dtype=np.float32)
    metas = [{"spec": "TS 23.501", "section": s, "chunk_id": i} for s, i in zip("1234", ids)]
    metas[2]["spec"] = "TS 24.501"
    vs._matrix = (ids, emb, metas, [f"doc {i}" for i in ids])
    return vs


def test_exact_order_and_ties() -> None:
    hits = _store()._exact_query(np.array([1.0, 0.0]), k=4, specs=None)
    assert [h.chunk_id for h in hits] == ["a", "d", "b", "c"]  # a/d tie: chunk id order
    assert [h.rank for h in hits] == [1, 2, 3, 4] and hits[0].score == 1.0


def test_exact_spec_filter() -> None:
    hits = _store()._exact_query(np.array([0.0, 2.0]), k=3, specs=["TS 24.501"])
    assert [h.chunk_id for h in hits] == ["c"]  # only one chunk of that spec
    assert hits[0].chunk["text_for_index"] == "doc c"
