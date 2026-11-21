"""Unit tests for RRF fusion, per-spec interleaving and reranker scoring (no models loaded)."""

from __future__ import annotations

import math

import pytest

from src.index.common import Hit
from src.retrieve.hybrid import Candidate, HybridRetriever, interleave, rrf_fuse
from src.retrieve.reranker import Reranker, sigmoid


def test_rrf_scores_and_order() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], rrf_k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert [cid for cid, _ in fused][0] == "b"  # in both lists
    assert set(scores) == {"a", "b", "c", "d"}


def test_rrf_tie_break_is_deterministic() -> None:
    fused = rrf_fuse([["x"], ["y"]], rrf_k=60)
    assert [cid for cid, _ in fused] == ["x", "y"]


def _cand(cid: str, spec: str = "TS 24.501") -> Candidate:
    return Candidate(cid, {"chunk_id": cid, "spec": spec, "section": "1", "title": "t", "text": cid, "text_for_index": cid})


def test_interleave_balances_specs_and_dedups() -> None:
    a = [_cand("a1"), _cand("a2"), _cand("a3")]
    b = [_cand("b1", "TS 23.501"), _cand("a2")]
    merged = [c.chunk_id for c in interleave([a, b])]
    assert merged == ["a1", "b1", "a2", "a3"]


class _FakeStore:
    def __init__(self, ids_by_spec: dict[str, list[str]]) -> None:
        self.ids_by_spec = ids_by_spec

    def _hits(self, specs):
        specs = specs or list(self.ids_by_spec)
        ids = [cid for s in specs for cid in self.ids_by_spec[s]]
        return [Hit(cid, 1.0 / (i + 1), i + 1, _cand(cid).chunk) for i, cid in enumerate(ids)]

    def query(self, query, k=10, specs=None):
        return self._hits(specs)[:k]


class _FakeBM25(_FakeStore):
    def query(self, query, k=10, specs=None):
        return list(reversed(self._hits(specs)))[:k]


def test_hybrid_retriever_dense_only_and_fused() -> None:
    store = _FakeStore({"TS 24.501": ["a", "b", "c"], "TS 23.501": ["x", "y"]})
    bm25 = _FakeBM25({"TS 24.501": ["a", "b", "c"], "TS 23.501": ["x", "y"]})
    retriever = HybridRetriever(store, bm25, dense_k=5, bm25_k=5, rrf_k=60)

    dense = retriever.retrieve("q", use_hybrid=False)
    assert [c.chunk_id for c in dense] == ["a", "b", "c", "x", "y"]
    assert all(c.bm25_rank is None for c in dense)
    assert [c.fused_rank for c in dense] == [1, 2, 3, 4, 5]

    fused = retriever.retrieve("q", use_hybrid=True)
    assert all(c.dense_rank and c.bm25_rank for c in fused)
    scores = [c.fused_score for c in fused]
    assert scores == sorted(scores, reverse=True)

    per_spec = retriever.retrieve("q", specs=["TS 24.501", "TS 23.501"], use_hybrid=True)
    specs_order = [c.chunk_id[0] in "abc" for c in per_spec]  # True = TS 24.501 chunk
    assert specs_order[:4] == [True, False, True, False]  # alternates between the two specs
    assert len(per_spec) == 5


def test_sigmoid_range() -> None:
    assert sigmoid(0) == 0.5
    assert 0 < sigmoid(-50) < 1e-20 and sigmoid(50) == pytest.approx(1.0)
    assert sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2.0)))


def test_reranker_sorts_by_score(monkeypatch) -> None:
    reranker = Reranker("dummy", device="cpu", use_fallback_on_cpu=False)
    monkeypatch.setattr(reranker, "score", lambda q, texts: [0.1, 0.9, 0.5])
    cands = [_cand("a"), _cand("b"), _cand("c")]
    for i, c in enumerate(cands, start=1):
        c.fused_rank = i
    ranked = reranker.rerank("q", cands)
    assert [c.chunk_id for c in ranked] == ["b", "c", "a"]
    assert [c.rerank_rank for c in ranked] == [1, 2, 3]


def test_reranker_uses_fallback_on_cpu() -> None:
    r = Reranker("big-model", fallback_model="small-model", use_fallback_on_cpu=True, device="cpu")
    assert r.model_name == "small-model"


def test_balance_by_spec_interleaves_reranked_results() -> None:
    from src.retrieve.pipeline import balance_by_spec

    def c(cid: str, spec: str, score: float) -> Candidate:
        cand = Candidate(cid, {"chunk_id": cid, "spec": spec})
        cand.rerank_score = score
        return cand

    ranked = [c("n1", "TS 24.501", 0.99), c("n2", "TS 24.501", 0.98), c("n3", "TS 24.501", 0.97),
              c("a1", "TS 23.501", 0.90), c("a2", "TS 23.501", 0.50)]
    out = [x.chunk_id for x in balance_by_spec(ranked, ["TS 23.501", "TS 24.501"])]
    assert out == ["n1", "a1", "n2", "a2", "n3"]  # best spec first, then alternate
