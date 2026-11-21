"""Retrieval pipeline: single entry point ``retrieve(query, ...)``.

Usage (CLI, compares configurations side by side):
    python -m src.retrieve.pipeline "What does cause #9 mean in 5GMM?" [-k 6] [--spec "TS 24.501"]

Stages: acronym expansion -> routing by spec -> dense / hybrid retrieval (RRF, per spec when
routed to both) -> cross-encoder reranking -> top-k.
Indexes and models are loaded once per :class:`Pipeline` and reused across queries.

Query variants: the expanded query (original + acronym expansions) is used for dense and BM25
retrieval; routing and reranking use the original question.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from src.config import load_config
from src.index.bm25_index import build_bm25
from src.index.embed import set_seed
from src.index.vector_store import VectorStore
from src.retrieve.acronyms import AcronymExpander, load_acronyms
from src.retrieve.hybrid import Candidate, HybridRetriever, interleave
from src.retrieve.reranker import Reranker
from src.retrieve.router import RouteDecision, Router

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Everything the generation step and the UI need about one retrieval."""

    query: str
    routed_specs: list[str]
    route_reason: str
    flags: dict[str, Any]
    candidates: list[Candidate]  # all fused candidates (fused order), with stage scores/ranks
    results: list[Candidate]  # final top-k, in final order
    timings: dict[str, float] = field(default_factory=dict)
    expanded_query: str | None = None
    acronyms: list[str] = field(default_factory=list)
    route: RouteDecision | None = None

    @property
    def route_label(self) -> str:
        """Text for the UI, e.g. ``Routed to: TS 24.501 (keywords: 5GMM, cause #)``."""
        return f"Routed to: {' + '.join(self.routed_specs)} ({self.route_reason})"

    @property
    def top_score(self) -> float | None:
        """Best reranker score (None if the reranker was not used)."""
        if not self.results or self.results[0].rerank_score is None:
            return None
        return self.results[0].rerank_score


def balance_by_spec(ranked: list[Candidate], specs: Sequence[str]) -> list[Candidate]:
    """Interleave reranked candidates spec by spec (keeps per-spec order).

    Used when a query is routed to several specs, so that the reranker cannot let one spec crowd
    out the other in the final top-k. The spec with the best-scored chunk comes first.
    """
    per_spec = {s: [c for c in ranked if c.chunk["spec"] == s] for s in specs}
    order = sorted(
        (s for s in specs if per_spec[s]),
        key=lambda s: -(per_spec[s][0].rerank_score or 0.0),
    )
    rest = [c for c in ranked if c.chunk["spec"] not in per_spec]
    return interleave([per_spec[s] for s in order]) + rest


class Pipeline:
    """Holds the indexes and models; :meth:`retrieve` runs one query."""

    def __init__(self, cfg: dict[str, Any] | None = None, llm_classifier: Any = None) -> None:
        """Load indexes and models.

        Args:
            cfg: Configuration (default ``config.yaml``).
            llm_classifier: Optional ``callable(prompt) -> str`` used by the router on ties.
        """
        self.cfg = cfg or load_config()
        set_seed(self.cfg["seed"])
        self.all_specs = [s["id"] for s in self.cfg["corpus"]["specs"]]
        self.vector_store = VectorStore.from_config(self.cfg)
        self.bm25 = build_bm25(self.cfg)  # loads the pickle (builds it only if missing/outdated)
        self.hybrid = HybridRetriever.from_config(self.cfg, self.vector_store, self.bm25)
        self.reranker = Reranker.from_config(self.cfg)
        self.router = Router(llm_classifier)
        self.expander = AcronymExpander(load_acronyms(self.cfg))

    def retrieve(
        self,
        query: str,
        use_hybrid: bool | None = None,
        use_reranker: bool | None = None,
        use_routing: bool | None = None,
        use_acronym_expansion: bool | None = None,
        specs: Sequence[str] | None = None,
        final_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve the top chunks for a query.

        Args:
            query: User question.
            use_hybrid / use_reranker / use_routing / use_acronym_expansion: Override the config
                flags (``None`` = value from config).
            specs: Manual spec filter (bypasses routing), e.g. ``["TS 24.501"]``.
            final_k: Number of chunks to return (default ``retrieval.final_k``).
        """
        r = self.cfg["retrieval"]
        use_hybrid = r["use_hybrid"] if use_hybrid is None else use_hybrid
        use_reranker = r["use_reranker"] if use_reranker is None else use_reranker
        use_routing = r["use_routing"] if use_routing is None else use_routing
        if use_acronym_expansion is None:
            use_acronym_expansion = r.get("use_acronym_expansion", True)
        final_k = final_k or r["final_k"]
        timings: dict[str, float] = {}

        search_query, acronyms = query, []
        if use_acronym_expansion:
            search_query, acronyms = self.expander.expand(query)

        decision: RouteDecision | None = None
        if specs:
            routed, reason = list(specs), "manual spec filter"
            search_specs: list[str] | None = routed
        elif use_routing:
            t0 = time.perf_counter()
            decision = self.router.route(query)
            timings["routing_s"] = time.perf_counter() - t0
            routed, reason = decision.specs, decision.reason
            search_specs = routed
        else:
            routed, reason = list(self.all_specs), "routing off: whole corpus"
            search_specs = None  # one global search over all specs

        _ = self.vector_store.embedder.model  # load outside the timed section
        t0 = time.perf_counter()
        candidates = self.hybrid.retrieve(search_query, specs=search_specs, use_hybrid=use_hybrid)
        timings["retrieval_s"] = time.perf_counter() - t0

        if use_reranker:
            _ = self.reranker.model  # load outside the timed section
            t0 = time.perf_counter()
            pool = candidates[: r.get("rerank_candidates", len(candidates))]
            ranked = self.reranker.rerank(query, pool)
            if search_specs and len(search_specs) > 1:
                ranked = balance_by_spec(ranked, search_specs)
            timings["rerank_s"] = time.perf_counter() - t0
        else:
            ranked = list(candidates)
        results = ranked[:final_k]
        for rank, c in enumerate(results, start=1):
            c.final_rank = rank

        flags = {
            "use_hybrid": use_hybrid,
            "use_reranker": use_reranker,
            "use_routing": use_routing,
            "use_acronym_expansion": use_acronym_expansion,
        }
        return RetrievalResult(
            query, routed, reason, flags, candidates, results, timings,
            expanded_query=search_query if acronyms else None, acronyms=acronyms, route=decision,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """Process-wide pipeline instance (models and indexes loaded once)."""
    return Pipeline()


def retrieve(query: str, **kwargs: Any) -> RetrievalResult:
    """Convenience wrapper around ``get_pipeline().retrieve``."""
    return get_pipeline().retrieve(query, **kwargs)


# --------------------------------------------------------------------------- CLI


def _cell(c: Candidate | None, score: float | None, width: int) -> str:
    if c is None:
        return " " * width
    spec = c.chunk["spec"].replace("TS ", "")
    s = f"{spec} §{c.chunk['section']} {score:.3f}" if score is not None else f"{spec} §{c.chunk['section']}"
    return s[:width].ljust(width)


def compare(pipe: Pipeline, query: str, k: int, specs: Sequence[str] | None) -> None:
    """Print dense / hybrid / hybrid+rerank (no routing) and full (with routing) side by side."""
    common = {"specs": specs, "final_k": k}
    runs = [
        ("dense", pipe.retrieve(query, use_hybrid=False, use_reranker=False, use_routing=False, **common), "dense"),
        ("hybrid (RRF)", pipe.retrieve(query, use_hybrid=True, use_reranker=False, use_routing=False, **common), "fused"),
        ("hybrid + rerank", pipe.retrieve(query, use_hybrid=True, use_reranker=True, use_routing=False, **common), "rerank"),
        ("full (+ routing)", pipe.retrieve(query, use_hybrid=True, use_reranker=True, use_routing=True, **common), "rerank"),
    ]
    w = 26
    full = runs[3][1]
    print(f"\nQuery: {query}")
    if full.expanded_query:
        print(f"Acronyms expanded: {', '.join(full.acronyms)}")
    print(f"{full.route_label}\n")
    print("rank | " + " | ".join(name.ljust(w) for name, _, _ in runs))
    print("-----+-" + "-+-".join("-" * w for _ in runs))
    for i in range(k):
        cells = []
        for _, res, kind in runs:
            c = res.results[i] if i < len(res.results) else None
            score = None
            if c is not None:
                score = {"dense": c.dense_score, "fused": c.fused_score, "rerank": c.rerank_score}[kind]
            cells.append(_cell(c, score, w))
        print(f"{i + 1:>4} | " + " | ".join(cells))
    print()
    titles: dict[str, str] = {}
    for _, res, _ in runs:
        for c in res.results:
            titles[f"{c.chunk['spec']} §{c.chunk['section']}"] = c.chunk["title"]
    for key, title in titles.items():
        print(f"  {key}: {title}")
    t = full.timings
    print(f"\nTimings (full): routing {t.get('routing_s', 0) * 1000:.1f} ms, retrieval {t.get('retrieval_s', 0):.2f} s, "
          f"rerank {t.get('rerank_s', 0):.2f} s [reranker: {pipe.reranker.model_name}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare dense / hybrid / hybrid+rerank / full retrieval")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=None, help="results per configuration (default final_k)")
    parser.add_argument("--spec", action="append", help="restrict to a spec (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    pipe = get_pipeline()
    compare(pipe, " ".join(args.query), args.k or pipe.cfg["retrieval"]["final_k"], args.spec)
    return 0


if __name__ == "__main__":
    from src.retrieve.pipeline import main as _main

    raise SystemExit(_main())
