"""Answer generation: single entry point ``answer(query, ...)``.

Usage:
    python -m src.generate.answer "What does cause #9 mean in 5GMM?" [--provider groq] [--model ...]
        [--no-hybrid] [--no-rerank] [--no-routing] [--no-acronyms] [--spec "TS 24.501"] [-k 6]

Flow:
  1. retrieval (routing, hybrid search, reranking) via the retrieval pipeline;
  2. refusal layer 1: if the reranker is on and the best reranker score is below
     ``reranker.refusal_threshold``, return the refusal message without calling the LLM;
  3. prompt + LLM call (temperature 0);
  4. refusal layer 2: the prompt rule; an LLM refusal is normalised to the exact message;
  5. citation check: every (spec, section) cited must belong to a retrieved chunk; invalid ones are
     reported (and logged), not removed;
  6. return an :class:`AnswerResult`.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from src.config import load_config
from src.generate.citation_check import CitationReport, check_citations
from src.generate.llm import LLM, LLMUnavailable, get_llm
from src.generate.prompt import build_messages
from src.retrieve.hybrid import Candidate
from src.retrieve.pipeline import Pipeline, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Answer text plus everything needed to display and evaluate it."""

    query: str
    answer: str
    is_refusal: bool
    refusal_layer: str | None  # None | "threshold" | "llm" | "no_context"
    routed_specs: list[str]
    route_reason: str
    chunks: list[Candidate]
    top_score: float | None
    llm: str
    flags: dict[str, Any]
    citations: list[dict[str, str]] = field(default_factory=list)
    invalid_citations: list[dict[str, str]] = field(default_factory=list)
    citation_precision: float | None = None  # valid / all cited pairs; None when nothing is cited
    latency: dict[str, float] = field(default_factory=dict)
    retrieval: RetrievalResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "is_refusal": self.is_refusal,
            "refusal_layer": self.refusal_layer,
            "routed_specs": self.routed_specs,
            "route_reason": self.route_reason,
            "top_score": self.top_score,
            "llm": self.llm,
            "flags": self.flags,
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "citation_precision": self.citation_precision,
            "latency": self.latency,
            "chunks": [c.to_dict() for c in self.chunks],
        }


def _canon(text: str) -> str:
    return re.sub(r"[\s\"'`.*_]+", " ", text).strip().lower()


_MODEL_MARKERS = re.compile(r"\s*【[^】]*】")


def clean_answer(text: str) -> str:
    """Remove model-specific reference marks such as ``【1†L7-L12】`` (the spec citations stay)."""
    return _MODEL_MARKERS.sub("", text).strip()


def is_refusal_text(text: str, refusal_message: str) -> bool:
    """True if the LLM reply is (essentially only) the refusal message.

    Tolerates quotes, trailing period, markdown emphasis and whitespace; a partial answer that
    also *contains* the sentence (rule 4 of the prompt) is not a refusal.
    """
    return _canon(text) == _canon(refusal_message)


class Answerer:
    """Retrieval pipeline + LLM."""

    def __init__(self, cfg: dict[str, Any] | None = None, llm: LLM | None = None,
                 pipeline: Pipeline | None = None) -> None:
        self.cfg = cfg or load_config()
        self.llm = llm or get_llm(cfg=self.cfg)
        classifier = None
        try:
            self.llm.check()
            classifier = self.llm.classify
        except LLMUnavailable as exc:
            logger.warning("LLM unavailable for routing ties (%s); ties will search both specs", exc)
        self.pipeline = pipeline or Pipeline(self.cfg, llm_classifier=classifier)
        if pipeline is not None and classifier is not None:
            self.pipeline.router.llm_classifier = classifier
        self.refusal_message: str = self.cfg["refusal_message"]
        self.threshold: float = self.cfg["reranker"]["refusal_threshold"]

    def answer(
        self,
        query: str,
        use_hybrid: bool | None = None,
        use_reranker: bool | None = None,
        use_routing: bool | None = None,
        use_acronym_expansion: bool | None = None,
        specs: Sequence[str] | None = None,
        final_k: int | None = None,
        llm: LLM | None = None,
    ) -> AnswerResult:
        """Answer one question (flags override ``config.yaml``; ``llm`` overrides the default LLM)."""
        llm = llm or self.llm
        t_start = time.perf_counter()
        ret = self.pipeline.retrieve(
            query, use_hybrid=use_hybrid, use_reranker=use_reranker, use_routing=use_routing,
            use_acronym_expansion=use_acronym_expansion, specs=specs, final_k=final_k,
        )
        latency = {"retrieval_s": sum(ret.timings.values())}

        def result(text: str, refusal: bool, layer: str | None,
                   report: CitationReport | None = None) -> AnswerResult:
            latency["total_s"] = time.perf_counter() - t_start
            report = report or CitationReport()
            return AnswerResult(
                query=query, answer=text, is_refusal=refusal, refusal_layer=layer,
                routed_specs=ret.routed_specs, route_reason=ret.route_reason, chunks=ret.results,
                top_score=ret.top_score, llm=llm.name, flags=ret.flags,
                citations=[{"spec": c.spec, "section": c.section} for c in report.valid],
                invalid_citations=[{"spec": c.spec, "section": c.section} for c in report.invalid],
                citation_precision=report.precision, latency=latency, retrieval=ret,
            )

        if not ret.results:
            return result(self.refusal_message, True, "no_context")

        # Refusal layer 1: reranker confidence (only meaningful when the reranker ran).
        if ret.flags["use_reranker"] and ret.top_score is not None and ret.top_score < self.threshold:
            logger.info("Refusal (layer 1): top reranker score %.3f < %.2f", ret.top_score, self.threshold)
            return result(self.refusal_message, True, "threshold")

        messages = build_messages(query, [c.chunk for c in ret.results], self.refusal_message)
        t0 = time.perf_counter()
        text = clean_answer(llm.generate(messages))
        latency["llm_s"] = time.perf_counter() - t0

        # Refusal layer 2: the prompt rule. Normalise to the exact message.
        if is_refusal_text(text, self.refusal_message):
            return result(self.refusal_message, True, "llm")

        # Citation check against the chunks actually given to the LLM.
        report = check_citations(text, [c.chunk for c in ret.results])
        return result(text, False, None, report)


@lru_cache(maxsize=1)
def get_answerer() -> Answerer:
    """Process-wide answerer (models, indexes and LLM client loaded once)."""
    return Answerer()


def answer(query: str, **kwargs: Any) -> AnswerResult:
    """Convenience wrapper around ``get_answerer().answer``."""
    return get_answerer().answer(query, **kwargs)


# --------------------------------------------------------------------------- CLI


def print_result(res: AnswerResult) -> None:
    print(f"\nQuestion: {res.query}")
    print(f"Routed to: {' + '.join(res.routed_specs)} ({res.route_reason})")
    top = f"{res.top_score:.3f}" if res.top_score is not None else "n/a"
    print(f"LLM: {res.llm} | top reranker score: {top} | refusal: {res.is_refusal}"
          + (f" (layer: {res.refusal_layer})" if res.is_refusal else ""))
    print("\nAnswer:\n" + res.answer)
    if not res.is_refusal:
        n_valid, n_invalid = len(res.citations), len(res.invalid_citations)
        prec = f"{res.citation_precision:.0%}" if res.citation_precision is not None else "n/a"
        print(f"\nCitations: {n_valid} valid, {n_invalid} invalid (precision {prec})")
        for c in res.citations:
            print(f"  [valid]   ({c['spec']}, §{c['section']})")
        for c in res.invalid_citations:
            print(f"  [INVALID] ({c['spec']}, §{c['section']}) - not in the retrieved context")
        if not n_valid and not n_invalid:
            print("  (the answer contains no citation)")
    print("\nSources:")
    for c in res.chunks:
        score = f"{c.rerank_score:.3f}" if c.rerank_score is not None else f"rrf {c.fused_score:.4f}"
        print(f"  [{c.final_rank}] {c.chunk['spec']} §{c.chunk['section']} {c.chunk['title']} ({score})")
    lat = ", ".join(f"{k} {v:.2f}s" for k, v in res.latency.items())
    print(f"\nLatency: {lat}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a question from TS 23.501 / TS 24.501")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--provider", choices=["groq", "gemini", "openai", "ollama"])
    parser.add_argument("--model")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-routing", action="store_true")
    parser.add_argument("--no-acronyms", action="store_true")
    parser.add_argument("--spec", action="append", help="manual spec filter (repeatable)")
    parser.add_argument("-k", type=int, help="number of context chunks")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    llm = get_llm(args.provider, args.model, cfg)
    try:
        llm.check()
    except LLMUnavailable as exc:
        print(f"LLM not available: {exc}", file=sys.stderr)
        return 2
    answerer = Answerer(cfg, llm=llm)
    res = answerer.answer(
        " ".join(args.query),
        use_hybrid=False if args.no_hybrid else None,
        use_reranker=False if args.no_rerank else None,
        use_routing=False if args.no_routing else None,
        use_acronym_expansion=False if args.no_acronyms else None,
        specs=args.spec,
        final_k=args.k,
    )
    print_result(res)
    return 0


if __name__ == "__main__":
    from src.generate.answer import main as _main

    raise SystemExit(_main())
