"""Test-set loading, the section matching rule and retrieval metrics (shared by the evaluations).

Matching rule: a retrieved chunk is relevant to a gold section if it has the same spec and its
section equals the gold section, is a sub-section of it (``5.5.1.2.2`` matches gold ``5.5.1.2``),
or the gold section (or a sub-section of it) is in the chunk's ``merged_sections``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from src.config import resolve_path

logger = logging.getLogger(__name__)

DEFAULT_TESTSET = "eval/testset.yaml"
K_VALUES = (1, 3, 5, 10)
BOTH = ["TS 23.501", "TS 24.501"]


# --------------------------------------------------------------------------- test set


def load_testset(path: str | Path | None = None, warn: bool = True) -> list[dict[str, Any]]:
    p = Path(path) if path else resolve_path(DEFAULT_TESTSET)
    entries = yaml.safe_load(p.read_text(encoding="utf-8"))
    for e in entries:
        for g in e["gold_sections"]:
            g["section"] = str(g["section"])
    if warn:
        unverified = [e["id"] for e in entries if not e.get("verified")]
        if unverified:
            logger.warning("%d/%d test-set entries are NOT verified (gold data may be wrong): %s",
                           len(unverified), len(entries), ", ".join(unverified))
    return entries


def unverified_ids(entries: Iterable[dict[str, Any]]) -> list[str]:
    return [e["id"] for e in entries if not e.get("verified")]


def question_group(entry: dict[str, Any]) -> str:
    """simple_23501 | simple_24501 | cross_spec | out_of_corpus."""
    if entry["type"] == "simple":
        return "simple_" + entry["expected_spec"][0].replace("TS ", "").replace(".", "")
    return entry["type"]


# --------------------------------------------------------------------------- matching rule


def _covers(section: str, gold: str) -> bool:
    return section == gold or section.startswith(gold + ".")


def matches_gold(chunk: dict[str, Any], gold: dict[str, str]) -> bool:
    if chunk["spec"] != gold["spec"]:
        return False
    sections = [chunk["section"], *(chunk.get("merged_sections") or [])]
    return any(_covers(s, gold["section"]) for s in sections)


def is_relevant(chunk: dict[str, Any], golds: Sequence[dict[str, str]]) -> bool:
    return any(matches_gold(chunk, g) for g in golds)


# --------------------------------------------------------------------------- metrics


def recall_at_k(chunks: Sequence[dict[str, Any]], golds: Sequence[dict[str, str]], k: int) -> float:
    """Fraction of gold sections matched by at least one of the top-k chunks."""
    if not golds:
        return 0.0
    top = chunks[:k]
    return sum(any(matches_gold(c, g) for c in top) for g in golds) / len(golds)


def hit_at_k(chunks: Sequence[dict[str, Any]], golds: Sequence[dict[str, str]], k: int) -> float:
    """1.0 if at least one of the top-k chunks is relevant."""
    return float(any(is_relevant(c, golds) for c in chunks[:k]))


def first_relevant_rank(chunks: Sequence[dict[str, Any]], golds: Sequence[dict[str, str]]) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        if is_relevant(c, golds):
            return rank
    return None


def reciprocal_rank(chunks: Sequence[dict[str, Any]], golds: Sequence[dict[str, str]]) -> float:
    rank = first_relevant_rank(chunks, golds)
    return 1.0 / rank if rank else 0.0


def both_specs_present(chunks: Sequence[dict[str, Any]], k: int) -> float:
    """Cross-spec: both specs appear among the top-k chunks."""
    return float(set(BOTH) <= {c["spec"] for c in chunks[:k]})


def both_specs_relevant(chunks: Sequence[dict[str, Any]], golds: Sequence[dict[str, str]], k: int) -> float:
    """Cross-spec: a relevant chunk from each spec appears among the top-k chunks."""
    top = chunks[:k]
    return float(all(any(c["spec"] == spec and is_relevant(c, golds) for c in top) for spec in BOTH))


def question_metrics(chunks: Sequence[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any]:
    golds = entry["gold_sections"]
    m: dict[str, Any] = {f"recall@{k}": recall_at_k(chunks, golds, k) for k in K_VALUES}
    m.update({f"hit@{k}": hit_at_k(chunks, golds, k) for k in K_VALUES})
    m["rr"] = reciprocal_rank(chunks, golds)
    m["first_rank"] = first_relevant_rank(chunks, golds)
    if entry["type"] == "cross_spec":
        for k in (5, 10):
            m[f"both_specs@{k}"] = both_specs_present(chunks, k)
            m[f"both_relevant@{k}"] = both_specs_relevant(chunks, golds, k)
    return m


def route_scores(routed: Sequence[str], expected: Sequence[str]) -> dict[str, bool]:
    """exact: same set of specs; covers: every expected spec was searched (no recall lost)."""
    r, e = set(routed), set(expected)
    return {"exact": r == e, "covers": e <= r}


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval of the mean (resampling questions)."""
    import random

    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi

