"""Tests for the section matching rule and retrieval metrics."""

from __future__ import annotations

import pytest

from eval.metrics import (
    both_specs_present,
    both_specs_relevant,
    first_relevant_rank,
    hit_at_k,
    matches_gold,
    question_group,
    question_metrics,
    recall_at_k,
    reciprocal_rank,
    route_scores,
)


def ch(spec: str, section: str, merged: list[str] | None = None) -> dict:
    return {"spec": f"TS {spec}", "section": section, "merged_sections": merged or []}


def gold(spec: str, section: str) -> dict:
    return {"spec": f"TS {spec}", "section": section}


def test_matching_rule() -> None:
    g = gold("24.501", "5.5.1.2")
    assert matches_gold(ch("24.501", "5.5.1.2"), g)
    assert matches_gold(ch("24.501", "5.5.1.2.2"), g)  # sub-section
    assert not matches_gold(ch("24.501", "5.5.1"), g)  # parent does not match
    assert not matches_gold(ch("24.501", "5.5.1.20"), g)  # prefix but not a sub-section
    assert not matches_gold(ch("23.501", "5.5.1.2"), g)  # other spec
    assert matches_gold(ch("23.501", "4.2.11", ["4.2.11", "4.2.12"]), gold("23.501", "4.2.12"))  # merged


def test_recall_hit_and_mrr() -> None:
    golds = [gold("24.501", "A.1"), gold("24.501", "9.11.3.2")]
    chunks = [ch("24.501", "10.2"), ch("24.501", "A.1"), ch("24.501", "5.5.1"), ch("24.501", "9.11.3.2")]
    assert recall_at_k(chunks, golds, 1) == 0.0
    assert recall_at_k(chunks, golds, 3) == 0.5
    assert recall_at_k(chunks, golds, 5) == 1.0
    assert hit_at_k(chunks, golds, 1) == 0.0 and hit_at_k(chunks, golds, 3) == 1.0
    assert first_relevant_rank(chunks, golds) == 2
    assert reciprocal_rank(chunks, golds) == pytest.approx(0.5)
    assert reciprocal_rank([ch("24.501", "10.2")], golds) == 0.0


def test_recall_counts_each_gold_once() -> None:
    golds = [gold("24.501", "A.1"), gold("24.501", "9.11.3.2")]
    chunks = [ch("24.501", "A.1"), ch("24.501", "A.1"), ch("24.501", "A.1")]  # repeated chunks of one section
    assert recall_at_k(chunks, golds, 3) == 0.5


def test_cross_spec_metrics() -> None:
    golds = [gold("23.501", "5.3.2.1"), gold("24.501", "5.5.1.2.2")]
    chunks = [ch("24.501", "5.5.1.2.2"), ch("23.501", "6.2.1"), ch("23.501", "5.3.2.1")]
    assert both_specs_present(chunks, 2) == 1.0
    assert both_specs_relevant(chunks, golds, 2) == 0.0  # the 23.501 chunk in the top 2 is not relevant
    assert both_specs_relevant(chunks, golds, 3) == 1.0
    entry = {"type": "cross_spec", "gold_sections": golds}
    m = question_metrics(chunks, entry)
    assert m["both_specs@5"] == 1.0 and m["both_relevant@5"] == 1.0 and m["first_rank"] == 1


def test_route_scores() -> None:
    assert route_scores(["TS 24.501"], ["TS 24.501"]) == {"exact": True, "covers": True}
    assert route_scores(["TS 23.501", "TS 24.501"], ["TS 24.501"]) == {"exact": False, "covers": True}
    assert route_scores(["TS 23.501"], ["TS 24.501"]) == {"exact": False, "covers": False}


def test_question_group() -> None:
    assert question_group({"type": "simple", "expected_spec": ["TS 23.501"]}) == "simple_23501"
    assert question_group({"type": "cross_spec", "expected_spec": []}) == "cross_spec"


def test_bootstrap_ci_contains_mean_and_is_deterministic() -> None:
    from eval.metrics import bootstrap_ci

    vals = [0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0]
    lo, hi = bootstrap_ci(vals)
    assert lo <= sum(vals) / len(vals) <= hi and 0.0 <= lo < hi <= 1.0
    assert bootstrap_ci(vals) == (lo, hi)
    assert bootstrap_ci([1.0]) is None
