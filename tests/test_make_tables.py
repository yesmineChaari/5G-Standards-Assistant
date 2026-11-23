"""Tests for the ablation helpers (paired deltas, formatting)."""

from __future__ import annotations

import pytest

from eval.make_tables import fmt_delta, micro, paired_delta


def test_paired_delta_uses_common_questions() -> None:
    full = {"q1": 1.0, "q2": 0.5, "q3": 1.0}
    other = {"q1": 0.0, "q2": 0.5, "q4": 1.0}  # q3/q4 not shared
    d, ci, n = paired_delta(full, other)
    assert n == 2 and d == pytest.approx(-0.5)
    assert ci[0] <= d <= ci[1]


def test_paired_delta_empty() -> None:
    assert paired_delta({}, {"q1": 1.0}) == (None, None, 0)


def test_fmt_delta_marks_significance() -> None:
    assert fmt_delta(-0.1, (-0.17, -0.04)).endswith("*")
    assert not fmt_delta(0.01, (-0.05, 0.07)).endswith("*")
    assert fmt_delta(None, None) == "-"


def test_micro_pools_citations_over_answered_in_corpus() -> None:
    rows = {
        "a": {"type": "simple_24501", "is_refusal": False, "citation_stats": {"n": 3, "valid": 3, "correct": 1}},
        "b": {"type": "cross_spec", "is_refusal": False, "citation_stats": {"n": 1, "valid": 0, "correct": 0}},
        "c": {"type": "simple_24501", "is_refusal": True, "citation_stats": {"n": 0, "valid": 0, "correct": 0}},
        "d": {"type": "out_of_corpus", "is_refusal": False, "citation_stats": {"n": 2, "valid": 2, "correct": 0}},
    }
    assert micro(rows, "valid") == pytest.approx(3 / 4)
    assert micro(rows, "correct") == pytest.approx(1 / 4)
