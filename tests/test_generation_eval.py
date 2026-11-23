"""Tests for the generation-evaluation helpers (no LLM, no models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.run_generation_eval import CachedLLM, aggregate, apply_ragas, citation_stats, read_manual, write_manual
from src.generate.llm import LLM

REFUSAL = "I could not find this information in the documents."


def _entry(qid: str = "S24-01", type_: str = "simple") -> dict:
    return {"id": qid, "type": type_, "question": "What does cause #9 mean?", "expected_spec": ["TS 24.501"],
            "gold_sections": [{"spec": "TS 24.501", "section": "A.1"}], "reference_answer": "ref"}


def _rec(answer: str = "x", refusal: bool = False, valid=None, invalid=None) -> dict:
    return {"answer": answer, "is_refusal": refusal, "refusal_layer": "llm" if refusal else None,
            "citations": valid or [], "invalid_citations": invalid or [], "contexts": ["c"],
            "latency": {"total_s": 2.0, "llm_s": 1.0}}


def test_citation_stats_validity_and_correctness() -> None:
    rec = _rec(valid=[{"spec": "TS 24.501", "section": "A.1"}, {"spec": "TS 24.501", "section": "5.5.1"}],
               invalid=[{"spec": "TS 24.501", "section": "A.1.2"}])
    s = citation_stats(rec, _entry())
    assert s == {"n": 3, "valid": 2, "correct": 2}  # A.1 and its sub-section A.1.2 match gold A.1


def test_aggregate_refusals_and_citations() -> None:
    def row(type_, refusal, exact=False, cit=(0, 0, 0)):
        return {"type": type_, "is_refusal": refusal, "exact_refusal": exact,
                "cit": {"n": cit[0], "valid": cit[1], "correct": cit[2]},
                "ragas": {"faithfulness": None, "answer_correctness": None}, "manual": None,
                "latency": {"total_s": 1.0}}

    rows = [row("simple_24501", False, cit=(2, 2, 1)), row("simple_24501", True),
            row("cross_spec", False, cit=(0, 0, 0)), row("out_of_corpus", True, exact=True),
            row("out_of_corpus", False)]
    a = aggregate(rows)
    assert a["false_refusal"] == pytest.approx(1 / 3)
    assert a["refusal_acc"] == pytest.approx(0.5)
    assert a["with_citation"] == pytest.approx(0.5)  # 1 of the 2 answered in-corpus questions
    assert a["cit_validity"] == 1.0 and a["cit_correctness"] == 0.5


class _FakeJudge:
    def __init__(self) -> None:
        self.calls = []

    def score(self, metric, question, answer, contexts, reference=""):
        self.calls.append(metric)
        return 0.8


def test_ragas_skips_refusals_and_out_of_corpus() -> None:
    judge = _FakeJudge()
    assert apply_ragas(judge, _rec(REFUSAL, refusal=True), _entry()) == {"faithfulness": None, "answer_correctness": 0.0}
    assert apply_ragas(judge, _rec(REFUSAL, refusal=True), _entry("OOC-01", "out_of_corpus"))["answer_correctness"] is None
    assert judge.calls == []
    assert apply_ragas(judge, _rec("answer"), _entry()) == {"faithfulness": 0.8, "answer_correctness": 0.8}


def test_manual_grades_survive_rewrite_only_for_same_answer(tmp_path: Path) -> None:
    path = tmp_path / "manual.csv"
    entries = [_entry("S24-01"), _entry("S24-02")]
    write_manual(path, entries, {"S24-01": _rec("a1"), "S24-02": _rec("a2")}, "full")
    text = path.read_text(encoding="utf-8-sig").replace("a1,,,,", "a1,,,2,ok", 1)
    path.write_text(text, encoding="utf-8-sig")
    assert read_manual(path) == {("full", "S24-01"): 2.0}
    write_manual(path, entries, {"S24-01": _rec("a1"), "S24-02": _rec("a2")}, "full")
    assert read_manual(path) == {("full", "S24-01"): 2.0}  # kept
    write_manual(path, entries, {"S24-01": _rec("a1 changed"), "S24-02": _rec("a2")}, "full")
    assert read_manual(path) == {}  # answer changed: grade dropped


def test_cached_llm_reuses_identical_calls(tmp_path: Path, monkeypatch) -> None:
    llm = LLM("groq", "m")
    calls = []
    monkeypatch.setattr(llm, "generate", lambda messages, temperature=None, max_tokens=None: calls.append(1) or "reply")
    cached = CachedLLM(llm, tmp_path / "calls.jsonl")
    msgs = [{"role": "user", "content": "q"}]
    assert cached.generate(msgs) == "reply" and cached.generate(msgs) == "reply"
    assert len(calls) == 1 and cached.hits == 1 and cached.misses == 1
    again = CachedLLM(llm, tmp_path / "calls.jsonl")  # reloaded from disk
    assert again.generate(msgs) == "reply" and len(calls) == 1
    assert again.generate([{"role": "user", "content": "other"}]) == "reply" and len(calls) == 2


def test_daily_limit_detection() -> None:
    from eval.judge import is_daily_limit

    assert is_daily_limit(Exception("Rate limit reached ... on tokens per day (TPD): Limit 200000"))
    assert not is_daily_limit(Exception("Rate limit reached ... on tokens per minute (TPM): Limit 8000"))
    assert is_daily_limit(Exception("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert not is_daily_limit(Exception("429 quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"))


def test_judge_comparison_rows() -> None:
    from eval.run_generation_eval import judge_comparison_rows

    comp = [{"config": "full", "judge": "groq:j", "metric": "faithfulness", "current": c, "other": o}
            for c, o in ((0.5, 0.7), (0.8, 1.0), (1.0, 1.0))]
    (row,) = judge_comparison_rows(comp)
    assert row[:4] == ["`full`", "faithfulness", "3", "0.77"]
    assert row[5] == "+0.13" and row[6] == "0.13"
    assert float(row[7]) > 0.9  # scores move together


def test_cache_only_judge_never_calls(tmp_path, monkeypatch) -> None:
    from eval import judge as judge_mod

    monkeypatch.setattr(judge_mod, "CACHE_FILE", str(tmp_path / "ragas.jsonl"))
    j = judge_mod.RagasJudge(provider="groq", model="m", cache_only=True)
    assert j.score("faithfulness", "q", "a", ["c"]) is None and j.missing == 0 and j._metrics is None
