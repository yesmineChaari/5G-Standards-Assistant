"""Tests for citation extraction and validation against the retrieved chunks."""

from __future__ import annotations

from typing import Any

import pytest

from src.generate.citation_check import Citation, check_citations, extract_citations, has_spec_mention


def _chunk(spec: str, section: str, merged: list[str] | None = None) -> dict[str, Any]:
    return {"spec": spec, "section": section, "merged_sections": merged or []}


def cits(text: str) -> list[tuple[str, str]]:
    return [(c.spec, c.section) for c in extract_citations(text)]


# --------------------------------------------------------------------------- extraction


def test_single_citation() -> None:
    assert cits("Cause #9 means X (TS 24.501, §A.1).") == [("TS 24.501", "A.1")]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(TS 24.501, §5.5.1.2.2)", [("TS 24.501", "5.5.1.2.2")]),
        ("(TS 24.501 §5.5.1.2.2)", [("TS 24.501", "5.5.1.2.2")]),  # no comma
        ("(TS24.501, § 5.5.1.2.2)", [("TS 24.501", "5.5.1.2.2")]),  # spacing
        ("(3GPP TS 23.501, §4.2.11)", [("TS 23.501", "4.2.11")]),
        ("(TS 23.501, §5.20b.1)", [("TS 23.501", "5.20b.1")]),  # letter suffix in the middle
        ("(TS 24.501, §5.4.1.2.3A.1)", [("TS 24.501", "5.4.1.2.3A.1")]),
        ("(TS 24.501, §9.11.3.9A)", [("TS 24.501", "9.11.3.9A")]),
        ("(TS 24.501, §A)", [("TS 24.501", "A")]),  # whole annex
    ],
)
def test_citation_variants(text: str, expected: list[tuple[str, str]]) -> None:
    assert cits(text) == expected


def test_multiple_citations_separated_by_semicolon() -> None:
    text = "Both apply (TS 24.501, §5.5.1.2.2; TS 23.501, §5.15.2)."
    assert cits(text) == [("TS 24.501", "5.5.1.2.2"), ("TS 23.501", "5.15.2")]


@pytest.mark.parametrize(
    "text",
    [
        "(TS 24.501, §5.6.1.5 and §5.5.1.3.5)",
        "(TS 24.501, §§5.6.1.5, 5.5.1.3.5)",
        "(TS 24.501, §5.6.1.5, §5.5.1.3.5)",
        "(TS 24.501, §5.6.1.5 / 5.5.1.3.5)",
    ],
)
def test_several_sections_for_one_spec(text: str) -> None:
    assert cits(text) == [("TS 24.501", "5.6.1.5"), ("TS 24.501", "5.5.1.3.5")]


def test_several_citations_in_one_answer_are_deduplicated_in_order() -> None:
    text = "A (TS 24.501, §A.1). B (TS 23.501, §6.2.6). C again (TS 24.501, §A.1)."
    assert cits(text) == [("TS 24.501", "A.1"), ("TS 23.501", "6.2.6")]


def test_sentence_period_and_bare_numbers_are_not_sections() -> None:
    assert cits("See TS 24.501, §A.1. 3 timers apply.") == [("TS 24.501", "A.1")]
    assert cits("TS 24.501 §5.5.1, 3 timers run") == [("TS 24.501", "5.5.1")]


def test_spec_without_section_is_not_a_citation() -> None:
    assert cits("This is defined in TS 24.501.") == []
    assert has_spec_mention("This is defined in TS 24.501.")
    assert not has_spec_mention("No spec here.")


def test_other_spec_is_extracted() -> None:
    assert cits("(TS 23.502, §4.2.2.2.2)") == [("TS 23.502", "4.2.2.2.2")]


# --------------------------------------------------------------------------- validation


def test_valid_and_invalid_citations() -> None:
    chunks = [_chunk("TS 24.501", "A.1"), _chunk("TS 24.501", "5.6.1.5")]
    text = "X (TS 24.501, §A.1). Y (TS 24.501, §5.6.1.5 and §5.5.1.3.5). Z (TS 23.501, §A.1)."
    report = check_citations(text, chunks)
    assert report.valid == [Citation("TS 24.501", "A.1"), Citation("TS 24.501", "5.6.1.5")]
    # wrong section, and right section number but wrong spec
    assert report.invalid == [Citation("TS 24.501", "5.5.1.3.5"), Citation("TS 23.501", "A.1")]
    assert report.precision == pytest.approx(2 / 4)


def test_merged_sections_are_valid() -> None:
    chunks = [_chunk("TS 23.501", "4.2.11", merged=["4.2.11", "4.2.12"])]
    report = check_citations("MBS (TS 23.501, §4.2.12).", chunks)
    assert report.valid == [Citation("TS 23.501", "4.2.12")] and report.invalid == []


def test_parent_or_child_section_is_invalid() -> None:
    chunks = [_chunk("TS 24.501", "5.5.1.2.2")]
    report = check_citations("(TS 24.501, §5.5.1) and (TS 24.501, §5.5.1.2.2.1)", chunks)
    assert report.valid == [] and len(report.invalid) == 2


def test_other_spec_is_invalid() -> None:
    report = check_citations("(TS 23.502, §4.2.2.2.2)", [_chunk("TS 23.501", "4.2.2.2.2")])
    assert report.invalid == [Citation("TS 23.502", "4.2.2.2.2")]


def test_no_citation_has_no_precision() -> None:
    report = check_citations("An answer without citations.", [_chunk("TS 24.501", "A.1")])
    assert report.n_total == 0 and report.precision is None


def test_invalid_citations_are_logged(caplog) -> None:
    with caplog.at_level("WARNING"):
        check_citations("(TS 24.501, §9.9.9)", [_chunk("TS 24.501", "A.1")])
    assert "(TS 24.501, §9.9.9)" in caplog.text


# --------------------------------------------------------------------------- integration in answer()


def test_answer_reports_citations_without_removing_them() -> None:
    from tests.test_answer import _answerer

    reply = "Cause #9 is X (TS 24.501, §A.1). Also Y (TS 24.501, §5.5.1.3.5)."
    ans, _ = _answerer(reply, top_score=0.9)
    res = ans.answer("What does cause #9 mean in 5GMM?")
    assert res.answer == reply  # nothing deleted
    assert res.citations == [{"spec": "TS 24.501", "section": "A.1"}]
    assert res.invalid_citations == [{"spec": "TS 24.501", "section": "5.5.1.3.5"}]
    assert res.citation_precision == pytest.approx(0.5)
    assert res.to_dict()["citation_precision"] == pytest.approx(0.5)


def test_refusal_has_no_citations() -> None:
    from tests.test_answer import _answerer

    ans, _ = _answerer("unused", top_score=0.01)
    res = ans.answer("q")
    assert res.is_refusal and res.citations == [] and res.citation_precision is None
