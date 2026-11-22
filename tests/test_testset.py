"""The evaluation test set stays valid (schema, counts, gold sections exist, out-of-corpus terms absent)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from find_sections import check_testset, find_terms, gold_exists  # noqa: E402

from src.ingest.chunker import load_chunks  # noqa: E402

TESTSET = ROOT / "eval" / "testset.yaml"
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
pytestmark = pytest.mark.skipif(not CHUNKS.exists(), reason="chunks.jsonl not built (run make ingest)")


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


def test_testset_is_valid(chunks) -> None:
    assert check_testset(TESTSET, chunks) == 0


def test_absent_terms_use_whole_words(chunks) -> None:
    found = find_terms(chunks, ["n78", "LTE-M"])
    assert found["n78"] == [] and found["LTE-M"]  # LTE-M occurs in TS 23.501, n78 does not


def test_gold_parent_section_exists_through_subsections() -> None:
    sections = {("TS 24.501", "5.1.3.2.1.1"), ("TS 24.501", "A.1")}
    assert gold_exists("TS 24.501", "5.1.3.2.1", sections)
    assert gold_exists("TS 24.501", "A.1", sections)
    assert not gold_exists("TS 24.501", "5.1.3.2.1.1.9", sections)
    assert not gold_exists("TS 23.501", "A.1", sections)
