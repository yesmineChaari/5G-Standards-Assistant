"""Tests for src.ingest.chunker (synthetic sections + checks on chunks.jsonl when present)."""

from __future__ import annotations

import pytest

from src.config import load_config, resolve_path
from src.ingest.chunker import Chunker, load_chunks
from src.ingest.parse_docx import Block, Section

SPEC = {"id": "TS 99.999", "short": "99999", "version": "18.0.0", "release": "Rel-18"}
MAX, OVERLAP, MIN, MAX_TABLE = 100, 20, 15, 150


@pytest.fixture(scope="module")
def chunker() -> Chunker:
    return Chunker(max_tokens=MAX, overlap_tokens=OVERLAP, min_tokens=MIN, max_table_tokens=MAX_TABLE)


def sec(number: str, title: str, blocks: list[Block], path: str | None = None) -> Section:
    parts = number.split(".")
    path = path or " > ".join(".".join(parts[: i + 1]) for i in range(len(parts)))
    return Section(SPEC["id"], number, title, len(parts), path, blocks=blocks)


def text(s: str) -> Block:
    return Block("text", s)


def long_paragraphs(n: int, words: int = 30) -> list[Block]:
    return [
        text(" ".join(f"para{i}word{j}" for j in range(words)) + f". End of paragraph {i}.")
        for i in range(n)
    ]


def table(n_rows: int, caption: str = "Table 1.1: Test table") -> Block:
    rows = ["| Cause | Meaning |", "|---|---|"]
    rows += [f"| #{i} | meaning of cause number {i} with some words |" for i in range(n_rows)]
    return Block("table", "\n".join(rows), caption=caption)


def test_short_section_single_chunk(chunker: Chunker) -> None:
    chunks = chunker.chunk_sections([sec("5.1", "General", [text("A " * 30)])], SPEC)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "99999_5.1_001"
    assert c.text_for_index.startswith("[TS 99.999 | Rel-18 | §5.1 | General]\n")
    assert c.text_for_index.endswith(c.text)
    assert c.merged_sections == []


def test_long_section_respects_max_tokens_and_overlaps(chunker: Chunker) -> None:
    chunks = chunker.chunk_sections([sec("5.2", "Long", long_paragraphs(12, words=6))], SPEC)
    assert len(chunks) > 1
    assert all(c.n_tokens <= MAX for c in chunks)
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    # overlap: each chunk after the first starts with text that ends the previous chunk
    for prev, cur in zip(chunks, chunks[1:]):
        first_line = cur.text.splitlines()[0]
        assert first_line in prev.text


def test_very_long_paragraph_split_by_sentences(chunker: Chunker) -> None:
    para = " ".join(f"Sentence number {i} has a few words in it." for i in range(80))
    chunks = chunker.chunk_sections([sec("5.3", "One paragraph", [text(para)])], SPEC)
    assert len(chunks) > 1
    assert all(c.n_tokens <= MAX for c in chunks)


def test_small_table_is_one_chunk(chunker: Chunker) -> None:
    chunks = chunker.chunk_sections([sec("9.1", "IE", [table(3)])], SPEC)
    assert len(chunks) == 1 and chunks[0].chunk_type == "table"
    assert chunks[0].text.startswith("Table 1.1: Test table\n| Cause | Meaning |")


def test_large_table_split_by_rows_with_header(chunker: Chunker) -> None:
    block = table(40)
    chunks = chunker.chunk_sections([sec("9.2", "Big IE", [block])], SPEC)
    assert len(chunks) > 1
    original_rows = set(block.content.splitlines()[2:])
    seen_rows: list[str] = []
    for c in chunks:
        assert c.chunk_type == "table"
        assert c.n_tokens <= MAX_TABLE
        lines = c.text.splitlines()
        assert lines[0] == "Table 1.1: Test table"
        assert lines[1] == "| Cause | Meaning |" and lines[2] == "|---|---|"
        for row in lines[3:]:
            assert row in original_rows  # never a partial row
            seen_rows.append(row)
    assert sorted(seen_rows) == sorted(original_rows)  # every row exactly once


def test_prose_row_in_table_is_chunked_as_text(chunker: Chunker) -> None:
    prose = "<br>".join(f"Field {i} (octet {i}) is coded as described below with details." for i in range(40))
    block = Block("table", f"| {prose} |\n|---|\n| NOTE: short note. |", caption="Table 9.3.1: Prose IE")
    chunks = chunker.chunk_sections([sec("9.3", "Prose IE", [block])], SPEC)
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    assert text_chunks and all(c.n_tokens <= MAX for c in text_chunks)
    assert all(c.text.startswith("Table 9.3.1: Prose IE (continued)") for c in text_chunks)
    joined = " ".join(c.text for c in chunks)
    assert "Field 0 (octet 0)" in joined and "Field 39 (octet 39)" in joined
    assert "NOTE: short note." in joined


def test_small_intro_text_folded_into_table(chunker: Chunker) -> None:
    s = sec("9.4", "Cause IE", [text("The cause IE is coded as follows."), table(3)])
    chunks = chunker.chunk_sections([s], SPEC)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "table"
    assert chunks[0].text.startswith("The cause IE is coded as follows.")


def test_small_sibling_sections_are_merged(chunker: Chunker) -> None:
    sections = [
        sec("5.4.1", "Void", [text("Void.")]),
        sec("5.4.2", "Short", [text("Short text.")]),
        sec("5.4.3", "Normal", [text("word " * 60)]),
    ]
    chunks = chunker.chunk_sections(sections, SPEC)
    merged = chunks[0]
    assert merged.section == "5.4.1"
    assert merged.merged_sections[:2] == ["5.4.1", "5.4.2"]
    assert "§5.4.2 Short" in merged.text
    assert all(c.n_tokens <= MAX for c in chunks)


def test_small_section_not_merged_with_child(chunker: Chunker) -> None:
    sections = [sec("5.5", "Parent", [text("Intro.")]), sec("5.5.1", "Child", [text("Child text.")])]
    chunks = chunker.chunk_sections(sections, SPEC)
    assert [c.section for c in chunks] == ["5.5", "5.5.1"]
    assert all(c.merged_sections == [] for c in chunks)


def test_duplicate_section_numbers_get_unique_ids(chunker: Chunker) -> None:
    sections = [sec("8.2.6.35", "A", [text("word " * 50)]), sec("8.2.6.35", "B", [text("word " * 50)])]
    ids = [c.chunk_id for c in chunker.chunk_sections(sections, SPEC)]
    assert len(ids) == len(set(ids)) == 2


# --------------------------------------------------------------------------- real chunks.jsonl


@pytest.fixture(scope="module")
def real_chunks() -> list[dict]:
    path = resolve_path(load_config()["paths"]["chunks_file"])
    if not path.exists():
        pytest.skip("chunks.jsonl not built (run `make ingest`)")
    return load_chunks(path)


def test_real_chunks_metadata(real_chunks: list[dict]) -> None:
    for c in real_chunks:
        for key in ("spec", "release", "section", "title", "chunk_id", "text", "text_for_index"):
            assert c[key], f"{c['chunk_id']}: empty {key}"
    assert len({c["chunk_id"] for c in real_chunks}) == len(real_chunks)


def test_real_chunks_size_limits(real_chunks: list[dict]) -> None:
    cfg = load_config()["chunking"]
    for c in real_chunks:
        limit = cfg["max_table_tokens"] if c["chunk_type"] == "table" else cfg["max_tokens"]
        assert c["n_tokens"] <= limit, c["chunk_id"]


def test_real_table_chunks_have_complete_rows(real_chunks: list[dict]) -> None:
    for c in real_chunks:
        if c["chunk_type"] != "table":
            continue
        for line in c["text"].splitlines():
            if line.startswith("|"):
                assert line.endswith("|"), f"{c['chunk_id']}: partial row {line[:60]!r}"


def test_real_known_chunks(real_chunks: list[dict]) -> None:
    by_section = {(c["spec"], c["section"]) for c in real_chunks}
    assert ("TS 24.501", "5.5.1.2.2") in by_section
    assert ("TS 24.501", "A.1") in by_section
    assert ("TS 23.501", "6.2.1") in by_section
    cause9 = [c for c in real_chunks if c["spec"] == "TS 24.501" and c["section"] == "A.1"]
    assert any("Cause #9" in c["text"] for c in cause9)
