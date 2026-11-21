"""Tests for src.ingest.parse_docx.

Unit tests build small synthetic .docx files. Tests on the real specs are skipped when the
documents have not been downloaded (`make download`).
"""

from __future__ import annotations

from pathlib import Path

import docx
import pytest
from docx.enum.style import WD_STYLE_TYPE

from src.config import load_config, resolve_path
from src.ingest.parse_docx import (
    Section,
    parse_docx,
    parse_heading,
    parse_spec,
    rows_to_markdown,
    section_level,
)

# --------------------------------------------------------------------------- unit tests


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5.5.1.2.2\tInitial registration initiation", ("5.5.1.2.2", "Initial registration initiation", "")),
        ("5.20b.1\tGroup attribute provisioning", ("5.20b.1", "Group attribute provisioning", "")),
        ("5.4.1.2.3A.1\tGeneral", ("5.4.1.2.3A.1", "General", "")),
        ("A.1\tCauses related to UE identification", ("A.1", "Causes related to UE identification", "")),
        ("1\tScope", ("1", "Scope", "")),
        (
            "Annex A (informative):\nCause values for 5GS mobility management",
            ("A", "Cause values for 5GS mobility management", "informative"),
        ),
        ("Foreword", None),
    ],
)
def test_parse_heading(text: str, expected: tuple[str, str, str] | None) -> None:
    assert parse_heading(text) == expected


def test_section_level() -> None:
    assert section_level("5") == 1
    assert section_level("5.5.1.2.2") == 5
    assert section_level("A") == 1
    assert section_level("D.1") == 2


def test_rows_to_markdown_pads_and_escapes() -> None:
    md = rows_to_markdown([["h1", "h2"], ["a"], ["", ""], ["b", "c"]])
    assert md.splitlines() == ["| h1 | h2 |", "|---|---|", "| a |  |", "| b | c |"]


@pytest.fixture()
def synthetic_docx(tmp_path: Path) -> Path:
    d = docx.Document()
    d.styles.add_style("TH", WD_STYLE_TYPE.PARAGRAPH)
    d.add_paragraph("Contents")
    d.add_heading("Foreword", level=1)
    d.add_paragraph("Foreword text that must be skipped.")
    d.add_heading("1\tScope", level=1)
    d.add_paragraph("Scope text.")
    d.add_heading("2\tReferences", level=1)
    d.add_paragraph("Reference list that must be skipped.")
    d.add_heading("2.1\tSub reference", level=2)
    d.add_paragraph("Also skipped.")
    d.add_heading("3\tDefinitions and abbreviations", level=1)
    d.add_heading("3.2\tAbbreviations", level=2)
    d.add_paragraph("AMF\tAccess and Mobility Management Function")
    d.add_heading("Annex A (informative):\nCause values", level=8)
    d.add_heading("A.1\tCauses related to UE identification", level=1)
    d.add_paragraph("Cause #3 – Illegal UE")
    d.add_paragraph("Table A.1.1: Example table", style="TH")
    t = d.add_table(rows=3, cols=3)
    t.cell(0, 0).text, t.cell(0, 1).text, t.cell(0, 2).text = "Cause", "Name", "Note"
    t.cell(1, 0).text, t.cell(1, 1).text = "#3", "Illegal UE"
    t.cell(1, 2).merge(t.cell(2, 2)).text = "vertical"
    t.cell(2, 0).merge(t.cell(2, 1)).text = "#9 spans two columns"
    d.add_heading("Annex B (informative):\nChange history", level=8)
    d.add_paragraph("History that must be skipped.")
    path = tmp_path / "synthetic.docx"
    d.save(str(path))
    return path


def test_synthetic_structure(synthetic_docx: Path) -> None:
    sections = parse_docx(synthetic_docx, "TS 99.999")
    numbers = [s.section for s in sections]
    assert numbers == ["1", "3", "3.2", "A", "A.1"]
    by_num = {s.section: s for s in sections}
    assert by_num["3.2"].section_path == "3 > 3.2"
    assert by_num["A.1"].level == 2 and by_num["A.1"].section_path == "A > A.1"
    assert by_num["A"].annex_type == "informative"
    assert by_num["3.2"].blocks[0].content == "AMF Access and Mobility Management Function"
    all_text = " ".join(b.content for s in sections for b in s.blocks)
    assert "skipped" not in all_text


def test_synthetic_table(synthetic_docx: Path) -> None:
    a1 = {s.section: s for s in parse_docx(synthetic_docx, "TS 99.999")}["A.1"]
    tables = [b for b in a1.blocks if b.type == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.caption == "Table A.1.1: Example table"
    lines = table.content.splitlines()
    assert lines[0] == "| Cause | Name | Note |"
    assert lines[2] == "| #3 | Illegal UE | vertical |"
    # vertical merge repeated, horizontal span written once
    assert lines[3] == "| #9 spans two columns |  | vertical |"
    # the caption is not duplicated as a text block
    assert all(b.content != table.caption for b in a1.blocks if b.type == "text")


# --------------------------------------------------------------------------- real specs


def _spec_available(short: str) -> bool:
    cfg = load_config()
    spec = next(s for s in cfg["corpus"]["specs"] if s["short"] == short)
    return (resolve_path(cfg["paths"]["raw_dir"]) / (Path(spec["file"]).stem + ".docx")).exists()


@pytest.fixture(scope="module")
def spec_23501() -> dict[str, Section]:
    if not _spec_available("23501"):
        pytest.skip("TS 23.501 .docx not downloaded")
    return {s.section: s for s in parse_spec("23501")}


@pytest.fixture(scope="module")
def spec_24501() -> dict[str, Section]:
    if not _spec_available("24501"):
        pytest.skip("TS 24.501 .docx not downloaded")
    return {s.section: s for s in parse_spec("24501")}


def test_known_sections_exist(spec_23501: dict[str, Section], spec_24501: dict[str, Section]) -> None:
    assert "6.2.1" in spec_23501
    assert "5.5.1" in spec_24501
    assert spec_24501["5.5.1.2.2"].title == "Initial registration initiation"
    assert spec_24501["5.5.1.2.2"].section_path == "5 > 5.5 > 5.5.1 > 5.5.1.2 > 5.5.1.2.2"
    assert spec_24501["9.11.3.2"].title == "5GMM cause"


def test_skipped_and_kept_clauses(spec_23501: dict[str, Section], spec_24501: dict[str, Section]) -> None:
    for secs in (spec_23501, spec_24501):
        assert "1" in secs and "3" in secs and "3.2" in secs
        assert not any(n == "2" or n.startswith("2.") for n in secs)
        assert not any("change history" in s.title.lower() for s in secs.values())


def test_annexes(spec_24501: dict[str, Section]) -> None:
    assert spec_24501["A"].title == "Cause values for 5GS mobility management"
    assert spec_24501["A.1"].level == 2
    assert spec_24501["A.1"].section_path == "A > A.1"
    text = " ".join(b.content for b in spec_24501["A.1"].blocks)
    assert "Cause #9" in text


def test_letter_suffix_sections(spec_23501: dict[str, Section], spec_24501: dict[str, Section]) -> None:
    assert "5.20b.1" in spec_23501
    assert "5.4.1.2.2.6A" in spec_24501


def test_tables_parsed(spec_24501: dict[str, Section]) -> None:
    tables = [b for b in spec_24501["9.11.3.2"].blocks if b.type == "table"]
    cause_table = next(t for t in tables if t.caption.startswith("Table 9.11.3.2.1"))
    assert "UE identity cannot be derived by the network" in cause_table.content
    assert cause_table.content.startswith("| ")


def test_every_section_has_metadata(spec_23501: dict[str, Section], spec_24501: dict[str, Section]) -> None:
    for secs in (spec_23501, spec_24501):
        for s in secs.values():
            assert s.spec and s.section and s.title and s.level >= 1 and s.section_path
