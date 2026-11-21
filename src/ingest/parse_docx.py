"""Parse a 3GPP specification (.docx) into an ordered list of sections.

Usage:
    python -m src.ingest.parse_docx [--spec 24501] [-n 30]

The document body is walked **in order** (paragraphs and tables). Headings are detected from
the paragraph style (``Heading 1``..``Heading 9``, and the 3GPP styles ``H1``..``H9``), with the
section number and title read from the heading text. The heading level is derived from the
section number (``5.5.1.2`` -> 4, annex ``A`` -> 1, ``A.1`` -> 2), because 3GPP annex
sub-clauses use ``Heading 1`` regardless of their depth.

Skipped: cover page and table of contents, Foreword, clause 2 (References) and the
"Change history" annex. Figures are lost; their captions are kept as text.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import docx
import docx.enum.style
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.config import get_spec, load_config, resolve_path

logger = logging.getLogger(__name__)

# Heading paragraph styles: Word "Heading N" and 3GPP custom "HN" (e.g. H6, H2).
HEADING_STYLE_RE = re.compile(r"^(?:Heading\s?|H)(\d)$")

# Section number + title, e.g. "5.5.1.2.2\tInitial registration", "5.20b.1 ...", "D.1 ...".
# Fallback detection on the text, as described in the implementation plan.
SECTION_RE = re.compile(
    r"^((?:\d+[A-Za-z]?)(?:\.\d+[A-Za-z]?)*|[A-Z](?:\.\d+[A-Za-z]?)+)\s+(\S.*)$", re.S
)

# Annex heading, e.g. "Annex A (informative):\nCause values for 5GS mobility management".
ANNEX_RE = re.compile(r"^Annex\s+([A-Z])\s*\(([^)]*)\)\s*:?\s*(.*)$", re.S)

# Caption paragraph style for tables ("TH" = table/figure header in 3GPP templates).
TABLE_CAPTION_STYLE = "TH"

SKIPPED_TOP_SECTIONS = {"2"}  # References
SKIPPED_TITLE_PATTERNS = (re.compile(r"change history", re.I),)


@dataclass
class Block:
    """A unit of content inside a section."""

    type: str  # "text" or "table"
    content: str
    style: str = ""
    caption: str = ""  # tables only: the "Table x.y: ..." caption, if any

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "content": self.content, "style": self.style, "caption": self.caption}


@dataclass
class Section:
    """A section of a specification and the content that directly belongs to it."""

    spec: str
    section: str
    title: str
    level: int
    section_path: str
    annex_type: str = ""  # "normative" / "informative" for annex headings
    blocks: list[Block] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "section": self.section,
            "title": self.title,
            "level": self.level,
            "section_path": self.section_path,
            "annex_type": self.annex_type,
            "blocks": [b.to_dict() for b in self.blocks],
        }


# --------------------------------------------------------------------------- text helpers


def clean_text(text: str) -> str:
    """Normalise whitespace: NBSP -> space, tabs -> space, collapse runs, strip."""
    text = text.replace("\xa0", " ").replace("‑", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def section_level(number: str) -> int:
    """Return the depth of a section number (``5.5.1`` -> 3, ``A`` -> 1, ``A.1`` -> 2)."""
    return number.count(".") + 1


def parse_heading(text: str) -> tuple[str, str, str] | None:
    """Parse heading text into ``(section, title, annex_type)``; ``None`` if unnumbered."""
    raw = text.strip()
    m = ANNEX_RE.match(raw)
    if m:
        return m.group(1), clean_text(m.group(3)).replace("\n", " "), m.group(2).strip().lower()
    m = SECTION_RE.match(raw)
    if m:
        return m.group(1), clean_text(m.group(2)).replace("\n", " "), ""
    return None


def heading_style_level(style_name: str) -> int | None:
    """Return N for heading styles ``Heading N`` / ``HN``, else ``None``."""
    m = HEADING_STYLE_RE.match(style_name or "")
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- tables


def _cell_text(tc: Any, doc: DocxDocument) -> str:
    """Text of a table cell, including nested tables, with line breaks as ``<br>``."""
    parts = [clean_text(Paragraph(p, doc).text).replace("\n", "<br>") for p in tc.iter(qn("w:p"))]
    text = "<br>".join(p for p in parts if p)
    return text.replace("|", "\\|")


def table_to_rows(table: Table, doc: DocxDocument) -> list[list[str]]:
    """Convert a table to a list of rows of cell strings.

    Merged cells: a vertically merged value is repeated in every row it spans (so each row is
    self-contained). A horizontally spanned value is written in its first column only, the
    other spanned columns stay empty (bit-layout tables would otherwise repeat a value 8 times).
    """
    rows: list[list[str]] = []
    for row in table.rows:
        out: list[str] = []
        prev_tc = None
        for cell in row.cells:
            if cell._tc is prev_tc:  # same cell spanning several grid columns
                out.append("")
            else:
                out.append(_cell_text(cell._tc, doc))
            prev_tc = cell._tc
        rows.append(out)
    return rows


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render rows as a Markdown table; the first row is the header row."""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- body walk


def iter_block_items(doc: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables of the document body in document order.

    Content controls (``w:sdt``) are descended into so that their paragraphs are not lost.
    """

    def walk(parent: Any) -> Iterator[Paragraph | Table]:
        for child in parent.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, doc)
            elif child.tag == qn("w:tbl"):
                yield Table(child, doc)
            elif child.tag == qn("w:sdt"):
                content = child.find(qn("w:sdtContent"))
                if content is not None:
                    yield from walk(content)

    yield from walk(doc.element.body)


def build_style_map(doc: DocxDocument) -> dict[str, str]:
    """Map paragraph style ids to style names (python-docx's ``Paragraph.style`` is slow)."""
    return {st.style_id: st.name for st in doc.styles if st.style_id}


def style_name(paragraph: Paragraph, style_map: dict[str, str], default: str = "Normal") -> str:
    """Return the style name of a paragraph using a precomputed id -> name map."""
    ppr = paragraph._p.pPr
    if ppr is None or ppr.pStyle is None:
        return default
    sid = ppr.pStyle.val
    return style_map.get(sid, sid)


def _is_skipped(section: str, title: str) -> bool:
    top = section.split(".")[0]
    if top in SKIPPED_TOP_SECTIONS:
        return True
    return any(p.search(title) for p in SKIPPED_TITLE_PATTERNS)


def parse_docx(path: str | Path, spec_id: str) -> list[Section]:
    """Parse a 3GPP .docx into an ordered list of :class:`Section`.

    Args:
        path: Path to the .docx file.
        spec_id: Spec identifier stored on every section, e.g. ``"TS 24.501"``.

    Returns:
        Sections in document order. Each section only holds the blocks that appear between its
        heading and the next heading (sub-sections are separate entries).
    """
    doc = docx.Document(str(path))
    style_map = build_style_map(doc)
    default_style = doc.styles.default(docx.enum.style.WD_STYLE_TYPE.PARAGRAPH)
    default_name = default_style.name if default_style is not None else "Normal"
    sections: list[Section] = []
    stack: list[Section] = []  # current heading path
    current: Section | None = None  # None while in front matter or a skipped subtree
    skip_level: int | None = None  # level of the skipped subtree root, if any
    pending_caption: str | None = None  # TH caption waiting for its table

    for item in iter_block_items(doc):
        if isinstance(item, Table):
            if current is not None:
                md = rows_to_markdown(table_to_rows(item, doc))
                if md:
                    current.blocks.append(
                        Block("table", md, style="table", caption=pending_caption or "")
                    )
            pending_caption = None
            continue

        style = style_name(item, style_map, default_name)
        if style.lower().startswith("toc"):
            continue
        text = item.text

        if heading_style_level(style) is not None and text.strip():
            if pending_caption and current is not None:
                current.blocks.append(Block("text", pending_caption, style=TABLE_CAPTION_STYLE))
            pending_caption = None
            parsed = parse_heading(text)
            if parsed is None:
                # Unnumbered heading (Foreword): skip its content until the next numbered heading.
                logger.debug("Skipping unnumbered heading %r", clean_text(text))
                current, skip_level = None, 0
                continue
            number, title, annex_type = parsed
            level = section_level(number)
            if skip_level is not None and skip_level and level > skip_level:
                continue  # still inside a skipped subtree (e.g. 2.x)
            skip_level = None
            while stack and stack[-1].level >= level:
                stack.pop()
            if _is_skipped(number, title):
                current, skip_level = None, level
                stack = []
                continue
            path = " > ".join([s.section for s in stack] + [number])
            current = Section(spec_id, number, title, level, path, annex_type)
            sections.append(current)
            stack.append(current)
            continue

        if current is None:
            continue
        content = clean_text(text)
        if not content:
            continue
        if pending_caption is not None:
            current.blocks.append(Block("text", pending_caption, style=TABLE_CAPTION_STYLE))
            pending_caption = None
        if style == TABLE_CAPTION_STYLE:
            pending_caption = content  # attach to the table that follows, if any
        else:
            current.blocks.append(Block("text", content, style=style))

    if pending_caption and current is not None:
        current.blocks.append(Block("text", pending_caption, style=TABLE_CAPTION_STYLE))
    _warn_duplicates(sections)
    logger.info("Parsed %s: %d sections", spec_id, len(sections))
    return sections


def _warn_duplicates(sections: list[Section]) -> None:
    seen: set[str] = set()
    for s in sections:
        if s.section in seen:
            logger.warning("%s: duplicate section number %s (%s)", s.spec, s.section, s.title)
        seen.add(s.section)


def parse_spec(spec_id: str, cfg: dict[str, Any] | None = None) -> list[Section]:
    """Parse the .docx of a spec listed in ``config.yaml`` (e.g. ``"TS 24.501"`` or ``"24501"``)."""
    cfg = cfg or load_config()
    spec = get_spec(cfg, spec_id)
    docx_path = resolve_path(cfg["paths"]["raw_dir"]) / (Path(spec["file"]).stem + ".docx")
    if not docx_path.exists():
        raise FileNotFoundError(f"{docx_path} not found; run `make download` first")
    return parse_docx(docx_path, spec["id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse 3GPP .docx specs and print detected sections")
    parser.add_argument("--spec", help="spec id or short id (default: all specs in config)")
    parser.add_argument("-n", type=int, default=30, help="number of sections to print per spec")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config()
    spec_ids = [args.spec] if args.spec else [s["id"] for s in cfg["corpus"]["specs"]]
    for spec_id in spec_ids:
        sections = parse_spec(spec_id, cfg)
        n_tables = sum(1 for s in sections for b in s.blocks if b.type == "table")
        n_annex = sum(1 for s in sections if s.section[0].isalpha())
        print(f"\n=== {sections[0].spec}: {len(sections)} sections, {n_tables} tables, "
              f"{n_annex} annex sections ===")
        print(f"{'section':<14} {'lvl':>3}  title")
        for s in sections[: args.n]:
            print(f"{s.section:<14} {s.level:>3}  {s.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
