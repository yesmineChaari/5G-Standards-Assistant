"""Structural chunking of parsed 3GPP sections, with a recursive fallback for long content.

Usage:
    python -m src.ingest.chunker            # parse all specs and write data/processed/chunks.jsonl

Rules (parameters from ``config.yaml`` -> ``chunking``):
  * One chunk per section by default (the content between a heading and the next heading).
  * Text longer than ``max_tokens`` is split at paragraph boundaries, greedily packing paragraphs,
    with ``overlap_tokens`` of overlap from the previous chunk. A paragraph longer than
    ``max_tokens`` is split at sentence boundaries.
  * Tables: a row is never split. A table up to ``max_table_tokens`` is one ``table`` chunk;
    larger tables are split by row groups, repeating the header row (and caption) in each part.
  * Inside a section, a text run shorter than ``min_tokens`` that sits next to a table is folded
    into that table chunk (e.g. the sentence introducing an information element).
  * A text-only section shorter than ``min_tokens`` is merged with the following sibling
    section(s). The chunk keeps the first section number in ``section`` and lists all merged
    numbers in ``merged_sections``; each merged part starts with a ``§<number> <title>`` line.
  * The text used for embeddings/BM25 starts with a header line
    ``[TS 24.501 | Rel-18 | §5.5.1.2.2 | Initial registration initiation]``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import tiktoken

from src.config import load_config, resolve_path
from src.ingest.parse_docx import Block, Section, parse_spec

logger = logging.getLogger(__name__)

# Sentence boundary: end punctuation followed by whitespace and a plausible sentence start.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z0-9(\"'\-])")


@dataclass
class Chunk:
    """One retrievable unit. Serialised as one JSON line in ``chunks.jsonl``."""

    chunk_id: str
    spec: str
    version: str
    release: str
    section: str
    title: str
    section_path: str
    merged_sections: list[str]
    chunk_type: str  # "text" or "table"
    part: int
    n_tokens: int
    text_for_index: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Piece:
    """Intermediate chunk content before ids/headers are assigned."""

    kind: str  # "text" or "table"
    text: str
    sections: list[Section] = field(default_factory=list)


class TokenCounter:
    """Token counting/truncation with a tiktoken encoding."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def tail(self, text: str, n_tokens: int) -> str:
        """Last ``n_tokens`` tokens of ``text``, trimmed to start on a word boundary."""
        ids = self._enc.encode(text, disallowed_special=())
        if len(ids) <= n_tokens:
            return text
        out = self._enc.decode(ids[-n_tokens:])
        return out.split(" ", 1)[1] if " " in out else out

    def head(self, text: str, n_tokens: int) -> str:
        ids = self._enc.encode(text, disallowed_special=())
        return self._enc.decode(ids[:n_tokens])


class Chunker:
    """Turn parsed :class:`Section` objects into :class:`Chunk` objects."""

    def __init__(
        self,
        max_tokens: int = 450,
        overlap_tokens: int = 60,
        min_tokens: int = 40,
        max_table_tokens: int = 800,
        tokenizer: str = "cl100k_base",
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.max_table_tokens = max_table_tokens
        self.tok = TokenCounter(tokenizer)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Chunker":
        c = cfg["chunking"]
        return cls(
            max_tokens=c["max_tokens"],
            overlap_tokens=c["overlap_tokens"],
            min_tokens=c["min_tokens"],
            max_table_tokens=c["max_table_tokens"],
            tokenizer=c.get("tokenizer", "cl100k_base"),
        )

    # ------------------------------------------------------------------ text splitting

    def split_sentences(self, paragraph: str) -> list[str]:
        """Split a paragraph into sentences; hard-split sentences longer than ``max_tokens``."""
        out: list[str] = []
        for sent in SENTENCE_SPLIT_RE.split(paragraph):
            sent = sent.strip()
            if not sent:
                continue
            while self.tok.count(sent) > self.max_tokens:
                head = self.tok.head(sent, self.max_tokens)
                cut = head.rfind(" ")
                head = head[:cut] if cut > len(head) // 2 else head
                out.append(head.strip())
                sent = sent[len(head):].strip()
            if sent:
                out.append(sent)
        return out

    def _units(self, paragraphs: list[str]) -> list[tuple[str, str]]:
        """Packing units ``(text, joiner)``: paragraphs, or sentences of over-long paragraphs."""
        units: list[tuple[str, str]] = []
        for para in paragraphs:
            if self.tok.count(para) <= self.max_tokens:
                units.append((para, "\n"))
            else:
                sents = self.split_sentences(para)
                units += [(s, "\n" if i == 0 else " ") for i, s in enumerate(sents)]
        return units

    def _overlap(self, units: list[str]) -> str:
        """Tail of the previous chunk (whole sentences when possible) up to ``overlap_tokens``."""
        if self.overlap_tokens <= 0 or not units:
            return ""
        sentences = SENTENCE_SPLIT_RE.split(units[-1].strip())
        picked: list[str] = []
        total = 0
        for sent in reversed(sentences):
            n = self.tok.count(sent)
            if total + n > self.overlap_tokens:
                break
            picked.insert(0, sent)
            total += n
        if picked:
            return " ".join(picked)
        return self.tok.tail(units[-1], self.overlap_tokens)

    def pack_text(self, paragraphs: list[str]) -> list[str]:
        """Greedily pack paragraphs into chunks of at most ``max_tokens`` with overlap."""
        text = "\n".join(paragraphs)
        if self.tok.count(text) <= self.max_tokens:
            return [text] if text.strip() else []
        chunks: list[str] = []
        current: list[str] = []
        current_text = ""
        for unit, joiner in self._units(paragraphs):
            candidate = f"{current_text}{joiner}{unit}" if current_text else unit
            if current and self.tok.count(candidate) > self.max_tokens:
                chunks.append(current_text)
                overlap = self._overlap(current)
                current = [overlap, unit] if overlap else [unit]
                current_text = f"{overlap}\n{unit}" if overlap else unit
                if self.tok.count(current_text) > self.max_tokens:  # overlap does not fit
                    current, current_text = [unit], unit
            else:
                current.append(unit)
                current_text = candidate
        if current_text.strip():
            chunks.append(current_text)
        return chunks

    # ------------------------------------------------------------------ tables

    @staticmethod
    def _row_cells(row: str) -> list[str]:
        inner = row.strip()
        inner = inner[1:] if inner.startswith("|") else inner
        inner = inner[:-1] if inner.endswith("|") else inner
        return [c.strip() for c in re.split(r"(?<!\\)\|", inner)]

    def _prose_row(self, row: str, caption: str) -> list[tuple[str, str]]:
        """Chunk an over-long single-cell row (prose stored in a table) as text pieces."""
        cell = next(c for c in self._row_cells(row) if c)
        paragraphs = [p.replace("\\|", "|").strip() for p in cell.split("<br>") if p.strip()]
        label = f"{caption} (continued)" if caption else "Table (continued)"
        budget_cfg = self.max_tokens - self.tok.count(label) - 1
        saved = self.max_tokens
        self.max_tokens = budget_cfg  # pack so that label + text fits in max_tokens
        try:
            parts = self.pack_text(paragraphs)
        finally:
            self.max_tokens = saved
        return [("text", f"{label}\n{p}") for p in parts]

    def _is_prose_row(self, row: str, budget: int) -> bool:
        return self.tok.count(row) > budget and sum(1 for c in self._row_cells(row) if c) == 1

    def split_table(self, block: Block) -> list[tuple[str, str]]:
        """Split a Markdown table into parts of at most ``max_table_tokens`` without splitting rows.

        Returns ``(kind, text)`` pairs. Each table part repeats the caption and the header row
        (+ separator). A row that is longer than the budget and has a single non-empty cell is
        prose stored in a table (e.g. IE field descriptions); it is chunked as text, each piece
        prefixed with the caption. Over-long rows with several cells are kept whole.
        """
        lines = block.content.splitlines()
        caption = block.caption.strip()
        cap = [caption] if caption else []
        full = "\n".join(cap + lines)
        if self.tok.count(full) <= self.max_table_tokens:
            return [("table", full)]
        if len(lines) <= 2:  # header-only table
            if self._is_prose_row(lines[0], self.max_table_tokens):
                return self._prose_row(lines[0], caption)
            logger.warning("Over-long table kept whole (%s)", caption or lines[0][:60])
            return [("table", full)]
        if self.tok.count(lines[0]) > self.max_table_tokens // 4:
            # Header row too large to repeat (e.g. prose in the first row): treat it as a body row.
            prefix = "\n".join(cap)
            body = [lines[0]] + lines[2:]
        else:
            prefix = "\n".join(cap + lines[:2])
            body = lines[2:]
        budget = self.max_table_tokens - self.tok.count(prefix) - 1
        parts: list[tuple[str, str]] = []
        group: list[str] = []
        group_tokens = 0

        def flush() -> None:
            nonlocal group, group_tokens
            if group:
                parts.append(("table", "\n".join(([prefix] if prefix else []) + group)))
                group, group_tokens = [], 0

        for row in body:
            n = self.tok.count(row) + 1
            if n > budget and self._is_prose_row(row, budget):
                flush()
                parts += self._prose_row(row, caption)
                continue
            if group and group_tokens + n > budget:
                flush()
            if n > budget:
                logger.warning("Table row longer than max_table_tokens kept whole (%s)", caption or row[:60])
            group.append(row)
            group_tokens += n
        flush()
        return parts

    # ------------------------------------------------------------------ section level

    def section_pieces(self, section: Section) -> list[_Piece]:
        """Split one section's blocks into text/table pieces (no cross-section merging)."""
        segments: list[tuple[str, Any]] = []  # ("text", [paragraphs]) | ("table", Block)
        for block in section.blocks:
            if block.type == "table":
                segments.append(("table", block))
            elif segments and segments[-1][0] == "text":
                segments[-1][1].append(block.content)
            else:
                segments.append(("text", [block.content]))

        pieces: list[_Piece] = []
        pending_small_text = ""  # short text waiting to be folded into the next table
        for i, (kind, payload) in enumerate(segments):
            if kind == "text":
                text = "\n".join(payload)
                has_table = any(k == "table" for k, _ in segments)
                if has_table and self.tok.count(text) < self.min_tokens:
                    # Prefer the preceding table (figure captions follow their figure),
                    # otherwise the following one (introductory sentence before a table).
                    prev_is_table = i > 0 and segments[i - 1][0] == "table" and pieces
                    if prev_is_table:
                        limit = self.max_table_tokens if pieces[-1].kind == "table" else self.max_tokens
                        merged = f"{pieces[-1].text}\n{text}"
                        if self.tok.count(merged) <= limit:
                            pieces[-1].text = merged
                            continue
                    next_is_table = i + 1 < len(segments) and segments[i + 1][0] == "table"
                    if next_is_table:
                        pending_small_text = text
                        continue
                for t in self.pack_text(payload):
                    pieces.append(_Piece("text", t, [section]))
            else:
                parts = self.split_table(payload)
                if pending_small_text:
                    kind0, text0 = parts[0]
                    limit = self.max_table_tokens if kind0 == "table" else self.max_tokens
                    candidate = f"{pending_small_text}\n{text0}"
                    if self.tok.count(candidate) <= limit:
                        parts[0] = (kind0, candidate)
                    else:
                        pieces.append(_Piece("text", pending_small_text, [section]))
                    pending_small_text = ""
                pieces += [_Piece(k, p, [section]) for k, p in parts]
        if pending_small_text:
            pieces.append(_Piece("text", pending_small_text, [section]))
        return pieces

    @staticmethod
    def _parent(section: Section) -> str:
        return section.section_path.rsplit(" > ", 1)[0] if " > " in section.section_path else ""

    def _is_small_text_section(self, section: Section, pieces: list[_Piece]) -> bool:
        return (
            len(pieces) == 1
            and pieces[0].kind == "text"
            and self.tok.count(pieces[0].text) < self.min_tokens
        )

    @staticmethod
    def _labelled(section: Section, text: str) -> str:
        return f"§{section.section} {section.title}\n{text}"

    def merge_small_sections(self, items: list[tuple[Section, list[_Piece]]]) -> list[_Piece]:
        """Merge short text-only sections with their following sibling sections."""
        out: list[_Piece] = []
        i = 0
        while i < len(items):
            section, pieces = items[i]
            if not self._is_small_text_section(section, pieces):
                out += pieces
                i += 1
                continue
            group = [(section, pieces[0].text)]
            total = self.tok.count(self._labelled(section, pieces[0].text))
            j = i + 1
            while total < self.min_tokens and j < len(items):
                nxt, nxt_pieces = items[j]
                if self._parent(nxt) != self._parent(section):
                    break
                if len(nxt_pieces) != 1 or nxt_pieces[0].kind != "text":
                    break
                labelled = self._labelled(nxt, nxt_pieces[0].text)
                n = self.tok.count(labelled)
                if total + n > self.max_tokens:
                    break
                group.append((nxt, nxt_pieces[0].text))
                total += n
                j += 1
            if len(group) == 1:
                out += pieces
            else:
                text = "\n".join(self._labelled(s, t) for s, t in group)
                out.append(_Piece("text", text, [s for s, _ in group]))
            i = j
        return out

    # ------------------------------------------------------------------ public API

    def chunk_sections(self, sections: Iterable[Section], spec_cfg: dict[str, Any]) -> list[Chunk]:
        """Chunk the sections of one spec.

        Args:
            sections: Parsed sections of one spec, in document order.
            spec_cfg: The spec entry from ``config.yaml`` (id, short, version, release).
        """
        items = [(s, self.section_pieces(s)) for s in sections]
        items = [(s, p) for s, p in items if p]
        pieces = self.merge_small_sections(items)

        chunks: list[Chunk] = []
        part_counter: dict[str, int] = defaultdict(int)
        for piece in pieces:
            first = piece.sections[0]
            part_counter[first.section] += 1
            part = part_counter[first.section]
            header = f"[{spec_cfg['id']} | {spec_cfg['release']} | §{first.section} | {first.title}]"
            chunks.append(
                Chunk(
                    chunk_id=f"{spec_cfg['short']}_{first.section}_{part:03d}",
                    spec=spec_cfg["id"],
                    version=spec_cfg["version"],
                    release=spec_cfg["release"],
                    section=first.section,
                    title=first.title,
                    section_path=first.section_path,
                    merged_sections=[s.section for s in piece.sections] if len(piece.sections) > 1 else [],
                    chunk_type=piece.kind,
                    part=part,
                    n_tokens=self.tok.count(piece.text),
                    text_for_index=f"{header}\n{piece.text}",
                    text=piece.text,
                )
            )
        return chunks


def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    """Write chunks as JSON lines (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load ``chunks.jsonl`` (default path from config) as a list of dicts."""
    if path is None:
        path = resolve_path(load_config()["paths"]["chunks_file"])
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_chunks(cfg: dict[str, Any] | None = None) -> list[Chunk]:
    """Parse and chunk every spec listed in the config."""
    cfg = cfg or load_config()
    chunker = Chunker.from_config(cfg)
    all_chunks: list[Chunk] = []
    for spec in cfg["corpus"]["specs"]:
        sections = parse_spec(spec["id"], cfg)
        chunks = chunker.chunk_sections(sections, spec)
        logger.info("%s: %d sections -> %d chunks", spec["id"], len(sections), len(chunks))
        all_chunks += chunks
    ids = [c.chunk_id for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate chunk ids produced")
    return all_chunks


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    chunks = build_chunks(cfg)
    out = resolve_path(cfg["paths"]["chunks_file"])
    write_jsonl(chunks, out)
    logger.info("Wrote %d chunks to %s", len(chunks), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
