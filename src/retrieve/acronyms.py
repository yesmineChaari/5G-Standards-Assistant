"""Acronym dictionary from clause 3.2 (Abbreviations) of the specs, and query expansion.

Usage:
    python -m src.retrieve.acronyms build            # writes data/processed/acronyms.json
    python -m src.retrieve.acronyms expand "What is the role of the NRF?"

The dictionary is read from the .docx (abbreviation paragraphs ``<ACRONYM>\\t<expansion>`` in
clause 3.2, styles EW/EX), because the tab is the only reliable separator for multi-word
acronyms such as "5G LAN". Expansion appends ``(NRF: Network Repository Function)`` for every
acronym found in the query.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import docx

from src.config import get_spec, load_config, resolve_path
from src.ingest.parse_docx import (
    build_style_map,
    clean_text,
    heading_style_level,
    iter_block_items,
    parse_heading,
    style_name,
)
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)

ABBREVIATION_STYLES = {"EW", "EX"}
ABBREVIATIONS_SECTION = "3.2"
MAX_EXPANSIONS = 8


def read_abbreviations(docx_path: str | Path) -> dict[str, str]:
    """Return ``{acronym: expansion}`` from clause 3.2 of a 3GPP .docx."""
    doc = docx.Document(str(docx_path))
    style_map = build_style_map(doc)
    inside = False
    out: dict[str, str] = {}
    for item in iter_block_items(doc):
        if not isinstance(item, Paragraph):
            continue
        style = style_name(item, style_map)
        if heading_style_level(style) is not None:
            parsed = parse_heading(item.text)
            if inside:
                break
            inside = parsed is not None and parsed[0] == ABBREVIATIONS_SECTION
            continue
        if not inside or style not in ABBREVIATION_STYLES:
            continue
        parts = [p for p in item.text.split("\t") if p.strip()]
        if len(parts) != 2:
            continue
        acronym, expansion = clean_text(parts[0]), clean_text(parts[1])
        if acronym and expansion and acronym != expansion:
            out[acronym] = expansion
    return out


def build_acronyms(cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Merge the abbreviations of all specs: ``{acronym: [expansion, ...]}`` (distinct, ordered)."""
    cfg = cfg or load_config()
    merged: dict[str, list[str]] = {}
    for spec in cfg["corpus"]["specs"]:
        path = resolve_path(cfg["paths"]["raw_dir"]) / (Path(get_spec(cfg, spec["id"])["file"]).stem + ".docx")
        entries = read_abbreviations(path)
        logger.info("%s: %d abbreviations", spec["id"], len(entries))
        for acronym, expansion in entries.items():
            lst = merged.setdefault(acronym, [])
            if _norm(expansion) not in (_norm(e) for e in lst):
                lst.append(expansion)
    return dict(sorted(merged.items()))


def _norm(expansion: str) -> str:
    """Comparison key ignoring case, spacing, hyphens and a final plural 's'."""
    key = re.sub(r"[^a-z0-9]", "", expansion.lower())
    return key[:-1] if key.endswith("s") else key


def acronyms_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg["paths"].get("acronyms_file", "data/processed/acronyms.json"))


def load_acronyms(cfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Load the acronym dictionary, building and saving it first if needed."""
    cfg = cfg or load_config()
    path = acronyms_path(cfg)
    if not path.exists():
        save_acronyms(build_acronyms(cfg), path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_acronyms(acronyms: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(acronyms, indent=1, ensure_ascii=False), encoding="utf-8")


class AcronymExpander:
    """Find known acronyms in a query (case-sensitive, whole tokens) and append their expansions."""

    def __init__(self, acronyms: dict[str, list[str]], max_expansions: int = MAX_EXPANSIONS) -> None:
        self.acronyms = {k: v for k, v in acronyms.items() if len(k) >= 2}
        self.max_expansions = max_expansions
        # Longest first so that "5G-GUTI" wins over "GUTI"; tokens delimited by non [A-Za-z0-9-].
        keys = sorted(self.acronyms, key=len, reverse=True)
        alternation = "|".join(re.escape(k) for k in keys)
        self._re = re.compile(rf"(?<![A-Za-z0-9\-])({alternation})(?![A-Za-z0-9\-])") if keys else None

    def find(self, query: str) -> list[str]:
        """Acronyms present in the query, in order of first appearance (no duplicates)."""
        if self._re is None:
            return []
        found: list[str] = []
        for m in self._re.finditer(query):
            if m.group(1) not in found:
                found.append(m.group(1))
        return found[: self.max_expansions]

    def expand(self, query: str) -> tuple[str, list[str]]:
        """Return ``(expanded_query, acronyms_found)``; the query is unchanged if none is found."""
        found = self.find(query)
        if not found:
            return query, []
        parts = [f"{a}: {' / '.join(self.acronyms[a])}" for a in found]
        return f"{query} ({'; '.join(parts)})", found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acronym dictionary from clause 3.2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("expand")
    e.add_argument("query")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    if args.cmd == "build":
        acronyms = build_acronyms(cfg)
        save_acronyms(acronyms, acronyms_path(cfg))
        multi = sum(1 for v in acronyms.values() if len(v) > 1)
        print(f"{len(acronyms)} acronyms ({multi} with several expansions) -> {acronyms_path(cfg)}")
    else:
        expanded, found = AcronymExpander(load_acronyms(cfg)).expand(args.query)
        print(f"found: {found}\nexpanded: {expanded}")
    return 0


if __name__ == "__main__":
    from src.retrieve.acronyms import main as _main

    raise SystemExit(_main())
