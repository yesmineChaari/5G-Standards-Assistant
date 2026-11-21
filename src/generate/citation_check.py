"""Citation extraction and validation.

The prompt asks for citations of the form ``(TS 24.501, §5.5.1.2.2)``. Models also write variants,
all of which are parsed:

    (TS 24.501, §A.1)
    (TS 24.501 §5.20b.1)                              no comma, letter suffixes
    (TS 24.501, §5.6.1.5 and §5.5.1.3.5)              several sections for one spec
    (TS 24.501, §§5.6.1.5, 5.5.1.3.5)
    (TS 24.501, §5.5.1.2.2; TS 23.501, §5.15.2)       several specs, separated by ";"
    (3GPP TS 23.501, §4.2.11)

A citation is **valid** when its (spec, section) equals the ``section`` of a retrieved chunk or one
of its ``merged_sections``. Anything else (another spec, an invented or parent/child section) is
**invalid**: it is reported and logged, never silently removed from the answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

# Section ids in the corpus: "A", "A.1", "5.20b.1", "5.4.1.2.3A.1", "9.11.3.9A".
SECTION = r"(?:[A-Z]|\d+[A-Za-z]?)(?:\.\d+[A-Za-z]?)*(?![A-Za-z0-9])"
SPEC = r"(?:3GPP\s+)?TS\s?(\d{2}\.\d{3})"
DOTTED_SECTION = r"(?:[A-Z]|\d+[A-Za-z]?)(?:\.\d+[A-Za-z]?)+(?![A-Za-z0-9])"
SEPARATOR = r"\s*(?:,|and|or|&|/)?\s*"
# Further sections in the list: either introduced by "§" again, or a dotted id ("5.5.1.3.5").
# A bare number after a comma ("..., 3 timers") is not taken as a section.
NEXT_SECTION = rf"{SEPARATOR}(?:§{{1,2}}\s*{SECTION}|{DOTTED_SECTION})"

SPEC_RE = re.compile(SPEC)
# Spec, optional comma, then a list of sections; the first one must be introduced by "§".
CITATION_RE = re.compile(
    rf"{SPEC}\s*,?\s*(?:clause|section)?\s*§{{1,2}}\s*(?P<list>{SECTION}(?:{NEXT_SECTION})*)"
)
SECTION_RE = re.compile(SECTION)


@dataclass(frozen=True)
class Citation:
    spec: str  # "TS 24.501"
    section: str  # "5.5.1.2.2"

    def label(self) -> str:
        return f"({self.spec}, §{self.section})"


@dataclass
class CitationReport:
    """All citations found in an answer, split into valid and invalid ones."""

    citations: list[Citation] = field(default_factory=list)  # unique, in order of appearance
    valid: list[Citation] = field(default_factory=list)
    invalid: list[Citation] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.citations)

    @property
    def precision(self) -> float | None:
        """Share of cited (spec, section) pairs that are in the retrieved context (None if no citation)."""
        return len(self.valid) / self.n_total if self.citations else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": [asdict(c) for c in self.valid],
            "invalid": [asdict(c) for c in self.invalid],
            "n_total": self.n_total,
            "precision": self.precision,
        }


def normalise_spec(number: str) -> str:
    return f"TS {number}"


def extract_citations(text: str) -> list[Citation]:
    """Return unique citations in order of appearance."""
    found: list[Citation] = []
    for m in CITATION_RE.finditer(text):
        spec = normalise_spec(m.group(1))
        for sec in SECTION_RE.findall(m.group("list")):
            found.append(Citation(spec, sec))
    return list(dict.fromkeys(found))


def context_sections(chunks: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    """(spec, section) pairs covered by the retrieved chunks, including merged sections."""
    pairs: set[tuple[str, str]] = set()
    for chunk in chunks:
        pairs.add((chunk["spec"], chunk["section"]))
        for sec in chunk.get("merged_sections") or []:
            pairs.add((chunk["spec"], sec))
    return pairs


def check_citations(text: str, chunks: Sequence[dict[str, Any]]) -> CitationReport:
    """Validate every citation in ``text`` against the retrieved ``chunks``."""
    allowed = context_sections(chunks)
    report = CitationReport(citations=extract_citations(text))
    for cit in report.citations:
        (report.valid if (cit.spec, cit.section) in allowed else report.invalid).append(cit)
    if report.invalid:
        logger.warning("Invalid citations (not in retrieved context): %s",
                       ", ".join(c.label() for c in report.invalid))
    return report


def has_spec_mention(text: str) -> bool:
    """True if the text names a spec at all (used to tell 'no citation' from 'unparsed citation')."""
    return bool(SPEC_RE.search(text))
