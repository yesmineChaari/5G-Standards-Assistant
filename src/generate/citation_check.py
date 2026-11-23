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


def citation_spans(text: str) -> list[tuple[int, int, list[Citation]]]:
    """``(start, end, citations)`` for every citation match in ``text`` (used to highlight them)."""
    spans = []
    for m in CITATION_RE.finditer(text):
        spec = normalise_spec(m.group(1))
        spans.append((m.start(), m.end(), [Citation(spec, sec) for sec in SECTION_RE.findall(m.group("list"))]))
    return spans


def extract_citations(text: str) -> list[Citation]:
    """Return unique citations in order of appearance."""
    found = [c for _, _, cits in citation_spans(text) for c in cits]
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


# --------------------------------------------------------------------------- passage references
#
# The context passages are numbered ``[1] [TS 24.501 | Rel-18 | §A.1 | ...]``. Instead of the
# requested ``(TS 24.501, §A.1)`` the model sometimes refers to a passage by its number or gives a
# section without its spec:
#
#     ... the Requested NSSAI [1].        ... in the UE MM Core Network Capability IE [2][6].
#     ... deregister the UE at any time【1†L9-L12】       (gpt-oss reference mark)
#     ... sends a SERVICE REQUEST【2†§5.6.1.2.1】
#     ... MICO indication IE [§5.3.6, "Mobile initiated connection only mode"].
#
# These references are unambiguous, so they are rewritten as standard citations: passage n becomes
# the (spec, section) of the n-th passage given to the model, and a section without spec gets the
# spec of the only retrieved passage with that section. A reference that cannot be resolved without
# guessing (number out of range, section in no passage or in passages of both specs) is left as is
# (bracket text) or removed (reference marks), so it is never counted as a citation.

_MARK_RE = re.compile(r"(\s*)【([^】]*)】")
_PASSAGE_MARK_RE = re.compile(r"^\s*(\d+)\s*†\s*(.*)$")
_NUM_REF_RE = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")
_BARE_SECTION_RE = re.compile(rf"(\s*)[\[(]\s*§{{1,2}}\s*(?P<sec>{SECTION})[^\])]*[\])]")
# A citation that ends right where a reference mark starts: the mark adds nothing.
_ENDS_WITH_CITATION_RE = re.compile(rf"{SPEC}\s*,?\s*§{{1,2}}\s*{SECTION}[^()]*\)\s*$")


def _chunk_sections(chunk: dict[str, Any]) -> set[str]:
    return {chunk["section"], *(chunk.get("merged_sections") or [])}


def _passage_citation(chunk: dict[str, Any], section: str | None = None) -> str:
    """``(TS 24.501, §5.3.7)`` for a passage; ``section`` is kept only if the passage covers it."""
    sec = section if section and section in _chunk_sections(chunk) else chunk["section"]
    return f"({chunk['spec']}, §{sec})"


def _unique_spec(section: str, chunks: Sequence[dict[str, Any]]) -> str | None:
    specs = {c["spec"] for c in chunks if section in _chunk_sections(c)}
    return specs.pop() if len(specs) == 1 else None


def _convert_mark(m: re.Match[str], text: str, chunks: Sequence[dict[str, Any]] | None) -> str:
    inner = m.group(2).strip()
    passage = _PASSAGE_MARK_RE.match(inner)
    content = passage.group(2) if passage else inner
    if SPEC_RE.search(content):  # an explicit citation in the model's bracket style
        return f" ({content.strip()})"
    if not chunks or not passage:
        return ""
    n = int(passage.group(1))
    if not 1 <= n <= len(chunks) or _ENDS_WITH_CITATION_RE.search(text[: m.start()]):
        return ""  # unknown passage, or the mark only repeats the citation just before it
    sec = SECTION_RE.search(content.split("§", 1)[1]) if "§" in content else None
    return " " + _passage_citation(chunks[n - 1], sec.group(0) if sec else None)


def resolve_references(text: str, chunks: Sequence[dict[str, Any]] | None = None) -> str:
    """Rewrite passage references as standard citations (see the comment above).

    Without ``chunks`` only the bracket style is normalised: ``【TS 24.501, §A.1】`` becomes
    ``(TS 24.501, §A.1)`` and other reference marks are removed.
    """
    text = _MARK_RE.sub(lambda m: _convert_mark(m, m.string, chunks), text)
    if not chunks:
        return text.strip()

    def numbered(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
        if not all(1 <= n <= len(chunks) for n in nums):
            return m.group(0)
        return " " + " ".join(dict.fromkeys(_passage_citation(chunks[n - 1]) for n in nums))

    def bare(m: re.Match[str]) -> str:
        spec = _unique_spec(m.group("sec"), chunks)
        return f" ({spec}, §{m.group('sec')})" if spec else m.group(0)

    text = _NUM_REF_RE.sub(numbered, text)
    text = _BARE_SECTION_RE.sub(bare, text)
    return text.strip()


def has_spec_mention(text: str) -> bool:
    """True if the text names a spec at all (used to tell 'no citation' from 'unparsed citation')."""
    return bool(SPEC_RE.search(text))
