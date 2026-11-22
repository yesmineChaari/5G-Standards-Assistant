"""Keyword search over ``chunks.jsonl`` to find candidate gold sections for the test set.

Usage:
    python scripts/find_sections.py search "T3510" "expir" [--spec 24501] [--title] [-n 15]
    python scripts/find_sections.py section "TS 24.501" 5.5.1.2.7          # full text of a section
    python scripts/find_sections.py absent n78 "Wi-Fi 7" "multi-link"       # out-of-corpus terms
    python scripts/find_sections.py check [eval/testset.yaml]               # validate the test set

``search``: every term must occur in the chunk (case-insensitive; a term is a regex if it starts with
``re:``). Results are sorted by number of matches, title matches first, and print
spec / section / title / snippet. This only proposes candidates: a person must read the spec text.

``absent``: word-boundary, case-insensitive search of each term over all chunk texts and titles.
Exit code 1 if any term is found.

``check``: YAML schema, unique ids, counts per type (10 / 10 / 8 / 3), every gold section exists in the
chunks, ``absent_terms`` of out-of-corpus entries really are absent, and lists unverified entries.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.config import get_spec, load_config, resolve_path  # noqa: E402
from src.ingest.chunker import load_chunks  # noqa: E402

EXPECTED_COUNTS = {"simple_23501": 10, "simple_24501": 10, "cross_spec": 8, "out_of_corpus": 3}
TYPES = {"simple", "cross_spec", "out_of_corpus"}


def compile_term(term: str) -> re.Pattern[str]:
    if term.startswith("re:"):
        return re.compile(term[3:], re.I)
    return re.compile(re.escape(term), re.I)


def word_pattern(term: str) -> re.Pattern[str]:
    """Whole-word match (``n78`` must not match ``n781``; ``Wi-Fi 7`` must not match ``Wi-Fi 70``)."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.I)


def snippet(text: str, pattern: re.Pattern[str], width: int = 110) -> str:
    m = pattern.search(text)
    if not m:
        return text[: 2 * width].replace("\n", " ")
    start, end = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("..." if start else "") + text[start:end].replace("\n", " ") + ("..." if end < len(text) else "")


def search(chunks: list[dict[str, Any]], terms: list[str], spec: str | None = None,
           title_only: bool = False, n: int = 15) -> list[tuple[int, dict[str, Any]]]:
    pats = [compile_term(t) for t in terms]
    hits = []
    for c in chunks:
        if spec and c["spec"] != spec:
            continue
        hay = c["title"] if title_only else f"{c['title']}\n{c['text']}"
        if all(p.search(hay) for p in pats):
            score = sum(len(p.findall(hay)) for p in pats) + 100 * sum(bool(p.search(c["title"])) for p in pats)
            hits.append((score, c))
    hits.sort(key=lambda h: -h[0])
    return hits[:n]


def section_chunks(chunks: list[dict[str, Any]], spec: str, section: str) -> list[dict[str, Any]]:
    return [c for c in chunks if c["spec"] == spec and (c["section"] == section or section in (c.get("merged_sections") or []))]


def find_terms(chunks: list[dict[str, Any]], terms: list[str]) -> dict[str, list[str]]:
    """term -> list of 'spec §section' where it occurs (empty list = absent)."""
    found: dict[str, list[str]] = {}
    for term in terms:
        pat = word_pattern(term)
        found[term] = [f"{c['spec']} §{c['section']}" for c in chunks if pat.search(c["title"]) or pat.search(c["text"])]
    return found


def known_sections(chunks: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs = set()
    for c in chunks:
        pairs.add((c["spec"], c["section"]))
        for s in c.get("merged_sections") or []:
            pairs.add((c["spec"], s))
    return pairs


def gold_exists(spec: str, section: str, sections: set[tuple[str, str]]) -> bool:
    """A gold section exists if a chunk has it, or has one of its sub-sections (heading-only parents
    such as ``5.1.3.2.1`` have no text of their own; the matching rule accepts sub-sections)."""
    return (spec, section) in sections or any(s == spec and sec.startswith(section + ".") for s, sec in sections)


def check_testset(path: Path, chunks: list[dict[str, Any]]) -> int:
    """Validate the test set; returns the number of errors."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(entries, list):
        print(f"{path}: expected a list of entries")
        return 1
    ids = [e.get("id") for e in entries]
    for dup, k in Counter(ids).items():
        if k > 1:
            errors.append(f"duplicate id {dup}")
    sections = known_sections(chunks)
    counts: Counter[str] = Counter()
    required = {"id", "type", "question", "expected_spec", "gold_sections", "reference_answer", "verified"}
    refusal = load_config()["refusal_message"]
    for e in entries:
        eid = e.get("id", "?")
        missing = required - set(e)
        if missing:
            errors.append(f"{eid}: missing fields {sorted(missing)}")
            continue
        if e["type"] not in TYPES:
            errors.append(f"{eid}: unknown type {e['type']!r}")
        if e["type"] == "simple":
            if len(e["expected_spec"]) != 1:
                errors.append(f"{eid}: a simple question needs exactly one expected_spec")
            counts["simple_" + e["expected_spec"][0].replace("TS ", "").replace(".", "")] += 1
        else:
            counts[e["type"]] += 1
        if e["type"] == "cross_spec" and set(e["expected_spec"]) != {"TS 23.501", "TS 24.501"}:
            errors.append(f"{eid}: cross_spec needs both specs in expected_spec")
        if e["type"] == "cross_spec" and {g["spec"] for g in e["gold_sections"]} != {"TS 23.501", "TS 24.501"}:
            errors.append(f"{eid}: cross_spec needs gold sections in both specs")
        for g in e["gold_sections"]:
            if not gold_exists(g["spec"], str(g["section"]), sections):
                errors.append(f"{eid}: gold section {g['spec']} §{g['section']} does not exist in the chunks")
            if g["spec"] not in e["expected_spec"]:
                errors.append(f"{eid}: gold spec {g['spec']} not in expected_spec")
        if e["type"] == "out_of_corpus":
            if e["gold_sections"] or e["expected_spec"]:
                errors.append(f"{eid}: out_of_corpus must have no gold sections and no expected_spec")
            if e["reference_answer"] != refusal:
                errors.append(f"{eid}: out_of_corpus reference_answer must be the exact refusal message")
            terms = e.get("absent_terms") or []
            if not terms:
                errors.append(f"{eid}: out_of_corpus needs absent_terms")
            for term, where in find_terms(chunks, terms).items():
                status = "absent (OK)" if not where else f"FOUND in {len(where)} chunks: {', '.join(where[:5])}"
                print(f"  {eid}: term {term!r}: {status}")
                if where:
                    errors.append(f"{eid}: term {term!r} occurs in the corpus")
        elif not e["gold_sections"]:
            errors.append(f"{eid}: in-corpus question without gold sections")
        if not str(e["reference_answer"]).strip():
            errors.append(f"{eid}: empty reference_answer")

    print(f"\nEntries: {len(entries)}  counts: {dict(counts)}  expected: {EXPECTED_COUNTS}")
    for k, v in EXPECTED_COUNTS.items():
        if counts.get(k, 0) != v:
            errors.append(f"count {k} = {counts.get(k, 0)}, expected {v}")
    if not 28 <= len(entries) <= 33:
        errors.append(f"{len(entries)} questions; the brief requires 28 to 33")
    unverified = [e.get("id") for e in entries if not e.get("verified")]
    if unverified:
        print(f"WARNING: {len(unverified)}/{len(entries)} entries are not verified: {', '.join(map(str, unverified))}")
    for err in errors:
        print("ERROR:", err)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return len(errors)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("search")
    p.add_argument("terms", nargs="+")
    p.add_argument("--spec", help="23501 / 24501 / 'TS 24.501'")
    p.add_argument("--title", action="store_true", help="match titles only")
    p.add_argument("-n", type=int, default=15)
    p = sub.add_parser("section")
    p.add_argument("spec")
    p.add_argument("section")
    p = sub.add_parser("absent")
    p.add_argument("terms", nargs="+")
    p = sub.add_parser("check")
    p.add_argument("path", nargs="?", default="eval/testset.yaml")
    args = parser.parse_args(argv)

    cfg = load_config()
    chunks = load_chunks()
    if args.cmd == "search":
        spec = get_spec(cfg, args.spec)["id"] if args.spec else None
        terms = args.terms
        hits = search(chunks, terms, spec, args.title, args.n)
        pat = compile_term(terms[0])
        for score, c in hits:
            print(f"{c['spec']} §{c['section']:<14} {c['title'][:70]}  [{c['chunk_type']}, score {score}]")
            print(f"    {snippet(c['text'], pat)}")
        print(f"\n{len(hits)} result(s) shown")
    elif args.cmd == "section":
        spec = get_spec(cfg, args.spec)["id"]
        found = section_chunks(chunks, spec, args.section)
        if not found:
            print(f"No chunk for {spec} §{args.section}")
            return 1
        for c in found:
            print(f"===== {c['chunk_id']} ({c['chunk_type']}, {c['n_tokens']} tokens)\n{c['text']}\n")
    elif args.cmd == "absent":
        found = find_terms(chunks, args.terms)
        for term, where in found.items():
            print(f"{term!r}: " + ("not found in the corpus" if not where else f"FOUND in {len(where)} chunks: {', '.join(where[:8])}"))
        return int(any(found.values()))
    elif args.cmd == "check":
        return int(check_testset(resolve_path(args.path) if not Path(args.path).is_absolute() else Path(args.path), chunks) > 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
