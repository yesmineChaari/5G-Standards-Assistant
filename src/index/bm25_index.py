"""Lexical BM25 index (rank_bm25.BM25Okapi) with a tokenizer that keeps 3GPP identifiers intact.

Usage:
    python -m src.index.bm25_index build [--force]
    python -m src.index.bm25_index query "T3510" [-k 5] [--spec "TS 24.501"]

Tokenizer (:func:`tokenize`):
  * lowercase; cause references normalised: ``cause #9``, ``cause 9``, ``cause value #9``,
    ``# 9`` -> ``#9`` (the word ``cause`` is kept as its own token);
  * identifiers kept as single tokens: ``t3510``, ``#9``, ``5gmm``, ``n2``, ``s-nssai``,
    ``9.11.3.2``, ``5gmm-registered.normal-service``; ``.``/``-`` inside a token are kept,
    surrounding punctuation is dropped;
  * hyphen/dot compounds that contain letters also emit their parts (``s-nssai`` -> ``s-nssai``,
    ``nssai``) so that a query for a part still matches; dotted numbers (section numbers) do not;
  * English stopwords removed; optional light plural stemming (S-stemmer) of plain alphabetic
    words only (``messages`` -> ``message``), never of identifiers.

The index is built over ``text_for_index`` (header line + text), so spec, section number and
title are searchable. It is pickled together with the chunks and a fingerprint (SHA256 of
chunks.jsonl + tokenizer settings) so that reloading never recomputes it.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import load_config, resolve_path
from src.index.common import Hit, file_sha256, print_hits
from src.ingest.chunker import load_chunks

logger = logging.getLogger(__name__)

TOKENIZER_VERSION = 1

# English stopwords (closed-class words). Negations are kept: "not allowed" matters in specs.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves also may shall e.g i.e etc via within
    whether either
    """.split()
)

_CAUSE_RE = re.compile(r"\bcause(?:\s+value)?\s*(?:no\.?\s*|number\s*)?#?\s*(\d+)\b")
_HASH_SPACE_RE = re.compile(r"#\s+(\d+)\b")
# A token: "#<digits>" or an alphanumeric run, possibly joined by ".", "-", "_" or "/".
_TOKEN_RE = re.compile(r"#\d+|[a-z0-9]+(?:[.\-_/][a-z0-9]+)*")
_PART_SPLIT_RE = re.compile(r"[.\-_/]")


def normalise_causes(text: str) -> str:
    """Rewrite cause references to ``cause #N`` (expects lowercase text)."""
    text = _HASH_SPACE_RE.sub(r"#\1", text)
    return _CAUSE_RE.sub(r"cause #\1", text)


def s_stem(word: str) -> str:
    """Harman's S-stemmer for plain alphabetic words (plural removal only)."""
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("us", "ss", "is")):
        return word[:-1]
    return word


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Tokenize text for BM25, keeping 3GPP identifiers intact (see module docstring)."""
    text = normalise_causes(text.lower())
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if tok in STOPWORDS:
            continue
        is_identifier = any(ch.isdigit() for ch in tok) or "#" in tok
        compound = _PART_SPLIT_RE.search(tok) is not None
        if compound:
            tokens.append(tok)
            has_letters = any(ch.isalpha() for ch in tok)
            if has_letters:
                for part in _PART_SPLIT_RE.split(tok):
                    if len(part) >= 2 and part not in STOPWORDS and part != tok:
                        tokens.append(part)
        elif is_identifier:
            tokens.append(tok)
        else:
            tokens.append(s_stem(tok) if stem else tok)
    return tokens


class BM25Index:
    """BM25Okapi over all chunks, with optional filtering by spec at query time."""

    def __init__(self, chunks: Sequence[dict[str, Any]], stem: bool = True, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.specs = np.array([c["spec"] for c in self.chunks])
        self.stem = stem
        self.k1, self.b = k1, b
        corpus = [tokenize(c["text_for_index"], stem=stem) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)
        self.fingerprint: dict[str, Any] = {}

    def query(self, query: str, k: int = 10, specs: Sequence[str] | None = None) -> list[Hit]:
        """Top-k chunks by BM25 score (only chunks with a positive score are returned)."""
        q_tokens = tokenize(query, stem=self.stem)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        if specs:
            scores = np.where(np.isin(self.specs, list(specs)), scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        hits: list[Hit] = []
        for idx in top:
            score = float(scores[idx])
            if not np.isfinite(score) or score <= 0:
                break
            hits.append(Hit(self.chunk_ids[idx], score, len(hits) + 1, self.chunks[idx]))
        return hits

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        """Load a pickled index (only load files produced by this project)."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, BM25Index):
            raise TypeError(f"{path} does not contain a BM25Index")
        return obj


def _fingerprint(chunks_sha: str, cfg: dict[str, Any]) -> dict[str, Any]:
    b = cfg["bm25"]
    return {
        "chunks_sha256": chunks_sha,
        "tokenizer_version": TOKENIZER_VERSION,
        "stem": b.get("stem", True),
        "k1": b.get("k1", 1.5),
        "b": b.get("b", 0.75),
    }


def build_bm25(cfg: dict[str, Any] | None = None, force: bool = False) -> BM25Index:
    """Build the BM25 index (or reuse the pickled one when up to date) and return it."""
    cfg = cfg or load_config()
    chunks_path = resolve_path(cfg["paths"]["chunks_file"])
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found; run `make ingest` first")
    index_path = resolve_path(cfg["bm25"]["path"])
    fp = _fingerprint(file_sha256(chunks_path), cfg)
    if not force and index_path.exists():
        try:
            existing = BM25Index.load(index_path)
            if existing.fingerprint == fp:
                logger.info("BM25 index up to date (%d chunks), nothing to do", len(existing.chunk_ids))
                return existing
        except Exception as exc:  # unreadable/old pickle: rebuild
            logger.warning("Could not reuse %s (%s), rebuilding", index_path, exc)
    b = cfg["bm25"]
    index = BM25Index(load_chunks(chunks_path), stem=b.get("stem", True), k1=b.get("k1", 1.5), b=b.get("b", 0.75))
    index.fingerprint = fp
    index.save(index_path)
    logger.info("BM25 index built over %d chunks -> %s", len(index.chunk_ids), index_path)
    return index


def load_bm25(cfg: dict[str, Any] | None = None) -> BM25Index:
    """Load the persisted BM25 index (raises if it has not been built)."""
    cfg = cfg or load_config()
    path = resolve_path(cfg["bm25"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make index` first")
    return BM25Index.load(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BM25 index")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the index (skipped if up to date)")
    b.add_argument("--force", action="store_true")
    q = sub.add_parser("query", help="BM25-only query")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--spec", action="append", help="restrict to a spec (repeatable)")
    q.add_argument("--show-tokens", action="store_true", help="print the query tokens")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    if args.cmd == "build":
        index = build_bm25(cfg, force=args.force)
        print(f"BM25 index ready: {len(index.chunk_ids)} chunks")
    else:
        index = load_bm25(cfg)
        if args.show_tokens:
            print("query tokens:", tokenize(args.text, stem=index.stem))
        print_hits(index.query(args.text, k=args.k, specs=args.spec))
    return 0


if __name__ == "__main__":
    # Run through the importable module so that pickled objects reference
    # `src.index.bm25_index.BM25Index` and not `__main__.BM25Index`.
    from src.index.bm25_index import main as _main

    raise SystemExit(_main())
