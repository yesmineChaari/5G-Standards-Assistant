"""Types and helpers shared by the dense and lexical indexes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Hit:
    """One retrieved chunk."""

    chunk_id: str
    score: float  # higher is better (cosine similarity for dense, BM25 score for lexical)
    rank: int  # 1-based
    chunk: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    """Hex SHA256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def print_hits(hits: list[Hit], width: int = 160) -> None:
    """Print hits as `rank. score spec §section title [id]` + a one-line snippet."""
    for h in hits:
        c = h.chunk
        snippet = " ".join(c["text"].split())[:width]
        print(f"{h.rank:>2}. {h.score:.4f}  {c['spec']} §{c['section']}  {c['title']}  [{c['chunk_id']}]")
        print(f"      {snippet}")
