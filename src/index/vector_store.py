"""Persistent Chroma vector store for the chunks.

Usage:
    python -m src.index.vector_store build [--force]
    python -m src.index.vector_store query "What does cause #9 mean in 5GMM?" [-k 5] [--spec "TS 24.501"]

The collection stores, for every chunk: its id, its embedding (of ``text_for_index``), the
``text_for_index`` as document, and all chunk fields as flat metadata (lists are serialised to
JSON strings). A fingerprint (embedding model + SHA256 of chunks.jsonl) is stored in the
collection metadata: ``build`` is skipped when the index is already up to date, so reloading
never recomputes embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from src.config import load_config, resolve_path
from src.index.common import Hit, file_sha256, print_hits
from src.index.embed import Embedder, set_seed
from src.ingest.chunker import load_chunks

logger = logging.getLogger(__name__)

ADD_BATCH = 256
LIST_FIELDS = ("merged_sections",)
METADATA_FIELDS = (
    "chunk_id",
    "spec",
    "version",
    "release",
    "section",
    "title",
    "section_path",
    "merged_sections",
    "chunk_type",
    "part",
    "n_tokens",
    "text",
)


def to_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Flatten a chunk dict into Chroma metadata (str/int/float/bool only)."""
    meta: dict[str, Any] = {}
    for key in METADATA_FIELDS:
        value = chunk[key]
        meta[key] = json.dumps(value) if key in LIST_FIELDS else value
    return meta


def from_metadata(meta: dict[str, Any], document: str | None = None) -> dict[str, Any]:
    """Inverse of :func:`to_metadata`; adds ``text_for_index`` from the stored document."""
    chunk = dict(meta)
    for key in LIST_FIELDS:
        if key in chunk and isinstance(chunk[key], str):
            chunk[key] = json.loads(chunk[key])
    if document is not None:
        chunk["text_for_index"] = document
    return chunk


class VectorStore:
    """Chroma-backed dense index over the chunks."""

    def __init__(self, path: str | Path, collection: str, embedder: Embedder, distance: str = "cosine",
                 exact_search: bool = True) -> None:
        import chromadb
        from chromadb.config import Settings

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self.distance = distance
        self.embedder = embedder
        # Exact search: cosine similarity against every stored vector (deterministic across runs).
        # Chroma's HNSW index is approximate and its results can vary slightly between processes.
        self.exact_search = exact_search
        self._matrix: Any = None  # (ids, unit-norm embeddings, metadatas, documents), loaded once
        self.client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False)
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None, embedder: Embedder | None = None) -> "VectorStore":
        cfg = cfg or load_config()
        vs = cfg["vector_store"]
        return cls(
            path=resolve_path(vs["path"]),
            collection=vs["collection"],
            embedder=embedder or Embedder.from_config(cfg),
            distance=vs.get("distance", "cosine"),
            exact_search=vs.get("exact_search", True),
        )

    # ------------------------------------------------------------------ build

    def _fingerprint(self, chunks_sha: str) -> dict[str, str]:
        return {"embedding_model": self.embedder.model_name, "chunks_sha256": chunks_sha}

    def is_up_to_date(self, chunks_sha: str) -> bool:
        """True if the collection exists, is non-empty and matches the model + chunks file."""
        try:
            col = self.client.get_collection(self.collection_name)
        except Exception:  # collection does not exist
            return False
        meta = col.metadata or {}
        fp = self._fingerprint(chunks_sha)
        return col.count() > 0 and all(meta.get(k) == v for k, v in fp.items())

    def build(self, chunks: Sequence[dict[str, Any]], chunks_sha: str, force: bool = False) -> int:
        """(Re)build the collection from chunks. Returns the number of indexed chunks.

        Embedding and insertion are done batch by batch (low memory). If a previous build with
        the same fingerprint was interrupted, chunks already in the collection are skipped
        (resume). A different fingerprint, or ``force``, recreates the collection.
        """
        fp = self._fingerprint(chunks_sha)
        col = None
        if not force:
            try:
                existing = self.client.get_collection(self.collection_name)
                meta = existing.metadata or {}
                if all(meta.get(k) == v for k, v in fp.items()):
                    col = existing
            except Exception:  # collection does not exist
                col = None
        if col is None:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            col = self.client.create_collection(
                self.collection_name, metadata={"hnsw:space": self.distance, **fp}
            )

        done = set(col.get(include=[])["ids"]) if col.count() else set()
        todo = [c for c in chunks if c["chunk_id"] not in done]
        if not todo:
            logger.info("Vector index up to date (%d chunks), nothing to do", col.count())
            return col.count()
        if done:
            logger.info("Resuming: %d chunks already indexed, %d to go", len(done), len(todo))

        truncated = self.embedder.count_truncated([c["text_for_index"] for c in todo])
        if truncated:
            logger.warning(
                "%d/%d chunks exceed %d model tokens and will be truncated for embedding",
                truncated, len(todo), self.embedder.model.max_seq_length,
            )
        t0 = time.time()
        logger.info("Embedding %d chunks with %s on %s", len(todo), self.embedder.model_name, self.embedder.device)
        for start in range(0, len(todo), ADD_BATCH):
            batch = todo[start : start + ADD_BATCH]
            vectors = self.embedder.embed_documents([c["text_for_index"] for c in batch])
            col.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=vectors.tolist(),
                documents=[c["text_for_index"] for c in batch],
                metadatas=[to_metadata(c) for c in batch],
            )
            n_done = start + len(batch)
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(todo) - n_done)
            logger.info("  %d/%d embedded (%.0f s elapsed, ~%.0f s left)", n_done, len(todo), elapsed, eta)
        logger.info("Indexed %d chunks into %s/%s", col.count(), self.path, self.collection_name)
        return col.count()

    # ------------------------------------------------------------------ query

    @property
    def collection(self) -> Any:
        return self.client.get_collection(self.collection_name)

    def count(self) -> int:
        return self.collection.count()

    def query(self, query: str, k: int = 10, specs: Sequence[str] | None = None) -> list[Hit]:
        """Dense top-k search, optionally restricted to the given specs (e.g. ``["TS 24.501"]``)."""
        where = None
        if specs:
            specs = list(specs)
            where = {"spec": specs[0]} if len(specs) == 1 else {"spec": {"$in": specs}}
        qvec = self.embedder.embed_query(query)
        if self.exact_search:
            return self._exact_query(qvec, k, specs)
        res = self.collection.query(
            query_embeddings=[qvec.tolist()],
            n_results=k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        hits: list[Hit] = []
        for rank, (cid, meta, doc, dist) in enumerate(
            zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]), start=1
        ):
            score = 1.0 - float(dist) if self.distance == "cosine" else -float(dist)
            hits.append(Hit(cid, score, rank, from_metadata(meta, doc)))
        return hits


    def _load_matrix(self) -> Any:
        if self._matrix is None:
            import numpy as np

            data = self.collection.get(include=["embeddings", "metadatas", "documents"])
            emb = np.asarray(data["embeddings"], dtype=np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
            order = np.argsort(np.asarray(data["ids"]), kind="stable")  # fixed order: chunk id
            self._matrix = ([data["ids"][i] for i in order], emb[order],
                            [data["metadatas"][i] for i in order], [data["documents"][i] for i in order])
        return self._matrix

    def _exact_query(self, qvec: Any, k: int, specs: Sequence[str] | None) -> list[Hit]:
        import numpy as np

        ids, emb, metas, docs = self._load_matrix()
        q = np.asarray(qvec, dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        scores = emb @ q
        if specs:
            allowed = set(specs)
            mask = np.fromiter((m["spec"] in allowed for m in metas), dtype=bool, count=len(metas))
            scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores, kind="stable")[:k]  # ties: chunk id order
        return [Hit(ids[i], float(scores[i]), rank, from_metadata(metas[i], docs[i]))
                for rank, i in enumerate(top, start=1) if np.isfinite(scores[i])]


def build_index(cfg: dict[str, Any] | None = None, force: bool = False) -> VectorStore:
    """Build (or reuse) the vector index for the chunks listed in the config."""
    cfg = cfg or load_config()
    set_seed(cfg["seed"])
    chunks_path = resolve_path(cfg["paths"]["chunks_file"])
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found; run `make ingest` first")
    store = VectorStore.from_config(cfg)
    store.build(load_chunks(chunks_path), file_sha256(chunks_path), force=force)
    return store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chroma vector index")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the index (skipped if up to date)")
    b.add_argument("--force", action="store_true", help="rebuild even if up to date")
    q = sub.add_parser("query", help="dense-only query")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--spec", action="append", help="restrict to a spec (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    if args.cmd == "build":
        store = build_index(cfg, force=args.force)
        print(f"Index ready: {store.count()} chunks")
    else:
        store = VectorStore.from_config(cfg)
        print_hits(store.query(args.text, k=args.k, specs=args.spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
