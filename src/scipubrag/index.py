"""An Index bundles everything retrieval needs, saved in one folder:

    index/<name>/
        manifest.json    embedder settings, store backend, counts, provenance
        chunks.jsonl     full Chunk records (the source of truth for text + metadata)
        vectors/         the VectorStore's files (NumPy or Chroma)

The BM25 index is rebuilt from chunks.jsonl on load (fast, and avoids pickles).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from scipubrag import __version__
from scipubrag.embedding import Embedder, embedder_from_info
from scipubrag.retrieval.bm25 import BM25Index
from scipubrag.stores import VectorStore, get_store_class
from scipubrag.stores.base import ScoredId, Where
from scipubrag.types import Chunk, Hit

MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl"
VECTORS = "vectors"


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode())
        h.update(c.text_for_embedding.encode())
    return h.hexdigest()[:16]


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `scipubrag chunk` first.")
    with path.open(encoding="utf-8") as fh:
        return [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]


class Index:
    def __init__(self, path: Path, manifest: dict, chunks: list[Chunk], store: VectorStore,
                 embedder: Embedder):
        self.path = Path(path)
        self.manifest = manifest
        self.chunks = chunks
        self.store = store
        self.embedder = embedder
        self._by_id = {c.chunk_id: c for c in chunks}
        self._bm25: BM25Index | None = None

    # -- building / loading --------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, path: str | Path,
              backend: str = "numpy", batch_size: int = 64, verbose: bool = True,
              extra_manifest: dict | None = None) -> Index:
        if not chunks:
            raise ValueError("No chunks to index")
        path = Path(path)
        store_cls = get_store_class(backend)  # fail fast if the backend isn't installed
        embedder.check()
        path.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        vectors: list[list[float]] = []
        texts = [c.text_for_embedding for c in chunks]
        for i in range(0, len(texts), batch_size):
            vectors.extend(embedder.embed_documents(texts[i:i + batch_size]))
            if verbose:
                done = min(i + batch_size, len(texts))
                print(f"  embedded {done}/{len(texts)} ({time.time() - t0:.0f}s)",
                      end="\r", file=sys.stderr, flush=True)
        if verbose:
            print(file=sys.stderr)

        store = store_cls.create(path / VECTORS)
        store.add([c.chunk_id for c in chunks], vectors, [c.flat_metadata() for c in chunks],
                  documents=[c.text for c in chunks])
        store.save()
        write_chunks(chunks, path / CHUNKS)

        manifest = {
            "scipubrag_version": __version__,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "embedder": embedder.info(),
            "store_backend": backend,
            "dim": len(vectors[0]),
            "n_chunks": len(chunks),
            "n_docs": len({c.doc_id for c in chunks}),
            "chunks_fingerprint": chunks_fingerprint(chunks),
            "embed_seconds": round(time.time() - t0, 1),
            **(extra_manifest or {}),
        }
        (path / MANIFEST).write_text(json.dumps(manifest, indent=2))
        return cls(path, manifest, chunks, store, embedder)

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> Index:
        path = Path(path)
        mpath = path / MANIFEST
        if not mpath.exists():
            raise FileNotFoundError(f"No index at {path}. Run `scipubrag index` first.")
        manifest = json.loads(mpath.read_text())
        store = get_store_class(manifest["store_backend"]).load(path / VECTORS)
        chunks = read_chunks(path / CHUNKS)
        embedder = embedder or embedder_from_info(manifest["embedder"])
        return cls(path, manifest, chunks, store, embedder)

    @staticmethod
    def is_current(path: str | Path, chunks: list[Chunk], embedder: Embedder,
                   backend: str) -> bool:
        """True if an index at path was built from these chunks with this embedder/backend."""
        mpath = Path(path) / MANIFEST
        if not mpath.exists():
            return False
        m = json.loads(mpath.read_text())
        return (m.get("chunks_fingerprint") == chunks_fingerprint(chunks)
                and m.get("embedder") == embedder.info()
                and m.get("store_backend") == backend)

    # -- access --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.chunks)
        return self._bm25

    def _hits(self, scored: list[ScoredId], source: str) -> list[Hit]:
        return [Hit(self._by_id[s.id], s.score, source, rank) for rank, s in
                enumerate(scored, start=1) if s.id in self._by_id]

    # -- primitive searches (retrievers in step 2 build on these) ------------

    def dense_search(self, query: str, k: int = 10, where: Where | None = None) -> list[Hit]:
        qvec = self.embedder.embed_query(query)
        return self._hits(self.store.search(qvec, k, where), "dense")

    def keyword_search(self, query: str, k: int = 10, where: Where | None = None) -> list[Hit]:
        return self._hits(self.bm25.search(query, k, where), "bm25")
