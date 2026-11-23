"""Exact (brute-force) cosine search with NumPy.

For corpora up to ~100k chunks this is as fast as an approximate index and has
no recall loss. Files:
    <path>/vectors.npy     float32 matrix, rows L2-normalized
    <path>/records.jsonl   {"id": ..., "metadata": {...}} per row, same order
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from scipubrag.stores.base import Metadata, ScoredId, VectorStore, Where, matches


class NumpyStore(VectorStore):
    backend = "numpy"

    def __init__(self, path: str | Path):
        super().__init__(path)
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []
        self._meta: list[Metadata] = []
        self._pos: dict[str, int] = {}

    @classmethod
    def create(cls, path: str | Path) -> NumpyStore:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
        return cls(p)

    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[Metadata],
            documents: list[str] | None = None) -> None:
        if not (len(ids) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, embeddings and metadatas must have the same length")
        dupes = [i for i in ids if i in self._pos]
        if dupes or len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate ids: {dupes[:5] or 'within batch'}")

        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError("embeddings must be a 2-D list")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.where(norms == 0, 1, norms)

        if self._vectors is not None and self._vectors.shape[1] != mat.shape[1]:
            raise ValueError(f"Dimension mismatch: store has {self._vectors.shape[1]}, "
                             f"got {mat.shape[1]}")
        self._vectors = mat if self._vectors is None else np.vstack([self._vectors, mat])
        for i, cid in enumerate(ids):
            self._pos[cid] = len(self._ids) + i
        self._ids.extend(ids)
        self._meta.extend(dict(m) for m in metadatas)

    def search(self, query_embedding: list[float], k: int = 10,
               where: Where | None = None) -> list[ScoredId]:
        if self._vectors is None or not self._ids:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._vectors @ q

        if where:
            mask = np.fromiter((matches(m, where) for m in self._meta), dtype=bool,
                               count=len(self._meta))
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredId(self._ids[i], float(scores[i])) for i in top]

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        dim = 0 if self._vectors is None else self._vectors.shape[1]
        vectors = self._vectors if self._vectors is not None else np.zeros((0, dim), np.float32)
        np.save(self.path / "vectors.npy", vectors)
        with (self.path / "records.jsonl").open("w", encoding="utf-8") as fh:
            for cid, meta in zip(self._ids, self._meta, strict=True):
                fh.write(json.dumps({"id": cid, "metadata": meta}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> NumpyStore:
        store = cls(path)
        vec_file, rec_file = store.path / "vectors.npy", store.path / "records.jsonl"
        if not vec_file.exists() or not rec_file.exists():
            raise FileNotFoundError(f"No NumPy store at {store.path}")
        store._vectors = np.load(vec_file)
        with rec_file.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                store._pos[rec["id"]] = len(store._ids)
                store._ids.append(rec["id"])
                store._meta.append(rec["metadata"])
        if len(store._ids) != store._vectors.shape[0]:
            raise ValueError(f"Corrupt store at {store.path}: row count mismatch")
        return store

    def __len__(self) -> int:
        return len(self._ids)
