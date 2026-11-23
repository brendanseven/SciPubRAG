"""Chroma-backed store (optional: pip install 'scipubrag[chroma]', Python <= 3.14 for now)."""

from __future__ import annotations

import shutil
from pathlib import Path

from scipubrag.stores.base import Metadata, ScoredId, VectorStore, Where

COLLECTION = "chunks"


def _import_chroma():
    try:
        import chromadb
    except ImportError as e:
        raise ImportError(
            "Chroma isn't installed. Run `pip install 'scipubrag[chroma]'` "
            "(needs a Python version onnxruntime supports, currently 3.11 to 3.14), "
            "or use store.backend = \"numpy\"."
        ) from e
    return chromadb


def to_chroma_where(where: Where | None) -> dict | None:
    """Our filter dicts -> Chroma's (which needs explicit $eq and $and for several keys)."""
    if not where:
        return None
    clauses = [{k: v if isinstance(v, dict) else {"$eq": v}} for k, v in where.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore(VectorStore):
    backend = "chroma"

    def __init__(self, path: str | Path):
        super().__init__(path)
        chromadb = _import_chroma()
        self._client = chromadb.PersistentClient(path=str(self.path))
        try:
            self._col = self._client.get_or_create_collection(
                COLLECTION, configuration={"hnsw": {"space": "cosine"}})
        except TypeError:  # Chroma < 1.0
            self._col = self._client.get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"})

    @classmethod
    def create(cls, path: str | Path) -> ChromaStore:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
        return cls(p)

    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[Metadata],
            documents: list[str] | None = None) -> None:
        step = 1000  # stay under Chroma's max batch size
        for i in range(0, len(ids), step):
            kwargs = {"ids": ids[i:i + step], "embeddings": embeddings[i:i + step],
                      "metadatas": metadatas[i:i + step]}
            if documents is not None:
                kwargs["documents"] = documents[i:i + step]
            self._col.add(**kwargs)

    def search(self, query_embedding: list[float], k: int = 10,
               where: Where | None = None) -> list[ScoredId]:
        n = len(self)
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[query_embedding], n_results=min(k, n),
                              where=to_chroma_where(where), include=["distances"])
        return [ScoredId(cid, 1.0 - float(dist))  # cosine distance -> similarity
                for cid, dist in zip(res["ids"][0], res["distances"][0], strict=True)]

    def save(self) -> None:
        pass  # PersistentClient writes as it goes

    @classmethod
    def load(cls, path: str | Path) -> ChromaStore:
        if not Path(path).exists():
            raise FileNotFoundError(f"No Chroma store at {path}")
        return cls(path)

    def __len__(self) -> int:
        return self._col.count()
