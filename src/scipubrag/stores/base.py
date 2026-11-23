"""VectorStore interface and the shared metadata filter language.

Filters (`where`) are plain dicts, the same for every backend:

    {"doc_id": "smith2021"}                         equality
    {"kind": {"$in": ["body", "abstract"]}}         membership
    {"part": {"$ne": "back"}}                       not equal
    {"page_start": {"$gte": 3}}                     comparisons: $gt $gte $lt $lte
    {"doc_id": "smith2021", "part": "body"}         several keys are ANDed
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Metadata = dict[str, str | int | float | bool]
Where = dict[str, Any]

_OPS = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$in": lambda a, b: a in b,
    "$nin": lambda a, b: a not in b,
    "$gt": lambda a, b: a is not None and a > b,
    "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b,
    "$lte": lambda a, b: a is not None and a <= b,
}


def matches(meta: Metadata, where: Where | None) -> bool:
    if not where:
        return True
    for key, cond in where.items():
        value = meta.get(key)
        if isinstance(cond, dict):
            for op, arg in cond.items():
                if op not in _OPS:
                    raise ValueError(f"Unsupported filter operator {op!r}")
                if not _OPS[op](value, arg):
                    return False
        elif value != cond:
            return False
    return True


@dataclass
class ScoredId:
    id: str
    score: float  # cosine similarity, higher is better


class VectorStore(ABC):
    """Stores normalized embeddings with flat metadata and answers k-NN queries.

    Implementations persist to a directory and must return cosine similarity
    scores, so results are comparable across backends.
    """

    backend: str = "base"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @abstractmethod
    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[Metadata],
            documents: list[str] | None = None) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], k: int = 10,
               where: Where | None = None) -> list[ScoredId]: ...

    @abstractmethod
    def save(self) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> VectorStore: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @classmethod
    def create(cls, path: str | Path) -> VectorStore:
        """New, empty store at path (replacing anything already there)."""
        return cls(path)
