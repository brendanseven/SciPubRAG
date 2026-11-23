"""Embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Documents and queries are embedded separately because many retrieval models
    are trained with different prefixes (or even different encoders) for each."""

    provider: str = "base"
    model: str = ""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    def check(self) -> None:
        """Raise a helpful error if the backend isn't usable. Optional."""
        return None  # noqa: B027 (intentionally optional hook)

    def info(self) -> dict:
        """Everything needed to recreate this embedder; stored in the index manifest."""
        return {"provider": self.provider, "model": self.model}
