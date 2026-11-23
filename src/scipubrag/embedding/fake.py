"""Deterministic bag-of-words hashing embedder for tests and offline development.

Not semantic, but texts that share words get similar vectors, which is enough
to exercise the whole pipeline without Ollama.
"""

from __future__ import annotations

import hashlib
import math
import re

from scipubrag.embedding.base import Embedder


class FakeEmbedder(Embedder):
    provider = "fake"

    def __init__(self, dim: int = 256, model: str = "hash-bow"):
        self.dim = dim
        self.model = model

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def info(self) -> dict:
        return {"provider": self.provider, "model": self.model, "dim": self.dim}
