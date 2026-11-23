"""Ollama embeddings over HTTP (standard library only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from scipubrag.embedding.base import Embedder

# Task prefixes some models were trained with. Unlisted models get none.
PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed2": ("", "query: "),
}


class OllamaEmbedder(Embedder):
    provider = "ollama"

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434",
                 num_ctx: int = 2048, batch_size: int = 32, timeout: int = 300,
                 doc_prefix: str | None = None, query_prefix: str | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.batch_size = batch_size
        self.timeout = timeout
        default_doc, default_query = PREFIXES.get(model.split(":")[0], ("", ""))
        self.doc_prefix = default_doc if doc_prefix is None else doc_prefix
        self.query_prefix = default_query if query_prefix is None else query_prefix

    # -- HTTP ----------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def check(self) -> None:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as resp:
                tags = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"Can't reach Ollama at {self.host}. Is it running?") from e
        names = {m.get("name", "") for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            raise RuntimeError(f"Model '{self.model}' isn't pulled. Run: ollama pull {self.model}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            out = self._post("/api/embed", {"model": self.model, "input": texts,
                                            "options": {"num_ctx": self.num_ctx}})
            return out["embeddings"]
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise RuntimeError(f"Ollama error {e.code}: {e.read().decode()[:300]}") from e
        # Older Ollama: one text per request on the legacy endpoint
        return [self._post("/api/embeddings", {"model": self.model, "prompt": t})["embedding"]
                for t in texts]

    # -- Embedder API --------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed([self.doc_prefix + t for t in texts[i:i + self.batch_size]]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def info(self) -> dict:
        return {"provider": self.provider, "model": self.model, "host": self.host,
                "num_ctx": self.num_ctx, "doc_prefix": self.doc_prefix,
                "query_prefix": self.query_prefix}
