from scipubrag.embedding.base import Embedder
from scipubrag.embedding.fake import FakeEmbedder
from scipubrag.embedding.ollama import OllamaEmbedder

__all__ = ["Embedder", "FakeEmbedder", "OllamaEmbedder", "embedder_from_info", "get_embedder"]


def get_embedder(cfg) -> Embedder:
    e = cfg.embedding
    if e.provider == "ollama":
        return OllamaEmbedder(e.model, e.host, e.num_ctx, e.batch_size)
    if e.provider == "fake":
        return FakeEmbedder()
    raise ValueError(f"Unknown embedding provider {e.provider!r} (expected 'ollama' or 'fake')")


def embedder_from_info(info: dict) -> Embedder:
    """Recreate the exact embedder an index was built with (from its manifest)."""
    provider = info.get("provider")
    if provider == "ollama":
        return OllamaEmbedder(info["model"], info.get("host", "http://localhost:11434"),
                              info.get("num_ctx", 2048), doc_prefix=info.get("doc_prefix"),
                              query_prefix=info.get("query_prefix"))
    if provider == "fake":
        return FakeEmbedder(dim=info.get("dim", 256), model=info.get("model", "hash-bow"))
    raise ValueError(f"Unknown embedding provider in manifest: {provider!r}")
