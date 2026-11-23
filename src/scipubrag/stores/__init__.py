from scipubrag.stores.base import ScoredId, VectorStore, matches
from scipubrag.stores.numpy_store import NumpyStore

__all__ = ["NumpyStore", "ScoredId", "VectorStore", "get_store_class", "matches"]


def get_store_class(backend: str) -> type[VectorStore]:
    if backend == "numpy":
        return NumpyStore
    if backend == "chroma":
        from scipubrag.stores.chroma_store import ChromaStore, _import_chroma

        _import_chroma()  # raise a helpful error now if chromadb isn't installed
        return ChromaStore
    raise ValueError(f"Unknown store backend {backend!r} (expected 'numpy' or 'chroma')")
