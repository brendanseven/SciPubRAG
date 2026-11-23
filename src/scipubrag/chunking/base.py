"""Chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from scipubrag.chunking.text import count_tokens
from scipubrag.types import Chunk, Document


class Chunker(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split one document into chunks."""

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        chunks = [c for d in docs for c in self.chunk(d)]
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})[:5]
            raise ValueError(f"Duplicate chunk ids (are two documents named the same?): {dupes}")
        return chunks

    @staticmethod
    def make_chunk(doc: Document, chunk_id: str, text: str, path: list[str],
                   pages: set[int] | list[int], kind: str = "body", part: str = "body",
                   header: bool = True) -> Chunk:
        title = doc.title or doc.doc_id
        prefix = " > ".join([title] + path)
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            text=text,
            text_for_embedding=f"{prefix}\n\n{text}" if header else text,
            kind=kind,
            part=part,
            section_path=list(path),
            pages=sorted(p for p in pages if p is not None),
            n_tokens=count_tokens(text),
            title=title,
            year=doc.year,
            first_author=doc.first_author,
        )
