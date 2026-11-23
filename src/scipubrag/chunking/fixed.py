"""Naive fixed-size chunking: the baseline to beat in the ablation table.

Ignores structure: all text is concatenated and cut into windows of `size`
tokens (approximated by words) with `overlap` tokens shared. No context header.
"""

from __future__ import annotations

from scipubrag.chunking.base import Chunker
from scipubrag.types import Chunk, Document


class FixedSizeChunker(Chunker):
    name = "fixed"

    def __init__(self, size: int = 512, overlap: int = 50, header: bool = False):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        # count_tokens estimates 1.3 tokens/word, so convert to a word budget
        self.words = max(1, int(size / 1.3))
        self.step = max(1, int((size - overlap) / 1.3))
        self.header = header

    def chunk(self, doc: Document) -> list[Chunk]:
        # (word, section_path, page) triples so chunks keep a citation location
        stream: list[tuple[str, tuple[str, ...], int | None]] = []
        for w in doc.abstract.split():
            stream.append((w, ("Abstract",), None))
        for section in doc.sections:
            for block in section.blocks:
                for w in block.text.split():
                    stream.append((w, tuple(section.path), block.page))

        out: list[Chunk] = []
        for i, start in enumerate(range(0, max(len(stream), 1), self.step)):
            window = stream[start:start + self.words]
            if not window:
                break
            text = " ".join(w for w, _, _ in window)
            path = list(window[0][1])
            pages = {p for _, _, p in window if p is not None}
            kind = "abstract" if path == ["Abstract"] else "body"
            out.append(self.make_chunk(doc, f"{doc.doc_id}__fixed__{i:04d}", text, path, pages,
                                       kind, "body", self.header))
            if start + self.words >= len(stream):
                break
        return out
