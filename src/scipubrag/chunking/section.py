"""Structure-aware chunking: paragraphs packed within sections, never across them."""

from __future__ import annotations

from scipubrag.chunking.base import Chunker
from scipubrag.chunking.text import count_tokens, slug, split_sentences
from scipubrag.types import Block, Chunk, Document

Unit = dict  # {"text": str, "pages": set[int]}


class SectionChunker(Chunker):
    """
    - Chunks never cross a section boundary.
    - Paragraphs are packed up to `target` tokens; oversized ones split on sentences.
    - Consecutive chunks in a section share `overlap` sentences.
    - A trailing piece under `min_tokens` is merged into the previous chunk.
    - Formulas stay attached to the paragraph before them.
    - Each chunk's embedding text is prefixed with 'Title > Section > Subsection'.
    """

    name = "section"

    def __init__(self, target: int = 400, max_tokens: int = 512, min_tokens: int = 80,
                 overlap: int = 1, include_captions: bool = True, include_back: bool = True,
                 header: bool = True):
        if not 0 < min_tokens <= target <= max_tokens:
            raise ValueError("Need 0 < min <= target <= max")
        self.target = target
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap = overlap
        self.include_captions = include_captions
        self.include_back = include_back
        self.header = header

    # -- packing -------------------------------------------------------------

    @staticmethod
    def _units(blocks: list[Block]) -> list[Unit]:
        units: list[Unit] = []
        for b in blocks:
            if b.type == "formula" and units:
                units[-1]["text"] += "\n" + b.text
                if b.page:
                    units[-1]["pages"].add(b.page)
            else:
                units.append({"text": b.text, "pages": {b.page} if b.page else set()})
        return units

    def _split_long(self, unit: Unit) -> list[Unit]:
        pieces, current = [], []
        for sent in split_sentences(unit["text"]):
            if current and count_tokens(" ".join(current + [sent])) > self.max_tokens:
                pieces.append(" ".join(current))
                current = []
            current.append(sent)
        if current:
            pieces.append(" ".join(current))
        return [{"text": p, "pages": set(unit["pages"])} for p in pieces]

    def pack(self, units: list[Unit]) -> list[Unit]:
        expanded: list[Unit] = []
        for u in units:
            expanded.extend(self._split_long(u) if count_tokens(u["text"]) > self.max_tokens
                            else [u])

        chunks: list[Unit] = []
        texts: list[str] = []
        pages: set[int] = set()
        for u in expanded:
            if texts and count_tokens("\n\n".join(texts + [u["text"]])) > self.target:
                chunks.append({"text": "\n\n".join(texts), "pages": set(pages)})
                tail = split_sentences(texts[-1])[-self.overlap:] if self.overlap > 0 else []
                tail_text = " ".join(tail)
                if tail and count_tokens(tail_text) + count_tokens(u["text"]) <= self.max_tokens:
                    texts, pages = [tail_text], set(chunks[-1]["pages"])
                else:
                    texts, pages = [], set()
            texts.append(u["text"])
            pages |= u["pages"]
        if texts:
            chunks.append({"text": "\n\n".join(texts), "pages": set(pages)})

        if len(chunks) >= 2 and count_tokens(chunks[-1]["text"]) < self.min_tokens:
            merged = chunks[-2]["text"] + "\n\n" + chunks[-1]["text"]
            if count_tokens(merged) <= self.max_tokens + self.min_tokens:
                chunks[-2] = {"text": merged, "pages": chunks[-2]["pages"] | chunks[-1]["pages"]}
                chunks.pop()
        return chunks

    # -- documents -----------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        did = doc.doc_id
        out: list[Chunk] = []

        if doc.abstract:
            units = [{"text": p, "pages": set()} for p in doc.abstract.split("\n\n") if p.strip()]
            for i, c in enumerate(self.pack(units)):
                out.append(self.make_chunk(doc, f"{did}__abstract__{i:03d}", c["text"],
                                           ["Abstract"], c["pages"], "abstract", "body",
                                           self.header))

        seen: dict[str, int] = {}
        for section in doc.sections:
            if section.part == "back" and not self.include_back:
                continue
            units = self._units(section.blocks)
            if not units:
                continue
            base = slug(section.path[-1])
            n = seen.get(base, 0)
            seen[base] = n + 1
            sec = base if n == 0 else f"{base}-{n}"
            for i, c in enumerate(self.pack(units)):
                out.append(self.make_chunk(doc, f"{did}__{sec}__{i:03d}", c["text"],
                                           section.path, c["pages"], "body", section.part,
                                           self.header))

        if self.include_captions:
            for i, fig in enumerate(doc.figures):
                if not fig.caption:
                    continue
                label = "Table" if fig.type == "table" else "Figure"
                name = f"{label} {fig.label}" if fig.label else label
                pages = {fig.page} if fig.page else set()
                out.append(self.make_chunk(doc, f"{did}__caption__{i:03d}", fig.caption,
                                           [f"{name} caption"], pages, "caption", "body",
                                           self.header))
        return out
