"""Core data types shared by every stage of the pipeline.

Document  -> what a Loader produces (a parsed paper, a markdown file, ...)
Chunk     -> what a Chunker produces and what gets embedded and indexed
Hit       -> a Chunk returned by a search, with its score
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    """A paragraph-level unit of text inside a section."""

    type: str  # "p" | "formula"
    text: str
    page: int | None = None


@dataclass
class Section:
    heading: str
    path: list[str]  # e.g. ["2 Methods", "2.1 Data"]
    blocks: list[Block]
    number: str | None = None
    part: str = "body"  # "body" | "back"


@dataclass
class Figure:
    id: str
    type: str  # "figure" | "table"
    caption: str
    label: str | None = None
    page: int | None = None


@dataclass
class Document:
    doc_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: str | None = None
    doi: str | None = None
    abstract: str = ""
    sections: list[Section] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    source: str | None = None  # original file path
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Document:
        return cls(
            doc_id=d.get("doc_id") or d["paper_id"],  # accept the older JSON format
            title=d.get("title", ""),
            authors=list(d.get("authors", [])),
            year=d.get("year"),
            doi=d.get("doi"),
            abstract=d.get("abstract", ""),
            sections=[
                Section(
                    heading=s["heading"],
                    path=list(s["path"]),
                    blocks=[Block(**b) for b in s["blocks"]],
                    number=s.get("number"),
                    part=s.get("part", "body"),
                )
                for s in d.get("sections", [])
            ],
            figures=[Figure(**f) for f in d.get("figures", [])],
            source=d.get("source"),
            metadata=dict(d.get("metadata", {})),
        )

    @property
    def first_author(self) -> str:
        return self.authors[0].split()[-1] if self.authors else ""


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str  # what the LLM sees
    text_for_embedding: str  # what gets embedded (text plus context header)
    kind: str = "body"  # "body" | "abstract" | "caption"
    part: str = "body"  # "body" | "back"
    section_path: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    n_tokens: int = 0
    title: str = ""
    year: str | None = None
    first_author: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in d.items() if k in known}
        if "doc_id" not in data and "paper_id" in d:
            data["doc_id"] = d["paper_id"]  # older chunk files
        return cls(**data)

    def flat_metadata(self) -> dict[str, str | int | float | bool]:
        """Scalar-only metadata, usable by any vector store for filtering."""
        meta: dict[str, str | int | float | bool] = {
            "doc_id": self.doc_id,
            "kind": self.kind,
            "part": self.part,
            "section": " > ".join(self.section_path),
            "pages": ",".join(str(p) for p in self.pages),
            "page_start": self.pages[0] if self.pages else -1,
            "n_tokens": self.n_tokens,
            "title": self.title,
            "year": self.year or "",
            "first_author": self.first_author,
        }
        for k, v in self.extra.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        return meta

    def citation(self) -> str:
        """Short human-readable source label, e.g. 'Smith 2021, 2.1 Data, p. 4'."""
        who = " ".join(x for x in [self.first_author, self.year or ""] if x) or self.title
        where = self.section_path[-1] if self.section_path else ""
        pages = f"p. {self.pages[0]}" if len(self.pages) == 1 else (
            f"pp. {self.pages[0]}-{self.pages[-1]}" if self.pages else ""
        )
        return ", ".join(x for x in [who, where, pages] if x)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str = ""  # which retriever produced it: "dense", "bm25", "hybrid", ...
    rank: int = 0
