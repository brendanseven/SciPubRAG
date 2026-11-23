"""Loader interface: turn source files into Documents."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from scipubrag.types import Document


def make_doc_id(path: Path) -> str:
    """File name without extensions, made id-safe: 'Smith 2021.tei.xml' -> 'smith-2021'."""
    name = path.name
    for suffix in (".grobid.tei.xml", ".tei.xml", ".xml", ".pdf", ".md", ".markdown", ".txt"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9_]+", "-", name.lower()).strip("-") or "doc"


class Loader(ABC):
    """Subclass this to support a new kind of source (HTML, Word, code, ...)."""

    #: file extensions this loader accepts, lowercase with the dot
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Load a single file into a Document."""

    def files(self, directory: Path) -> list[Path]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Input folder not found: {directory}")
        return sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in self.extensions
        )

    def load_dir(self, directory: Path) -> Iterator[Document]:
        for path in self.files(directory):
            yield self.load(path)
