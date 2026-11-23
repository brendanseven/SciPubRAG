"""Markdown / plain-text loader. A template for reusing the pipeline on non-PDF corpora."""

from __future__ import annotations

import re
from pathlib import Path

from scipubrag.loaders.base import Loader, make_doc_id
from scipubrag.types import Block, Document, Section

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


class MarkdownLoader(Loader):
    """Sections come from '#' headings; paragraphs are blank-line separated.

    The first level-1 heading becomes the title. Optional front matter lines like
    'authors: A, B' and 'year: 2024' at the top of the file are read as metadata.
    """

    extensions = (".md", ".markdown", ".txt")

    def load(self, path: Path) -> Document:
        path = Path(path)
        lines = path.read_text(encoding="utf-8").splitlines()

        meta: dict[str, str] = {}
        if lines and lines[0].strip() == "---":
            end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
            if end:
                for line in lines[1:end]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip().lower()] = v.strip()
                lines = lines[end + 1:]

        title = meta.get("title", "")
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []  # (level, heading)
        para: list[str] = []
        current: Section | None = None

        def flush_para() -> None:
            nonlocal current
            text = " ".join(s.strip() for s in para).strip()
            para.clear()
            if not text:
                return
            if current is None:
                current = Section(heading="Body", path=["Body"], blocks=[])
                sections.append(current)
            current.blocks.append(Block("p", text))

        for line in lines:
            m = _HEADING.match(line)
            if m:
                flush_para()
                level, heading = len(m.group(1)), m.group(2).strip()
                if level == 1 and not title:
                    title = heading
                    continue
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
                current = Section(heading=heading, path=[h for _, h in stack], blocks=[])
                sections.append(current)
            elif line.strip():
                para.append(line)
            else:
                flush_para()
        flush_para()

        authors = [a.strip() for a in meta.get("authors", "").split(",") if a.strip()]
        return Document(
            doc_id=make_doc_id(path),
            title=title or path.stem,
            authors=authors,
            year=meta.get("year"),
            abstract=meta.get("abstract", ""),
            sections=[s for s in sections if s.blocks],
            source=str(path),
        )
