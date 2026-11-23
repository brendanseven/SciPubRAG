"""Parse GROBID TEI XML into a Document (standard library only)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from scipubrag.loaders.base import make_doc_id
from scipubrag.types import Block, Document, Figure, Section

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


def clean(text: str) -> str:
    """Collapse whitespace and fix the spacing GROBID leaves around punctuation."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.;:)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def element_text(el: ET.Element | None, strip_citations: bool = False) -> str:
    if el is None:
        return ""
    if not strip_citations:
        return clean("".join(el.itertext()))

    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        is_bibr = node.tag == f"{TEI}ref" and node.get("type") == "bibr"
        if node.text and not is_bibr:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    text = re.sub(r"[\[(]\s*[,;\s]*[\])]", "", clean("".join(parts)))  # empty brackets
    return clean(text)


def page_of(el: ET.Element) -> int | None:
    """First page from GROBID's coords attribute: 'page,x,y,w,h;page,...'."""
    coords = el.get("coords")
    if not coords:
        return None
    try:
        return int(coords.split(";")[0].split(",")[0])
    except (ValueError, IndexError):
        return None


def _parse_header(root: ET.Element) -> dict:
    header = root.find("tei:teiHeader", NS)
    out = {"title": "", "authors": [], "year": None, "doi": None, "abstract": ""}
    if header is None:
        return out

    out["title"] = element_text(header.find(".//tei:titleStmt/tei:title", NS))

    for pers in header.findall(".//tei:sourceDesc//tei:analytic/tei:author/tei:persName", NS):
        forenames = " ".join(clean(f.text or "") for f in pers.findall("tei:forename", NS))
        surname = clean(pers.findtext("tei:surname", default="", namespaces=NS))
        full = clean(f"{forenames} {surname}")
        if full:
            out["authors"].append(full)

    for xpath in (".//tei:publicationStmt/tei:date", ".//tei:sourceDesc//tei:imprint/tei:date"):
        date_el = header.find(xpath, NS)
        if date_el is not None:
            match = re.search(r"\d{4}", date_el.get("when") or date_el.text or "")
            if match:
                out["year"] = match.group(0)
                break

    doi_el = header.find(".//tei:idno[@type='DOI']", NS)
    if doi_el is not None and doi_el.text:
        out["doi"] = clean(doi_el.text)

    abstract_el = header.find(".//tei:profileDesc/tei:abstract", NS)
    if abstract_el is not None:
        paras = [element_text(p) for p in abstract_el.iter(f"{TEI}p")]
        out["abstract"] = "\n\n".join(p for p in paras if p) or element_text(abstract_el)
    return out


def _parse_divs(container: ET.Element | None, part: str, strip_citations: bool,
                skip_types: set[str]) -> list[Section]:
    """Flatten <div>s into Sections, rebuilding hierarchy from head/@n ('3' > '3.1')."""
    if container is None:
        return []

    skipped: set[int] = set()
    for div in container.iter(f"{TEI}div"):
        if div.get("type") in skip_types:
            skipped.update(id(d) for d in div.iter(f"{TEI}div"))

    sections: list[Section] = []
    numbered: dict[str, str] = {}
    last_path: list[str] = []

    for div in container.iter(f"{TEI}div"):
        if id(div) in skipped:
            continue
        head_el = div.find("tei:head", NS)
        has_blocks = any(c.tag in (f"{TEI}p", f"{TEI}formula") for c in div)
        if not has_blocks and head_el is None:
            continue

        heading = element_text(head_el)
        number = head_el.get("n") if head_el is not None else None
        if heading:
            label = f"{number} {heading}" if number else heading
            if number:
                numbered[number] = label
                pieces = number.rstrip(".").split(".")
                parents = [numbered[".".join(pieces[:i])] for i in range(1, len(pieces))
                           if ".".join(pieces[:i]) in numbered]
                path = parents + [label]
            else:
                path = [label]
            last_path = path
        else:
            path = last_path or (["Back matter"] if part == "back" else ["Body"])
            heading = path[-1]

        blocks: list[Block] = []
        for child in div:
            if child.tag == f"{TEI}p":
                text = element_text(child, strip_citations)
                if text:
                    blocks.append(Block("p", text, page_of(child)))
            elif child.tag == f"{TEI}formula":
                text = element_text(child)
                if text:
                    blocks.append(Block("formula", text, page_of(child)))
        if not blocks:
            continue

        if sections and sections[-1].path == path and sections[-1].part == part:
            sections[-1].blocks.extend(blocks)  # GROBID split one section in two
        else:
            sections.append(Section(heading=heading, path=path, blocks=blocks,
                                    number=number, part=part))
    return sections


def _parse_figures(text_el: ET.Element | None) -> list[Figure]:
    if text_el is None:
        return []
    figures = []
    for i, fig in enumerate(text_el.iter(f"{TEI}figure")):
        caption = element_text(fig.find("tei:figDesc", NS))
        head = element_text(fig.find("tei:head", NS))
        label = element_text(fig.find("tei:label", NS)) or None
        if not (caption or head):
            continue
        full = clean(f"{head}. {caption}") if head and caption and head not in caption \
            else (caption or head)
        figures.append(Figure(
            id=fig.get(XML_ID) or f"fig_{i}",
            type="table" if fig.get("type") == "table" else "figure",
            caption=full, label=label, page=page_of(fig),
        ))
    return figures


def parse_tei(path: str | Path, strip_citations: bool = False,
              include_acknowledgements: bool = False, doc_id: str | None = None) -> Document:
    path = Path(path)
    root = ET.parse(path).getroot()
    text_el = root.find("tei:text", NS)
    body = text_el.find("tei:body", NS) if text_el is not None else None
    back = text_el.find("tei:back", NS) if text_el is not None else None

    back_skip = {"references"}
    if not include_acknowledgements:
        back_skip |= {"acknowledgement", "acknowledgment", "funding"}

    header = _parse_header(root)
    return Document(
        doc_id=doc_id or make_doc_id(path),
        sections=_parse_divs(body, "body", strip_citations, set())
        + _parse_divs(back, "back", strip_citations, back_skip),
        figures=_parse_figures(text_el),
        source=str(path),
        **header,
    )
