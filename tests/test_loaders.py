from scipubrag.loaders import MarkdownLoader, make_doc_id, parse_tei
from scipubrag.types import Document


def test_parse_tei_header(tei_path):
    doc = parse_tei(tei_path)
    assert doc.doc_id == "smith2021"
    assert doc.title == "Attention in Soil Microbes"
    assert doc.authors == ["Jane A Smith", "Bo Li"]
    assert doc.year == "2021"
    assert doc.doi == "10.1000/xyz"
    assert doc.abstract == "We study things."


def test_parse_tei_sections(tei_path):
    doc = parse_tei(tei_path)
    paths = [s.path for s in doc.sections]
    assert ["2 Methods", "2.1 Data"] in paths            # hierarchy rebuilt from numbering
    assert ["Acknowledgements"] not in paths             # skipped by default
    assert ["Appendix A"] in paths                       # back matter kept
    data = next(s for s in doc.sections if s.path[-1] == "2.1 Data")
    assert [b.type for b in data.blocks] == ["p", "formula", "p"]  # untitled div merged in
    assert data.blocks[0].page == 2 and data.blocks[-1].page == 3


def test_parse_tei_options(tei_path):
    keep = parse_tei(tei_path)
    strip = parse_tei(tei_path, strip_citations=True, include_acknowledgements=True)
    assert "[3]" in keep.sections[0].blocks[0].text
    assert strip.sections[0].blocks[0].text == "Prior work showed X."
    assert ["Acknowledgements"] in [s.path for s in strip.sections]


def test_parse_tei_figures(tei_path):
    figs = parse_tei(tei_path).figures
    assert [f.type for f in figs] == ["figure", "table"]
    assert figs[0].caption == "Figure 1. Results overview." and figs[0].page == 3


def test_document_roundtrip(tei_path):
    doc = parse_tei(tei_path)
    assert Document.from_dict(doc.to_dict()) == doc


def test_make_doc_id(tmp_path):
    assert make_doc_id(tmp_path / "Smith 2021 (final).pdf") == "smith-2021-final"
    assert make_doc_id(tmp_path / "paper1.grobid.tei.xml") == "paper1"


def test_markdown_loader(tmp_path):
    md = tmp_path / "Notes.md"
    md.write_text("---\nauthors: A. Writer\nyear: 2024\n---\n# My Notes\n\nIntro text.\n\n"
                  "## Setup\n\nFirst para\ncontinues here.\n\n### Details\n\nDeep text.\n")
    doc = MarkdownLoader().load(md)
    assert doc.title == "My Notes" and doc.year == "2024" and doc.authors == ["A. Writer"]
    assert [s.path for s in doc.sections] == [["Body"], ["Setup"], ["Setup", "Details"]]
    assert doc.sections[1].blocks[0].text == "First para continues here."
