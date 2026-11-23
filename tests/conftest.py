from pathlib import Path

import pytest

from scipubrag.types import Block, Document, Figure, Section

FIXTURES = Path(__file__).parent / "fixtures"

_TOPICS = {
    "soil": "Soil nitrogen was measured at twelve field sites using mass spectrometry.",
    "vision": "The convolutional network was trained on labeled satellite imagery tiles.",
    "climate": "Regional climate models projected rainfall under three emission scenarios.",
}


def make_doc(doc_id: str, topic: str, n_paragraphs: int = 6) -> Document:
    sentence = _TOPICS[topic]
    para = " ".join([sentence] * 8)
    return Document(
        doc_id=doc_id,
        title=f"A study of {topic}",
        authors=["Ada Lovelace", "Alan Turing"],
        year="2023",
        abstract=f"We study {topic}. " + sentence,
        sections=[
            Section("Introduction", ["1 Introduction"],
                    [Block("p", para, 1) for _ in range(n_paragraphs)], "1"),
            Section("Methods", ["2 Methods"],
                    [Block("p", para, 2), Block("formula", "y = ax + b", 2)], "2"),
            Section("Appendix", ["Appendix A"], [Block("p", "Extra appendix material.", 9)],
                    part="back"),
        ],
        figures=[Figure("fig_0", "figure", f"Overview of the {topic} results.", "1", 3)],
    )


@pytest.fixture
def docs() -> list[Document]:
    return [make_doc("soil2023", "soil"), make_doc("vision2023", "vision"),
            make_doc("climate2023", "climate")]


@pytest.fixture
def tei_path() -> Path:
    return FIXTURES / "smith2021.tei.xml"
