import pytest

from scipubrag.chunking import FixedSizeChunker, SectionChunker, count_tokens, split_sentences


def test_split_sentences_handles_abbreviations():
    text = "As shown by Smith et al. in Fig. 2, values were 3.5 mg. Next one. J. Smith agreed."
    assert split_sentences(text) == [
        "As shown by Smith et al. in Fig. 2, values were 3.5 mg.", "Next one.",
        "J. Smith agreed."]


def test_section_chunker_limits_and_boundaries(docs):
    chunker = SectionChunker(target=150, max_tokens=200, min_tokens=30)
    chunks = chunker.chunk_all(docs)
    body = [c for c in chunks if c.kind == "body"]
    assert all(c.n_tokens <= 200 + 30 for c in body)
    # every chunk belongs to exactly one section path
    for c in body:
        assert len(c.section_path) >= 1
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_section_chunker_kinds_and_headers(docs):
    chunks = SectionChunker().chunk(docs[0])
    kinds = {c.kind for c in chunks}
    assert kinds == {"abstract", "body", "caption"}
    first_body = next(c for c in chunks if c.kind == "body")
    assert first_body.text_for_embedding.startswith("A study of soil > 1 Introduction\n\n")
    assert first_body.text in first_body.text_for_embedding


def test_formula_stays_with_paragraph(docs):
    methods = [c for c in SectionChunker().chunk(docs[0]) if c.section_path == ["2 Methods"]]
    assert any("y = ax + b" in c.text for c in methods)


def test_include_back_false(docs):
    chunks = SectionChunker(include_back=False).chunk(docs[0])
    assert all(c.part == "body" for c in chunks)


def test_overlap_shares_a_sentence(docs):
    chunks = [c for c in SectionChunker(target=120, max_tokens=200, min_tokens=20, overlap=1)
              .chunk(docs[0]) if c.section_path == ["1 Introduction"]]
    assert len(chunks) >= 2
    last_sentence = split_sentences(chunks[0].text)[-1]
    assert chunks[1].text.startswith(last_sentence)


def test_invalid_sizes():
    with pytest.raises(ValueError):
        SectionChunker(target=600, max_tokens=512)


def test_fixed_chunker_covers_everything(docs):
    chunks = FixedSizeChunker(size=100, overlap=20).chunk(docs[0])
    assert len(chunks) > 3
    assert all(count_tokens(c.text) <= 110 for c in chunks)
    assert chunks[0].text_for_embedding == chunks[0].text  # no header in the baseline
    assert chunks[-1].pages  # citation location survives
