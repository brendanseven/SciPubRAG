from scipubrag.chunking.base import Chunker
from scipubrag.chunking.fixed import FixedSizeChunker
from scipubrag.chunking.section import SectionChunker
from scipubrag.chunking.text import TOKENIZER, count_tokens, split_sentences

__all__ = ["TOKENIZER", "Chunker", "FixedSizeChunker", "SectionChunker", "count_tokens",
           "get_chunker", "split_sentences"]


def get_chunker(cfg) -> Chunker:
    c = cfg.chunking
    if c.strategy == "section":
        return SectionChunker(c.target, c.max, c.min, c.overlap, c.include_captions,
                              c.include_back)
    if c.strategy == "fixed":
        return FixedSizeChunker(size=c.max, overlap=c.overlap * 10)
    raise ValueError(f"Unknown chunking strategy {c.strategy!r} (expected 'section' or 'fixed')")
