"""Token counting and sentence splitting with no required dependencies."""

from __future__ import annotations

import re

try:  # optional: pip install scipubrag[tokens]
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))

    TOKENIZER = "tiktoken/cl100k_base"
except Exception:

    def count_tokens(text: str) -> int:
        # Scientific prose averages about 1.3 tokens per word
        return int(len(text.split()) * 1.3) + 1

    TOKENIZER = "words*1.3"


_ABBREVIATIONS = [
    "et al.", "e.g.", "i.e.", "cf.", "vs.", "Fig.", "Figs.", "Eq.", "Eqs.", "Ref.", "Refs.",
    "Sec.", "Tab.", "No.", "approx.", "ca.", "resp.", "Dr.", "Prof.", "St.", "Jr.", "Inc.",
    "Ltd.", "Co.", "Vol.", "pp.", "min.", "max.",
]
_MARK = "\u0000"


def split_sentences(text: str) -> list[str]:
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _MARK))
    protected = re.sub(r"(\d)\.(\d)", rf"\1{_MARK}\2", protected)  # decimals
    protected = re.sub(r"\b([A-Z])\.\s(?=[A-Z])", rf"\1{_MARK} ", protected)  # initials
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])", protected)
    return [p.replace(_MARK, ".").strip() for p in parts if p.strip()]


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "section"
