"""BM25 keyword index over chunks (rebuilt in memory at load time; no pickles)."""

from __future__ import annotations

import math
import re

import numpy as np
from rank_bm25 import BM25Okapi

from scipubrag.stores.base import ScoredId, Where, matches
from scipubrag.types import Chunk

_STOPWORDS = set("""
a an and are as at be been but by can could did do does for from had has have how if in into
is it its may might of on or our such than that the their them then there these they this those
to was we were what when where which while who why will with would you your also however using
used use based between both each more most other
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase tokens that keep 'bert-base', 'co2', 'f1-score' and '0.05' intact."""
    tokens = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class _LuceneBM25(BM25Okapi):
    """Okapi BM25 with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

    The classic Okapi IDF is zero for a term in exactly half the documents and
    negative above that, which makes matching terms invisible in small corpora.
    Lucene's variant is always positive.
    """

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.ids = [c.chunk_id for c in chunks]
        self._meta = [c.flat_metadata() for c in chunks]
        # Index the context header too, so titles and section names are searchable
        corpus = [tokenize(c.text_for_embedding) for c in chunks]
        self._bm25 = _LuceneBM25(corpus, k1=k1, b=b) if chunks else None

    def search(self, query: str, k: int = 10, where: Where | None = None) -> list[ScoredId]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        if where:
            mask = np.fromiter((matches(m, where) for m in self._meta), dtype=bool,
                               count=len(self._meta))
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)
        return [ScoredId(self.ids[i], float(scores[i])) for i in order[:k]
                if np.isfinite(scores[i]) and scores[i] > 0]
