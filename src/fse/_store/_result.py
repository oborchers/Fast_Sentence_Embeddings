"""Position-based search results, with optional document IDs outside the tuple fields."""

from __future__ import annotations

from typing import NamedTuple

from numpy import ndarray


class SearchResult(NamedTuple):
    """Positions (int64) and cosine scores (float32), best first."""

    ids: ndarray
    scores: ndarray


class DocumentSearchResult(SearchResult):
    """Keep two-item unpacking while exposing one document ID per hit."""

    doc_ids: ndarray

    def __new__(cls, ids: ndarray, scores: ndarray, doc_ids: ndarray):
        result = super().__new__(cls, ids, scores)
        result.doc_ids = doc_ids
        return result

    def __reduce__(self):
        return type(self), (self.ids, self.scores, self.doc_ids)
