"""Position-based search results, with optional document IDs outside the tuple fields."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NamedTuple, Self

import numpy as np
from numpy.typing import NDArray


class SearchResult(NamedTuple):
    """Positions (int64) and cosine scores (float32), best first."""

    ids: NDArray[np.int64]
    scores: NDArray[np.float32]


_FIELDS = ("ids", "scores", "doc_ids")


class DocumentSearchResult(SearchResult):
    """Keep two-item unpacking while exposing one document ID per hit.

    Unpacking, ``len`` (2), indexing and equality use the two tuple fields ``ids`` and
    ``scores``. ``doc_ids`` is an attribute that ``_replace``, ``_asdict`` and ``repr``
    keep; ``_make`` raises because two values hold no document IDs.
    """

    doc_ids: NDArray[np.int64]

    def __new__(
        cls, ids: NDArray[np.int64], scores: NDArray[np.float32], doc_ids: NDArray[np.int64]
    ):
        result = super().__new__(cls, ids, scores)
        result.doc_ids = doc_ids
        return result

    def __reduce__(self):
        return type(self), (self.ids, self.scores, self.doc_ids)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ids={self.ids!r}, scores={self.scores!r}, "
            f"doc_ids={self.doc_ids!r})"
        )

    def _asdict(self) -> dict[str, NDArray[Any]]:
        return {"ids": self.ids, "scores": self.scores, "doc_ids": self.doc_ids}

    def _replace(self, /, **changes: Any) -> DocumentSearchResult:
        unknown = set(changes) - set(_FIELDS)
        if unknown:
            raise TypeError(f"Got unexpected field names: {sorted(unknown)!r}")
        values = {**self._asdict(), **changes}
        return type(self)(values["ids"], values["scores"], values["doc_ids"])

    __replace__ = _replace

    @classmethod
    # mypy types the generated NamedTuple._make with a type variable that no override matches
    def _make(cls, iterable: Iterable[Any]) -> Self:  # type: ignore[override]
        del iterable
        raise TypeError(
            "DocumentSearchResult needs doc_ids; use DocumentSearchResult(ids, scores, doc_ids)"
        )
