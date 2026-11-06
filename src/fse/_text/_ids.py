"""Parallel document IDs as lists, NumPy arrays or primitive Arrow columns."""

from __future__ import annotations

from typing import Any

from fse import _native
from fse._ids import checked_ids, id_block


class IdSource:
    """Own an Arrow export, but only copy the blocks needed by pending tasks."""

    def __init__(self, value: Any):
        kind = type(value)
        self._arrow = None
        if hasattr(kind, "__arrow_c_stream__"):
            self._arrow = _native.ArrowIds.from_stream(value.__arrow_c_stream__())
        elif hasattr(kind, "__arrow_c_array__"):
            self._arrow = _native.ArrowIds.from_array(*value.__arrow_c_array__())
        else:
            self._values = checked_ids(value)

    def __len__(self) -> int:
        return len(self._values) if self._arrow is None else self._arrow.rows

    def block(self, begin: int, end: int) -> Any:
        return (
            id_block(self._values, begin, end)
            if self._arrow is None
            else self._arrow.block(begin, end)
        )

    def __enter__(self) -> IdSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._arrow is not None:
            self._arrow.release()

    def window(self, first: int, last: int) -> IdWindow:
        """The IDs of rows ``first`` to ``last - 1`` as a source of their own."""
        if not 0 <= first <= last <= len(self):
            raise ValueError(f"rows {first} to {last} are not rows of {len(self)} IDs")
        return IdWindow(self, first, last)


class IdWindow:
    """A part of the IDs of an ``IdSource`` (which stays open for as long as it is used),
    numbered from 0."""

    def __init__(self, source: IdSource, first: int, last: int):
        self._source, self._first, self._last = source, first, last

    def __len__(self) -> int:
        return self._last - self._first

    def block(self, begin: int, end: int) -> Any:
        return self._source.block(self._first + begin, self._first + end)
