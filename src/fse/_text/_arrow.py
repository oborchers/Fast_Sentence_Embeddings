#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Arrow string columns as a source: any object that exports one through the PyCapsule
interface (``__arrow_c_stream__`` or ``__arrow_c_array__``), such as a pyarrow Array or
ChunkedArray, a pandas 3 Series or a Polars Series, read in place by ``_native`` without
the GIL (ARCHITECTURE.md, "Text sources").

The whole column is taken over when the source is made, in the calling thread, so its
rows are known and every row is checked by the scan before the first one is encoded.
Rows are numbered across the chunks of the column, and a chunk of the runtime may span
several of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fse import _log, _native
from fse._contract import Batch

from ._scan import Chunk, Scan, _Buffered


def is_arrow(texts: object) -> bool:
    """Whether ``texts`` exports Arrow data through the PyCapsule interface."""
    kind = type(texts)
    return hasattr(kind, "__arrow_c_stream__") or hasattr(kind, "__arrow_c_array__")


def _import(texts: Any) -> _native.ArrowColumn:
    """The column of ``texts``: the stream if it exports one, else its one array. No
    schema is requested; fse reads every string layout itself."""
    if hasattr(type(texts), "__arrow_c_stream__"):
        return _native.ArrowColumn.from_stream(texts.__arrow_c_stream__())
    return _native.ArrowColumn.from_array(*texts.__arrow_c_array__())


class ArrowSource(_Buffered):
    """The rows of one Arrow string column, owned until ``close`` (or the end of the call
    that made it)."""

    def __init__(self, texts: Any):
        if not is_arrow(texts):
            raise TypeError(f"{type(texts).__name__} exports no Arrow data")
        self._column = _import(texts)
        self._data = None
        self._offsets = None
        if self._column.copied_bytes:
            _log.emit(
                "runtime.arrow_copied",
                "fse runtime: the offsets or views of the Arrow column were not aligned: fse "
                "copied %d bytes of them (no text)",
                self._column.copied_bytes,
                bytes=self._column.copied_bytes,
            )

    def __len__(self) -> int:
        return self._column.rows

    @property
    def nbytes(self) -> int:
        """The bytes of the UTF-8 text of all rows."""
        return self._column.nbytes

    @property
    def copied_bytes(self) -> int:
        """The bytes fse copied of this column (only misaligned offsets or views)."""
        return self._column.copied_bytes

    def _positions(self) -> int:
        return self._column.rows

    def _start_at_or_after(self, byte: int) -> int:
        """A row near ``byte`` of the text: the chunk that holds it, then in proportion
        within the chunk (scan ranges only balance the work)."""
        starts, bytes_before = self._column.chunk_starts, self._column.byte_starts
        k = int(np.searchsorted(bytes_before, byte, side="right")) - 1
        if k >= len(starts) - 1:
            return int(starts[-1])
        size = int(bytes_before[k + 1] - bytes_before[k])
        rows = int(starts[k + 1] - starts[k])
        within = (byte - int(bytes_before[k])) * rows // size if size else 0
        return int(starts[k]) + within

    def _lines(self) -> bool:
        return False  # positions are row numbers

    def scan_until_error(
        self,
        *,
        workers: int = 1,
        chunk_bytes: int | None = None,
        chunk_rows: int | None = None,
        first_row: int = 0,
    ) -> tuple[Scan, Exception | None]:
        """As for every source; a null ends the column like an invalid sentence: the rows
        before it are the chunks, and its error follows them unless one of them is
        invalid."""
        scan, error = super().scan_until_error(
            workers=workers, chunk_bytes=chunk_bytes, chunk_rows=chunk_rows, first_row=first_row
        )
        if error is None and self._column.null_row >= 0:
            error = _native.arrow_null_error(first_row + self._column.null_row)
        return scan, error

    def _scanner(self, chunk_bytes: int, chunk_rows: int):
        self._check_open()
        column = self._column

        def scan(span: tuple[int, int]):
            return column.scan(span[0], span[1], chunk_bytes, chunk_rows)

        return scan

    def _span_bytes(self, chunk: Chunk) -> int:
        return self._column.span_bytes(chunk.start, chunk.end)

    def _error(self, byte: int, sentence: int, first_row: int = 0) -> UnicodeDecodeError:
        raw = self._column.row_bytes(sentence)
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as err:
            return UnicodeDecodeError(
                err.encoding,
                err.object,
                err.start,
                err.end,
                f"{err.reason} in sentence {first_row + sentence} (byte {byte} of the row)",
            )
        raise AssertionError(f"no UTF-8 error in row {sentence}")  # pragma: no cover

    def _check_open(self) -> None:
        if self._column.released:
            raise ValueError("the ArrowSource is closed")

    def encode(
        self, chunk: Chunk, lookup: _native.TextLookup, workspace: _native.Workspace
    ) -> Batch:
        self._check_open()
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunk must be a Chunk, not {type(chunk).__name__}")
        return self._column.encode(lookup, chunk.start, chunk.end, chunk.first_row, workspace)

    def close(self) -> None:
        """Release the Arrow arrays; raises while a worker still reads them."""
        self._column.release()

    def __enter__(self) -> ArrowSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
