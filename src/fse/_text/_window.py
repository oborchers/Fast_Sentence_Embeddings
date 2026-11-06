#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""A window onto the rows of a source that sentences share as one UTF-8 buffer.

A resumed run reads a part of a source: the rows after those the store holds, up to the end
of the partition. The window is such a part as a source of its own, whose first row is row 0
of the run. It shares the bytes of the source (no copy of the text; a window onto strings
copies only their offsets) and delegates the checks of the source it came from: whether it
is still open, and whether the file it reads has changed (docs/architecture/runtime.md,
"Row bases").
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fse._contract import Batch

from ._scan import Chunk, _Buffered


class Window(_Buffered):
    """Rows ``first`` to ``last - 1`` of ``base`` (a source of strings, of token lists or
    of the lines of a file) as a source of their own."""

    def __init__(self, base: Any, data: Any, offsets: Any, rows: int):
        self._base = base
        self._data, self._offsets = data, offsets
        self._rows = rows
        self._verified = bool(getattr(base, "_verified", False))

    @classmethod
    def of_rows(cls, base: Any, first: int, last: int) -> Window:
        """The rows ``first`` to ``last - 1`` of a source of strings or of token lists."""
        if not 0 <= first <= last <= len(base):
            raise ValueError(f"rows {first} to {last} are not rows of a source of {len(base)}")
        low, high = int(base._offsets[first]), int(base._offsets[last])
        offsets = base._offsets[first : last + 1] - low
        offsets.flags.writeable = False
        return cls(base, memoryview(base._data)[low:high], offsets, last - first)

    @classmethod
    def of_bytes(cls, base: Any, first_byte: int, last_byte: int, rows: int) -> Window:
        """The ``rows`` lines of a file that lie between two bytes (a line starts at the
        first one and the last one ends the last line)."""
        data = memoryview(base._data)[first_byte:last_byte] if last_byte > first_byte else b""
        return cls(base, data, None, rows)

    def __len__(self) -> int:
        return self._rows

    def _check_open(self) -> None:
        self._base._check_open()

    def _after_encode(self) -> Any:
        return self._base._after_encode()

    def encode(self, chunk: Chunk, lookup: Any, workspace: Any) -> Batch:
        batch = super().encode(chunk, lookup, workspace)
        self._base._check_open()  # a change of the file while the chunk was encoded
        return batch

    def close(self) -> None:
        """Let go of the bytes of the source (a window has no file of its own)."""
        if isinstance(self._data, memoryview):
            self._data.release()
        self._data = b""
        self._offsets = None if self._offsets is None else np.zeros(1, np.int64)
        self._rows = 0

    def __enter__(self) -> Window:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
