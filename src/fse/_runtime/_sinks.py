#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Where the runtime puts finished rows (ARCHITECTURE.md, "Runtime"; PR 6 adds more).

The runtime calls both methods in the calling thread: ``target`` reserves the rows of a
task before the task is submitted, ``commit`` hands them back once they are finished, in
row order, each row once. A target is written only by the worker of its task and not
reused before its commit; after an error nothing more is committed.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy import ndarray


class Sink(Protocol):
    dim: int

    def target(self, first_row: int, n_rows: int) -> ndarray:
        """Writable C-contiguous float32 rows ``(n_rows, dim)`` for one task."""
        ...

    def commit(self, first_row: int, rows: ndarray) -> None: ...


class _Rows:
    def __init__(self, dim: int, start: int):
        self.dim = dim
        self.next_row = start

    def _check_commit(self, first_row: int, rows: ndarray):
        if first_row != self.next_row:
            raise ValueError(f"rows from {first_row} committed, expected rows from {self.next_row}")
        if rows.ndim != 2 or rows.shape[1] != self.dim:
            raise ValueError(f"rows of {self.dim} values expected, not shape {rows.shape}")
        self.next_row += len(rows)


class ArrayRows(_Rows):
    """A known number of rows in one array; a target is a view of it, so nothing is copied."""

    def __init__(self, rows: int, dim: int):
        super().__init__(dim, 0)
        self._out = np.empty((rows, dim), dtype=np.float32)

    def target(self, first_row: int, n_rows: int) -> ndarray:
        if not 0 <= first_row <= first_row + n_rows <= len(self._out):
            raise ValueError(f"rows {first_row} to {first_row + n_rows} beyond {len(self._out)}")
        return self._out[first_row : first_row + n_rows]

    def commit(self, first_row: int, rows: ndarray) -> None:
        self._check_commit(first_row, rows)

    def result(self) -> ndarray:
        if self.next_row != len(self._out):
            raise RuntimeError(f"{self.next_row} of {len(self._out)} rows were committed")
        return self._out


class GrowingRows(_Rows):
    """Rows of a stream whose length is not known in advance, joined at the end."""

    def __init__(self, dim: int):
        super().__init__(dim, 0)
        self._parts: list[ndarray] = []

    def target(self, first_row: int, n_rows: int) -> ndarray:
        if first_row < self.next_row:
            raise ValueError(f"rows from {first_row} were committed already")
        return np.empty((n_rows, self.dim), dtype=np.float32)

    def commit(self, first_row: int, rows: ndarray) -> None:
        self._check_commit(first_row, rows)
        self._parts.append(rows)

    def result(self) -> ndarray:
        if not self._parts:
            return np.empty((0, self.dim), dtype=np.float32)
        out = np.concatenate(self._parts) if len(self._parts) > 1 else self._parts[0]
        self._parts = [out]
        return out
