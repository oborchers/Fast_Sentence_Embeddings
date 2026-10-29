#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Where the runtime puts finished rows (ARCHITECTURE.md, "Runtime"; PR 6 adds more).

The runtime calls both methods in the calling thread: ``target`` reserves the rows of a
task before the task is submitted, ``commit`` hands them back once they are finished, in
row order, each row once. A target is written only by the worker of its task and not
reused before its commit; after an error nothing more is committed. A sink with a
``dtype`` of float16 or int8 takes stored rows: the worker of a task stores its finished
rows into the target itself (stage 5, ``store`` of the backend).
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy import ndarray

from fse import _codes, _gram

# What a target and a commit hold: float32 rows, float16 codes, or int8 codes and scales;
# for a sink that wants inverse norms, a float32 vector of one per row follows.
Stored = ndarray | tuple[ndarray, ...]


class QuantizedRows(NamedTuple):
    """int8 sentence vectors: the codes and one float32 scale per row (row = codes * scale)."""

    codes: ndarray
    scales: ndarray


class Sink(Protocol):
    """Where an encoder puts the rows it finishes: ``fse.Sink``.

    ``dim`` values per row, stored as ``dtype`` (``float32`` if the sink has none). The
    encoder calls both methods in its calling thread: ``target`` reserves the rows of a
    task before the task runs, ``commit`` hands them back once they are finished, in row
    order, each row once; a target is valid only until its commit. Optional:
    ``inverse_norms`` (True: the target ends with a float32 vector the workers fill with
    the inverse norm of every stored row), ``encoder`` (the fingerprint of the only
    encoder the sink takes), ``flush()`` (called after the last commit of a successful
    run) and ``durable_rows`` (the number of leading rows that are durable, for a stream
    that confirms its input only after that; a sink without it promises nothing).
    """

    dim: int

    def target(self, first_row: int, n_rows: int) -> Stored:
        """Writable C-contiguous rows ``(n_rows, dim)`` for one task: float32 rows, float16
        codes, or for int8 the codes and a float32 vector of one scale per row; then the
        float32 vector of the inverse norms if the sink wants them."""
        ...

    def commit(self, first_row: int, rows: Stored) -> None: ...


def sink_dtype(sink: object) -> str:
    """The storage dtype of ``sink``: its ``dtype``, float32 without one."""
    return _codes.check_dtype(getattr(sink, "dtype", "float32"))


def wants_inverse_norms(sink: object) -> bool:
    """Whether the targets of ``sink`` end with a vector for the inverse norms."""
    wants = getattr(sink, "inverse_norms", False)
    if not isinstance(wants, bool):
        raise TypeError(f"the inverse_norms of a sink must be a bool, not {type(wants).__name__}")
    return wants


def split_target(target: Stored, dtype: str, inverse: bool) -> tuple[ndarray, Any, Any]:
    """The codes (or float32 rows), the int8 scales (or None) and the inverse norms (or
    None) of a target of a sink of ``dtype``."""
    parts = target if isinstance(target, tuple) else (target,)
    want = 1 + (dtype == "int8") + inverse
    if len(parts) != want or not all(isinstance(p, ndarray) for p in parts):
        raise ValueError(f"a target of a {dtype} sink holds {want} arrays, not {len(parts)}")
    codes = parts[0]
    scales = parts[1] if dtype == "int8" else None
    norms = parts[-1] if inverse else None
    _check_rows(codes, dtype)
    _check_apart(parts)
    for name, array in (("scales", scales), ("inverse norms", norms)):
        if array is not None and (
            array.dtype != np.float32
            or array.shape != (len(codes),)
            or not array.flags.writeable
            or not array.flags.c_contiguous
        ):
            raise ValueError(
                f"the {name} of a target must be a writable float32 vector of {len(codes)} values"
            )
    return codes, scales, norms


class _Rows:
    def __init__(self, dim: int, start: int, dtype: str = "float32"):
        self.dim = dim
        self.dtype = _codes.check_dtype(dtype)
        self.next_row = start

    def _new(self, n_rows: int) -> Stored:
        codes = np.empty((n_rows, self.dim), dtype=_codes.STORAGE_TYPES[self.dtype])
        if self.dtype == "int8":
            return QuantizedRows(codes, np.empty(n_rows, dtype=np.float32))
        return codes

    def _check_commit(self, first_row: int, rows: Stored) -> int:
        """The number of rows of a commit, checked: in order, of the dtype and dimension."""
        if first_row != self.next_row:
            raise ValueError(f"rows from {first_row} committed, expected rows from {self.next_row}")
        codes, scales = rows if isinstance(rows, tuple) else (rows, None)
        want = _codes.STORAGE_TYPES[self.dtype]
        if (
            not isinstance(codes, ndarray)
            or codes.dtype != want
            or codes.ndim != 2
            or codes.shape[1] != self.dim
            or not codes.flags.c_contiguous
            or (self.dtype == "int8")
            != (isinstance(scales, ndarray) and scales.shape == (len(codes),))
        ):
            raise ValueError(f"C-contiguous {self.dtype} rows of {self.dim} values expected")
        self.next_row += len(codes)
        return len(codes)


def _first(rows: Stored) -> ndarray:
    return rows[0] if isinstance(rows, tuple) else rows


class ArrayRows(_Rows):
    """A known number of rows in one array; a target is a view of it, so nothing is copied."""

    def __init__(self, rows: int, dim: int, dtype: str = "float32"):
        super().__init__(dim, 0, dtype)
        self._out = self._new(rows)

    def _view(self, first_row: int, n_rows: int) -> Stored:
        if isinstance(self._out, tuple):
            codes, scales = self._out
            return codes[first_row : first_row + n_rows], scales[first_row : first_row + n_rows]
        return self._out[first_row : first_row + n_rows]

    def target(self, first_row: int, n_rows: int) -> Stored:
        total = len(_first(self._out))
        if not 0 <= first_row <= first_row + n_rows <= total:
            raise ValueError(f"rows {first_row} to {first_row + n_rows} beyond {total}")
        return self._view(first_row, n_rows)

    def commit(self, first_row: int, rows: Stored) -> None:
        n = len(_first(rows)) if isinstance(_first(rows), ndarray) else 0
        reserved = self._view(first_row, n)
        if not _same(rows, reserved):
            raise ValueError(f"the rows from {first_row} are not the target reserved for them")
        self._check_commit(first_row, rows)

    def result(self) -> Stored:
        total = len(_first(self._out))
        if self.next_row != total:
            raise RuntimeError(f"{self.next_row} of {total} rows were committed")
        return self._out


def _same(rows: Stored, reserved: Stored) -> bool:
    """Whether ``rows`` are the very arrays reserved (same shape and memory)."""
    got = rows if isinstance(rows, tuple) else (rows,)
    want = reserved if isinstance(reserved, tuple) else (reserved,)
    return len(got) == len(want) and all(
        isinstance(g, ndarray)
        and g.shape == w.shape
        and g.__array_interface__["data"][0] == w.__array_interface__["data"][0]
        for g, w in zip(got, want, strict=True)
    )


class GrowingRows(_Rows):
    """Rows of a stream whose length is not known in advance, joined at the end."""

    def __init__(self, dim: int, dtype: str = "float32"):
        super().__init__(dim, 0, dtype)
        self._parts: list[Stored] = []

    def target(self, first_row: int, n_rows: int) -> Stored:
        if first_row < self.next_row:
            raise ValueError(f"rows from {first_row} were committed already")
        return self._new(n_rows)

    def commit(self, first_row: int, rows: Stored) -> None:
        self._check_commit(first_row, rows)
        self._parts.append(rows)

    def result(self) -> Stored:
        """All rows; joining the parts makes one copy of them (the window bounds only the
        rows in flight, not this result)."""
        if len(self._parts) != 1:
            if not self._parts:
                out = self._new(0)
            elif self.dtype == "int8":
                parts = [p for p in self._parts if isinstance(p, tuple)]
                codes = np.concatenate([p[0] for p in parts])
                out = QuantizedRows(codes, np.concatenate([p[1] for p in parts]))
            else:
                out = np.concatenate(self._parts)
            self._parts = [out]
        return self._parts[0]


def _check_rows(codes: ndarray, dtype: str) -> None:
    want = np.dtype(_codes.STORAGE_TYPES[dtype])
    if (
        codes.dtype != want
        or codes.ndim != 2
        or not codes.flags.writeable
        or not codes.flags.c_contiguous
    ):
        raise ValueError(
            f"the rows of a target of a {dtype} sink must be a writable C-contiguous {want} matrix"
        )


def _check_apart(parts: tuple) -> None:
    """The arrays of a target must not overlap, or one is written over another."""
    for k, a in enumerate(parts):
        if any(np.may_share_memory(a, b) for b in parts[k + 1 :]):
            raise ValueError("the arrays of a target must not overlap")


class MemorySink(GrowingRows):
    """``fse.MemorySink``: the rows of a run in memory, as ``encode`` returns them without a
    sink; ``result()`` joins them. It keeps nothing beyond the process, so it has no
    ``durable_rows``."""

    def __init__(self, dim: int, dtype: str = "float32"):
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError(f"dim must be a positive int, not {dim!r}")
        super().__init__(dim, dtype)


class GramSink(_Rows):
    """The Gram matrix of the rows instead of the rows: what fit needs (PR 5).

    Rows are gathered into blocks of REMOVAL_BLOCK_ROWS counted from row 0, each block
    product comes from the backend, and the float64 sum runs in block order, as
    fse._gram.gram_matrix does on all rows at once, so the result is the same.
    """

    def __init__(self, dim: int, backend: Any):
        super().__init__(dim, 0)
        self.backend = backend
        self.gram = np.zeros((dim, dim), dtype=np.float64)
        self._block = np.empty((_gram.REMOVAL_BLOCK_ROWS, dim), dtype=np.float32)
        self._filled = 0
        self._row = 0

    def target(self, first_row: int, n_rows: int) -> ndarray:
        if first_row < self.next_row:
            raise ValueError(f"rows from {first_row} were committed already")
        return np.empty((n_rows, self.dim), dtype=np.float32)

    def commit(self, first_row: int, rows: Stored) -> None:
        self._check_commit(first_row, rows)
        assert isinstance(rows, ndarray)
        self._row = first_row  # the row of the stream that rows[0] is
        while len(rows):
            take = min(len(rows), len(self._block) - self._filled)
            self._block[self._filled : self._filled + take] = rows[:take]
            self._filled += take
            self._row += take
            rows = rows[take:]
            if self._filled == len(self._block):
                self._add()

    def result(self) -> ndarray:
        if self._filled:
            self._add()
        return self.gram

    def _add(self):
        block = self._block[: self._filled]
        if not np.isfinite(block).all():
            begin = self._row - self._filled
            raise ValueError(f"the vectors contain NaN or infinite values in rows from {begin}")
        _gram.add_gram_block(self.gram, self.backend.gram_block(block))
        self._filled = 0
