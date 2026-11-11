#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What the head pipelines share: the fold of a table, the buffers of a worker, the output
targets of a run and the step from logits to outputs (docs/architecture/runtime.md,
"Runtime")."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _log
from fse._backends import PoolInput
from fse._contract import Head, Link

from ._executor import prepare
from ._head_spec import HeadSpec
from ._pool import resolve_workers
from ._sinks import ArrayRows, GrowingRows

F32 = np.float32
# The rows of a table folded at once by one task of the fold.
_FOLD_ROWS = 1 << 14


def fold(backend: Any, pool: PoolInput, weight: ndarray, workers: Any) -> PoolInput:
    """The pool of the folded table: every word and ngram row of ``pool`` scored with every
    row of ``weight`` (``score(row, scale, a_j, 0)``, k float32 columns, no scales), in
    blocks of rows over ``workers`` threads; weights and ``oov_weight`` are kept."""
    count, _ = resolve_workers(workers)
    start = time.perf_counter()
    vectors = _fold_rows(backend, pool.vectors, pool.scales, weight, workers, "word")
    ngrams = None
    if pool.ngram_vectors is not None:
        ngrams = _fold_rows(
            backend, pool.ngram_vectors, pool.ngram_scales, weight, workers, "ngram"
        )
    rows = len(vectors) + (0 if ngrams is None else len(ngrams))
    _log.emit(
        "head.folded",
        "fse runtime: folded a head of %d outputs into %d rows of the table",
        len(weight),
        rows,
        rows=rows,
        outputs=len(weight),
        workers=count,
        backend=backend.name,
        seconds=time.perf_counter() - start,
    )
    return PoolInput(vectors, pool.weights, None, ngrams, None, pool.oov_weight, pool.aggregation)


def _fold_rows(backend, stored, scales, weight, workers, kind) -> ndarray:
    out = np.empty((len(stored), len(weight)), dtype=F32)

    def block(begin: int) -> None:
        end = min(len(stored), begin + _FOLD_ROWS)
        part = None if scales is None else scales[begin:end]
        try:
            out[begin:end] = backend.scores(stored[begin:end], part, weight, 0.0)
        except ValueError as error:
            raise ValueError(
                f"the head does not fold into the table: {kind} rows {begin} to {end - 1}: {error}"
            ) from None

    starts = range(0, len(stored), _FOLD_ROWS)
    count, shared = resolve_workers(workers)
    prepare(
        starts, block,
        workers=count, inline=len(starts) < 2, pool=shared,
    )  # fmt: skip
    out.setflags(write=False)
    return out


def buffer(local: threading.local, name: str, n_rows: int, dim: int) -> ndarray:
    """A float32 (n_rows, dim) buffer of the worker thread, reused across its tasks."""
    buffer = getattr(local, name, None)
    if buffer is None or len(buffer) < n_rows or buffer.shape[1] != dim:
        buffer = np.empty((n_rows, dim), dtype=F32)
        setattr(local, name, buffer)
    return buffer[:n_rows]


class Labels:
    """The label indices (int64) of a run: one array of a known number of rows, or parts."""

    dim = 1

    def __init__(self, rows: int | None, width: int | None = None):
        self.next_row = 0
        self._shape = () if width is None else (width,)
        self._dtype = np.int64 if width is None else np.bool_
        self._out = None if rows is None else np.empty((rows, *self._shape), dtype=self._dtype)
        self._parts: list[ndarray] = []

    def target(self, first_row: int, n_rows: int) -> ndarray:
        if self._out is None:
            return np.empty((n_rows, *self._shape), dtype=self._dtype)
        return self._out[first_row : first_row + n_rows]

    def commit(self, first_row: int, rows: ndarray) -> None:
        if first_row != self.next_row:
            raise ValueError(f"rows from {first_row} committed, expected rows from {self.next_row}")
        self.next_row += len(rows)
        if self._out is None:
            self._parts.append(rows)

    def result(self) -> ndarray:
        if self._out is not None:
            return self._out
        return (
            np.concatenate(self._parts)
            if self._parts
            else np.empty((0, *self._shape), dtype=self._dtype)
        )


def own(rows: int | None, width: int, output: str, multilabel: bool = False) -> Any:
    if output == "labels":
        return Labels(rows, width if multilabel else None)
    return GrowingRows(width) if rows is None else ArrayRows(rows, width)


def own_vectors(rows: int | None, dim: int, dtype: str) -> Any:
    return GrowingRows(dim, dtype) if rows is None else ArrayRows(rows, dim, dtype)


def result(outputs: Any, output: str) -> ndarray:
    result = outputs.result()
    return result if output == "labels" else np.asarray(result)


def write_outputs(
    backend: Any,
    local: threading.local,
    head: HeadSpec,
    logits: ndarray,
    target: ndarray,
    output: str,
    first: int,
    tag: str = "",
) -> None:
    """The ``output`` ("proba" or "labels") of the finished ``logits`` of ``head`` into
    ``target``; the label rule linear@2 decides on the probabilities, in the buffer ``tag``
    of the worker."""
    if output == "proba":
        backend.outputs(logits, "proba", head.link, target, first)
    elif head.rule is Head.LINEAR2:
        probabilities = buffer(local, f"head_proba{tag}", len(logits), head.width("proba"))
        backend.outputs(logits, "proba", head.link, probabilities, first)
        if head.link is Link.MULTILABEL:
            assert head.thresholds is not None
            backend.decisions(probabilities, head.thresholds, target)
        else:
            if target.dtype != np.intp:
                raise ValueError("the label rule linear@2 needs a platform whose int64 is intp")
            np.argmax(probabilities, axis=1, out=target)
    else:
        backend.outputs(logits, "labels", None, target, first)
