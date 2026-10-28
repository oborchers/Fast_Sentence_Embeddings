#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Search and score over the shards of a store, block by block, in worker threads.

A task is one block of rows of one shard against one block of queries; it computes the
cosines with the backend (``cosines``: the queries normalized by normalize@1, dot16 on the
stored rows, times their inverse norms) and keeps the best ``min(k, rows)`` of every query
column. The order is the one of ``fse._reference.top_k``: the larger score first, equal
scores (-0.0 equals 0.0) by the smaller row; it is total, so merging the candidates of the
blocks in any order gives the same ids. Blocks and query blocks are sized so a task holds
at most about 4 MiB of scores, whatever the number of queries.

``exact=False`` decodes each block to float32 and multiplies with BLAS instead: faster for
many queries, but the values differ from the contract in the last bits, so ties and the
order of nearly equal scores are not reproducible.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _reference
from fse._contract import Normalize

F32 = np.float32
_TASK_SCORES = 1 << 20  # scores of one task: rows times queries
_MIN_ROWS, _MAX_ROWS = 4096, 65_536
_QUERY_BLOCK = 256


def blocks(
    shards: Sequence[tuple[Any, dict]], n_queries: int
) -> Iterator[tuple[int, int, int, int]]:
    """(shard index, first row in it, end row, first row of the store) of every block."""
    rows = max(_MIN_ROWS, min(_MAX_ROWS, _TASK_SCORES // max(1, min(n_queries, _QUERY_BLOCK))))
    for index, (shard, _) in enumerate(shards):
        for begin in range(0, shard.rows, rows):
            yield index, begin, min(shard.rows, begin + rows), shard.first_row + begin


def query_blocks(n_queries: int) -> list[slice]:
    return [slice(b, min(n_queries, b + _QUERY_BLOCK)) for b in range(0, n_queries, _QUERY_BLOCK)]


def best(scores: ndarray, first_row: int, k: int) -> tuple[ndarray, ndarray]:
    """The best ``min(k, rows)`` of every column of ``scores`` (rows, queries): their scores
    and store rows, each (queries, kept), in no particular order."""
    n, q = scores.shape
    kept = min(k, n)
    ids = np.empty((q, kept), np.int64)
    values = np.empty((q, kept), F32)
    for j in range(q):
        column = scores[:, j]
        if kept < n:
            part = np.argpartition(-column, kept - 1)[:kept]
            threshold = column[part].min()
            above = np.flatnonzero(column > threshold)
            equal = np.flatnonzero(column == threshold)[: kept - len(above)]  # the smaller rows
            rows = np.concatenate([above, equal])
        else:
            rows = np.arange(n)
        ids[j] = rows + first_row
        values[j] = column[rows]
    return values, ids


def merge(values: ndarray, ids: ndarray, k: int) -> tuple[ndarray, ndarray]:
    """The best ``k`` of candidates (queries, n): sorted, the larger score first, equal
    scores by the smaller row."""
    out_ids = np.empty((len(values), min(k, values.shape[1])), np.int64)
    out_values = np.empty(out_ids.shape, F32)
    for j in range(len(values)):
        order = np.lexsort((ids[j], -values[j].astype(np.float64)))[: out_ids.shape[1]]
        out_ids[j], out_values[j] = ids[j][order], values[j][order]
    return out_values, out_ids


def check_queries(backend: Any, queries: ndarray) -> None:
    """Check every query before any block is searched: a zero query, or one normalize@1
    cannot normalize, raises naming it, as the backend's cosines does."""
    backend.cosines(np.zeros((0, queries.shape[1]), F32), np.zeros(0, F32), queries)


def _blas_cosines(stored: ndarray, inverse: ndarray, unit: ndarray) -> ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.asarray(stored, F32) @ unit.T) * inverse[:, None]


def search(
    backend: Any,
    shards: Sequence[tuple[Any, dict]],
    queries: ndarray,
    k: int,
    *,
    workers: int,
    exact: bool,
) -> tuple[ndarray, ndarray]:
    """The ids (int64) and scores (float32) of the ``k`` best rows for every query, each
    (queries, k)."""
    check_queries(backend, queries)
    # The tolerance mode normalizes in numpy: its results are not bit-defined anyway.
    normalized = None if exact else _reference.postprocess(queries, [Normalize()])
    tasks = [(b, qb) for b in blocks(shards, len(queries)) for qb in query_blocks(len(queries))]

    def run(task):
        (index, begin, end, first_row), qb = task
        tensors = shards[index][1]
        stored, inverse = tensors["vectors"][begin:end], tensors["inv_norms"][begin:end]
        if normalized is None:
            scores = backend.cosines(stored, inverse, queries[qb])
        else:
            scores = _blas_cosines(stored, inverse, normalized[qb])
        return qb, best(scores, first_row, k)

    candidates = _Candidates(len(queries), k)
    for result in _run(run, tasks, workers, "fse-search"):
        candidates.add(*result)
    return candidates.result()


class _Candidates:
    """The candidates of every query so far, merged down to ``k`` once they grow."""

    def __init__(self, n_queries: int, k: int):
        self.k = k
        self.values = [np.empty(0, F32) for _ in range(n_queries)]
        self.ids = [np.empty(0, np.int64) for _ in range(n_queries)]

    def add(self, qb: slice, found: tuple[ndarray, ndarray]) -> None:
        values, ids = found
        for j, q in enumerate(range(qb.start, qb.stop)):
            self.values[q] = np.concatenate([self.values[q], values[j]])
            self.ids[q] = np.concatenate([self.ids[q], ids[j]])
            if len(self.values[q]) > 4 * self.k:
                top_values, top_ids = merge(self.values[q][None, :], self.ids[q][None, :], self.k)
                self.values[q], self.ids[q] = top_values[0], top_ids[0]

    def result(self) -> tuple[ndarray, ndarray]:
        top = [
            merge(v[None, :], i[None, :], self.k)
            for v, i in zip(self.values, self.ids, strict=True)
        ]
        ids = np.stack([t[1][0] for t in top]) if top else np.empty((0, self.k), np.int64)
        values = np.stack([t[0][0] for t in top]) if top else np.empty((0, self.k), F32)
        return ids, values


def score(
    backend: Any,
    shards: Sequence[tuple[Any, dict]],
    weights: ndarray,
    bias: Any,
    *,
    workers: int,
    exact: bool,
) -> ndarray:
    """The linear score of every stored row with every weight vector, (rows, vectors)."""
    biases = _biases(bias, len(weights))
    out = np.empty((sum(shard.rows for shard, _ in shards), len(weights)), F32)

    def run(task):
        (index, begin, end, first_row), qb = task
        tensors = shards[index][1]
        part = _score_block(backend, tensors, begin, end, first_row, weights[qb], biases[qb], exact)
        out[first_row : first_row + end - begin, qb] = part

    tasks = [(b, qb) for b in blocks(shards, len(weights)) for qb in query_blocks(len(weights))]
    for _ in _run(run, tasks, workers, "fse-score"):
        pass
    return out


def _score_block(backend, tensors, begin, end, first_row, weights, biases, exact) -> ndarray:
    stored = tensors["vectors"][begin:end]
    scales = tensors["scales"][begin:end] if "scales" in tensors else None
    try:
        if exact:
            return backend.scores(stored, scales, weights, biases)
        with np.errstate(over="ignore", invalid="ignore"):
            part = _reference.dequantize(stored, scales) @ weights.T + biases
        if not np.isfinite(part).all():
            raise ValueError("a score overflows float32")
        return part
    except ValueError as error:
        raise ValueError(f"rows {first_row} to {first_row + end - begin - 1}: {error}") from None


def _run(fn, tasks: list, workers: int, name: str) -> Iterator[Any]:
    """The results of ``fn`` on every task, in task order: in this thread, or in a pool."""
    if workers == 1 or len(tasks) < 2:
        yield from map(fn, tasks)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name) as pool:
        yield from pool.map(fn, tasks)


def _biases(bias: Any, n: int) -> ndarray:
    array = np.asarray(bias)
    if array.dtype.kind not in "fiu" or array.ndim > 1 or (array.ndim == 1 and array.shape != (n,)):
        raise ValueError(f"bias must be one number or {n} numbers, one per weight vector")
    with np.errstate(over="ignore"):
        biases = np.ascontiguousarray(np.broadcast_to(array.astype(F32), (n,)))
    if not np.isfinite(biases).all():
        raise ValueError("bias must be finite in float32")
    return biases
