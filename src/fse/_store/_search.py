#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Search and score over the shards of a store, block by block, in worker threads.

A task is one block of rows of one shard against one block of queries; it computes the
cosines with the backend (``cosines``: the queries normalized by normalize@1, dot16 on the
stored rows, times their inverse norms) and keeps the candidates of every query: the rows
that can still be among its best ``k``. The order is the one of ``fse._reference.top_k``:
the larger score first, equal scores (-0.0 equals 0.0) by the smaller row; it is total, so
merging the candidates of the blocks in any order gives the same ids. Blocks and query
blocks are sized so a task holds at most about 4 MiB of scores, whatever the number of
queries.

Every query has a floor, the k-th best score merged so far: a row below it cannot be
among the best, so a block passes only the rows at or above it (as FAISS keeps a heap per
query). A task reads the floors when it starts; an older floor is lower, which costs time
but never a row. Selecting per query column instead cost more than the cosines (one
argpartition per query and block: 3.1 of 3.4 s for 256 queries on 859,346 rows).

Without ``exact`` each block is decoded to float32 (at most _DECODE_BYTES of it per task)
and multiplied with BLAS instead: faster for many queries, since BLAS uses the matrix
units of the CPU, but the values differ from the contract in the last bits, so ties and
the order of nearly equal scores are not reproducible. For a few queries the kernel is as
fast or faster, since every row is read once and need not be decoded: the mode "auto"
takes it for at most AUTO_EXACT_MAX queries (859,346 rows, M1 Pro, 10 workers: equal at 8
to 12 queries; for 1 query 11 against 39 ms on int8 rows).
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _log, _reference
from fse._contract import Normalize

F32 = np.float32
_TASK_SCORES = 1 << 20  # scores of one task: rows times queries
_MIN_ROWS, _MAX_ROWS = 4096, 65_536
_QUERY_BLOCK = 256
_DECODE_BYTES = 16 << 20  # the float32 rows one BLAS task decodes
AUTO_EXACT_MAX = 8
MODES = ("auto", "exact", "blas")


def is_exact(mode: str, n_queries: int) -> bool:
    """Whether ``mode`` computes ``n_queries`` queries (or weight vectors) by the contract."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    return mode == "exact" or (mode == "auto" and n_queries <= AUTO_EXACT_MAX)


def blocks(
    shards: Sequence[tuple[Any, dict]], n_queries: int, decode_dim: int = 0, rows: int = 0
) -> Iterator[tuple[int, int, int, int]]:
    """(shard index, first row in it, end row, first row of the store) of every block; a
    block decoded to float32 rows of ``decode_dim`` values holds at most _DECODE_BYTES;
    ``rows`` fixes the rows of a block instead."""
    if not rows:
        rows = max(_MIN_ROWS, min(_MAX_ROWS, _TASK_SCORES // max(1, min(n_queries, _QUERY_BLOCK))))
    if decode_dim:
        rows = min(rows, max(1, _DECODE_BYTES // (4 * decode_dim)))
    for index, (shard, _) in enumerate(shards):
        for begin in range(0, shard.rows, rows):
            yield index, begin, min(shard.rows, begin + rows), shard.first_row + begin


def query_blocks(n_queries: int) -> list[slice]:
    return [slice(b, min(n_queries, b + _QUERY_BLOCK)) for b in range(0, n_queries, _QUERY_BLOCK)]


def order_key(values: ndarray) -> ndarray:
    """int32 keys in the order of the float32 ``values`` (finite or infinite), -0.0 equal to
    0.0: integer comparisons, whatever flushing or rounding the thread set, keep every
    subnormal score apart from zero."""
    bits = np.ascontiguousarray(values, F32).view(np.int32)
    sign = bits >> 31  # -1 for a negative value, else 0
    key = sign & np.int32(0x7FFFFFFF)
    key ^= bits  # the magnitude bits of negative values reversed
    key -= sign  # negative keys up by one, so -0.0 (key -1) meets 0.0
    return key


def from_key(keys: ndarray) -> ndarray:
    """The float32 values of order keys (0.0 for the key of -0.0)."""
    sign = keys >> 31
    return ((keys + sign) ^ (sign & np.int32(0x7FFFFFFF))).view(F32)


LOWEST = int(order_key(np.array([-np.inf], F32))[0])


def best(
    scores: ndarray, first_row: int, k: int, floor: ndarray
) -> tuple[ndarray, ndarray, ndarray]:
    """The candidates of every query column of ``scores`` (rows, queries) for its best
    ``k``, given its ``floor`` (an order_key): at most ``min(k, rows)`` per query, the
    best of the block at or above the floor. Returns their queries (column), scores and
    store rows, flat.

    A float comparison selects a superset: flushing subnormals to zero, if the thread
    set it, maps values monotonically, so it never drops a row at or above the floor.
    The order keys of the selected rows then decide exactly."""
    n, q = scores.shape
    kept = min(k, n)
    floor = floor.copy()
    mask = scores >= from_key(floor)
    over = np.flatnonzero(np.count_nonzero(mask, axis=0) > kept)
    if len(over):  # the floor keeps too many: rise to the kept-th best of the block
        sub = order_key(scores[:, over])
        floor[over] = np.maximum(floor[over], np.partition(sub, n - kept, axis=0)[n - kept])
        mask[:, over] = sub >= floor[over]
    flat = np.flatnonzero(mask)
    order = np.argsort(flat % q, kind="stable")  # by query, then by row
    rows, queries = np.divmod(flat[order], q)
    values = scores.reshape(-1)[flat[order]]
    keys = order_key(values)
    exact = keys >= floor[queries]
    rows, queries, values, keys = rows[exact], queries[exact], values[exact], keys[exact]
    return _ties(rows, queries, values, keys, floor, kept, first_row)


def _ties(rows, queries, values, keys, floor, kept, first_row):
    """Keep the rows equal to the floor only up to ``kept`` per query, the smaller rows."""
    equal = keys == floor[queries]
    above = np.bincount(queries[~equal], minlength=len(floor))
    before = np.cumsum(equal) - equal
    rank = before - before[np.searchsorted(queries, queries)] + 1
    keep = ~equal | (rank <= kept - above[queries])
    return queries[keep], values[keep], rows[keep] + first_row


class Candidates:
    """The candidates of every query so far, merged down to its best ``k`` once they
    outgrow the merged ones; ``floor`` is the order_key of the k-th best score of every
    query (of -inf before it has ``k``)."""

    def __init__(self, n_queries: int, k: int):
        self.k = k
        self.floor = np.full(n_queries, LOWEST, np.int32)
        self._merged = (np.empty(0, np.int64), np.empty(0, F32), np.empty(0, np.int64))
        self._pending: list[tuple[ndarray, ndarray, ndarray]] = []
        self._size = 0

    def add(self, first_query: int, found: tuple[ndarray, ndarray, ndarray]) -> None:
        queries, values, ids = found
        self._pending.append((queries + first_query, values, ids))
        self._size += len(queries)
        if self._size >= max(1, len(self._merged[0])):
            self._merge()

    def _merge(self) -> None:
        queries, values, ids = (
            np.concatenate(part) for part in zip(self._merged, *self._pending, strict=True)
        )
        keys = order_key(values)
        order = np.lexsort((ids, -keys.astype(np.int64), queries))
        queries, values, ids, keys = queries[order], values[order], ids[order], keys[order]
        rank = np.arange(len(queries)) - np.searchsorted(queries, queries)
        keep = rank < self.k
        self._merged = queries[keep], values[keep], ids[keep]
        last = rank == self.k - 1
        floor = self.floor.copy()
        floor[queries[last]] = keys[last]
        self.floor = floor  # a new array: tasks that read the old one keep it
        self._pending, self._size = [], 0

    def result(self, width: int) -> tuple[ndarray, ndarray]:
        """The ids and scores of every query, (queries, width), the best first; ``width``
        is ``min(k, rows of the store)``, the candidates every query has."""
        self._merge()
        queries, values, ids = self._merged
        if len(queries) != len(self.floor) * width:
            raise RuntimeError("a block of the search was lost")
        shape = (len(self.floor), width)
        return ids.reshape(shape), values.reshape(shape)


def check_queries(backend: Any, queries: ndarray) -> None:
    """Check every query before any block is searched: a zero query, or one normalize@1
    cannot normalize, raises naming it, as the backend's cosines does."""
    backend.cosines(np.zeros((0, queries.shape[1]), F32), np.zeros(0, F32), queries)


def _blas_cosines(stored: ndarray, inverse: ndarray, unit: ndarray) -> ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (np.asarray(stored, F32) @ unit.T) * inverse[:, None]
    if not np.isfinite(scores).all():
        query, row = np.argwhere(~np.isfinite(scores.T))[0]  # (query, row), as the kernel
        raise ValueError(f"query {query}: the cosine of row {row} is not finite")
    return scores


def _block_cosines(backend, tensors, block, queries, normalized) -> ndarray:
    """The cosines of one block of rows with one block of queries; an error names where
    they are in the store and among the queries."""
    (_, begin, end, first_row), qb = block
    stored, inverse = tensors["vectors"][begin:end], tensors["inv_norms"][begin:end]
    try:
        if normalized is None:
            return backend.cosines(stored, inverse, queries[qb])
        return _blas_cosines(stored, inverse, normalized[qb])
    except ValueError as error:
        raise ValueError(
            f"rows {first_row} to {first_row + end - begin - 1} of the store, queries "
            f"{qb.start} to {qb.stop - 1}: {error}"
        ) from None


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
    _log.emit(
        "store.search",
        "fse store: %d queries, k %d, %s, over %d rows on %d workers",
        len(queries),
        k,
        "exact" if exact else "blas",
        sum(shard.rows for shard, _ in shards),
        workers,
        queries=len(queries),
        k=k,
        mode="exact" if exact else "blas",
        rows=sum(shard.rows for shard, _ in shards),
        workers=workers,
    )
    # The tolerance mode normalizes in numpy: its results are not bit-defined anyway.
    normalized = None if exact else _reference.postprocess(queries, [Normalize()])
    decode = 0 if exact else queries.shape[1]
    tasks = _tasks(shards, len(queries), decode)
    candidates = Candidates(len(queries), k)

    def run(task):
        (index, _, _, first_row), qb = task
        floor = candidates.floor[qb]
        scores = _block_cosines(backend, shards[index][1], task, queries, normalized)
        return qb.start, best(scores, first_row, k, floor)

    for result in _run(run, tasks, workers, "fse-search"):
        candidates.add(*result)
    return candidates.result(min(k, sum(shard.rows for shard, _ in shards)))


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
    biases = _biases(backend, bias, len(weights))
    out = np.empty((sum(shard.rows for shard, _ in shards), len(weights)), F32)

    def run(task):
        (index, begin, end, first_row), qb = task
        tensors = shards[index][1]
        part = _score_block(backend, tensors, begin, end, first_row, weights[qb], biases[qb], exact)
        out[first_row : first_row + end - begin, qb] = part

    decode = 0 if exact else weights.shape[1]
    tasks = _tasks(shards, len(weights), decode)
    for _ in _run(run, tasks, workers, "fse-score"):
        pass
    return out


def score_blocks(
    backend: Any,
    shards: Sequence[tuple[Any, dict]],
    weights: ndarray,
    bias: Any,
    *,
    workers: int,
    then: Any = None,
) -> Iterator[tuple[int, Any]]:
    """The linear scores of the stored rows with every weight vector by the contract, block
    by block in row order: ``(first row of the store, block)`` with the (rows, vectors)
    scores of the block, or what ``then(first_row, scores)`` makes of them in the worker
    (a head's outputs). A block holds at most _TASK_SCORES values of scores and as many of
    what ``then`` makes, all vectors at once (unlike search, whose query blocks cap the
    width), so a wide head takes fewer rows per block; at most two blocks per worker are
    in flight or waiting, so a consumer that keeps nothing holds no more than that,
    whatever the store's size."""
    biases = _biases(backend, bias, len(weights))
    rows = max(1, min(_MAX_ROWS, _TASK_SCORES // max(1, len(weights))))

    def run(block):
        index, begin, end, first_row = block
        part = _score_block(backend, shards[index][1], begin, end, first_row, weights, biases, True)
        return first_row, part if then is None else then(first_row, part)

    yield from _run(run, blocks(shards, len(weights), rows=rows), workers, "fse-score")


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


def _tasks(shards: Sequence[tuple[Any, dict]], n_queries: int, decode: int) -> Iterator[tuple]:
    """Every (block of rows, block of queries), made as the search goes."""
    query_slices = query_blocks(n_queries)
    for block in blocks(shards, n_queries, decode):
        for qb in query_slices:
            yield block, qb


def _run(fn, tasks: Iterable[Any], workers: int, name: str) -> Iterator[Any]:
    """The results of ``fn`` on every task, in task order: in this thread, or in a pool
    with at most two tasks per worker in flight, so finished results stay bounded and a
    task starts with the floors merged so far."""
    tasks = iter(tasks)
    head = list(itertools.islice(tasks, 2))
    if workers == 1 or len(head) < 2:
        yield from map(fn, itertools.chain(head, tasks))
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name) as pool:
        pending: deque = deque()
        for task in itertools.chain(head, tasks):
            pending.append(pool.submit(fn, task))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _biases(backend: Any, bias: Any, n: int) -> ndarray:
    array = np.asarray(bias)
    if array.dtype.kind not in "fiu" or array.ndim > 1 or (array.ndim == 1 and array.shape != (n,)):
        raise ValueError(f"bias must be one number or {n} numbers, one per weight vector")
    biases = np.ascontiguousarray(np.broadcast_to(backend.float32(array), (n,)))
    if not np.isfinite(biases).all():
        raise ValueError("bias must be finite in float32")
    return biases
