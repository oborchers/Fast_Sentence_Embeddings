#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The int8 prefilter of the exact search (docs/architecture/store.md, "Search").

For an int8 store and a few queries, a native pass over every block estimates the cosine of
every row from integers and bounds the error (docs/architecture/numerical-contract.md, "int8
prefilter (no new arithmetic)"). Only the rows whose upper bound still reaches the k-th best
estimate minus the slack can be among the best ``k``; they go to ``_rescore.exact_sets``, the
exact path of ``rescore``. The result is the one of ``search(mode="exact")`` bit for bit: the
survivors hold every row that can be among the best ``k``, ties included, and the order is
total.

``search`` returns ``None`` whenever it does not apply, and the caller takes the exact path
for the whole call: the call is outside the domain (see ``applies``), a query is outside the
proven range (``query_range``), a stored row is not safe (``unsafe_row``: NaN, infinite or
negative inverse norm, a norm that disagrees with the codes), a block keeps more rows than its
buffer holds (``overflow``), or the survivors outgrow the budget of the call (``budget``). The
exact path then gives the bits and the errors of a call without the prefilter.

The survivors of a query are the rows with ``fl(a + E) >= fl(T_a - E)`` for the final k-th
estimate ``T_a``. A running threshold of the blocks so far is never above ``T_a``, so a block
that starts late keeps fewer rows but never one too few. That makes the set of survivors
depend on the store, the selection and the query only, not on the order of the threads.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _log, _reference_prefilter
from fse._contract import Normalize

from . import _filtered, _rescore, _search, _selection

F32 = np.float32
MAX_QUERIES = 8
MAX_K = 100
MIN_DIM, MAX_DIM = 256, 1024
MIN_ROWS = 65_536
_BUDGET_MIN = 65_536  # survivors per query the main thread holds: max(64 * k, this)
_BLOCK_BYTES = 16 << 20  # the rows of one block of a selection


def applies(
    backend: Any,
    shards: Any,
    queries: ndarray,
    k: int,
    selected: int,
    exact: bool,
    grouped: bool,
    kind: str,
    filtered: bool,
) -> bool:
    """Whether a call is inside the domain of the prefilter: the exact path is chosen, no
    ``max_per_doc``, no ``doc_ids`` filter (its row count is not known before the rows are
    read), 1 to ``MAX_QUERIES`` queries, ``k`` up to ``MAX_K``, an int8 store of ``MIN_DIM``
    to ``MAX_DIM`` values per row, at least ``MIN_ROWS`` selected rows, and a backend that
    enables a variant."""
    if not exact or grouped or (filtered and kind != "rows"):
        return False
    if not 1 <= len(queries) <= MAX_QUERIES or k > MAX_K or selected < MIN_ROWS:
        return False
    if not shards or not MIN_DIM <= queries.shape[1] <= MAX_DIM:
        return False
    if any(tensors["vectors"].dtype != np.int8 for _, tensors in shards):
        return False
    return backend.prefilter_variant() is not None


class _Queries:
    """The queries as the exact path takes them and their int8 form with the slack."""

    def __init__(self, backend: Any, queries: ndarray):
        units = np.array(queries, dtype=F32, copy=True)
        backend.postprocess_pooled(units, (Normalize(),), 0)
        self.codes = np.empty(units.shape, np.int8)
        self.scales = np.empty(len(units), F32)
        backend.store(units, self.codes, self.scales, 0)
        self.slack, ok = backend.int8_bounds(units, self.codes, self.scales)
        self.ok = bool(ok.all())


class _Survivors:
    """What the main thread keeps of the passes: the running k-th estimate (a float64 value
    per query, never above the final one), the heap behind it and the rows kept so far
    (global positions) with their upper bounds."""

    def __init__(self, width: int, k: int, slack: ndarray):
        self.k, self.slack, self.read = k, slack, 0
        self.budget = max(64 * k, _BUDGET_MIN)
        self.heap = np.full((width, k), -np.inf)
        self.threshold = np.full(width, -np.inf)
        self.rows: list[list[ndarray]] = [[] for _ in range(width)]
        self.upper: list[list[ndarray]] = [[] for _ in range(width)]
        self.count = [0] * width

    def add(self, got: Any, base: Any, block_rows: int) -> str | None:
        """Take the result of one block; the reason for a fallback, or ``None``."""
        if got.status != "ok":
            return "unsafe_row" if got.status == "unsafe" else "overflow"
        self.read += block_rows
        self.heap = _reference_prefilter.merge_heaps([self.heap, got.heap], self.k)
        self.threshold = _reference_prefilter.kth_estimate(self.heap)  # a new array
        for j, (rows, upper) in enumerate(zip(got.rows, got.upper, strict=True)):
            self.rows[j].append(base[rows] if isinstance(base, ndarray) else rows + base)
            self.upper[j].append(upper)
            self.count[j] += len(rows)
            if self.count[j] > self.budget and self._compact(j) > self.budget:
                return "budget"
        return None

    def _compact(self, j: int) -> int:
        """Drop the rows of query ``j`` that the running threshold already cuts; the count
        left. Safe: the running threshold is never above the final one."""
        kept = self.final(j)
        self.rows[j], self.upper[j] = [kept[0]], [kept[1]]
        self.count[j] = len(kept[0])
        return self.count[j]

    def final(self, j: int) -> tuple[ndarray, ndarray]:
        rows = np.concatenate(self.rows[j] or [np.empty(0, np.int64)])
        upper = np.concatenate(self.upper[j] or [np.empty(0, np.float64)])
        keep = np.asarray(
            _reference_prefilter.keeps(
                upper, _reference_prefilter.floor_of(self.threshold[j], self.slack[j])
            )
        )
        return rows[keep], upper[keep]

    def csr(self) -> tuple[ndarray, ndarray]:
        """The survivors against the final threshold: the offsets of every query and the
        positions (ascending within a query)."""
        found = [self.final(j)[0] for j in range(len(self.rows))]
        offsets = np.zeros(len(found) + 1, np.int64)
        np.cumsum([len(rows) for rows in found], out=offsets[1:])
        return offsets, np.concatenate(found).astype(np.int64, copy=False)

    def release(self) -> None:
        self.rows, self.upper, self.heap = [], [], np.empty((0, 0))


def _plain_tasks(shards: Any, width: int):
    """(rows, inverse norms, first position of the block) of every block of the store."""
    for index, begin, end, first in _search.blocks(shards, width):
        tensors = shards[index][1]
        yield tensors["vectors"][begin:end], tensors["inv_norms"][begin:end], first


def _selected_tasks(shards: Any, values: ndarray, kind: str, dim: int):
    """(rows, inverse norms, positions of the block) of the blocks of a ``rows=`` selection."""
    size = max(1, min(_search._MAX_ROWS, _BLOCK_BYTES // dim))
    for index, positions in _selection.blocks(shards, values, kind, size):
        shard, tensors = shards[index]
        stored = _filtered._part(tensors["vectors"], positions)
        inverse = _filtered._part(tensors["inv_norms"], positions)
        yield stored, inverse, positions + shard.first_row


def _passes(backend, queries, k, state, abort, tasks, workers):
    def run(task):
        stored, inverse, base = task
        if abort.is_set():
            return None
        threshold = state.threshold  # read once: a task keeps the one it started with
        got = backend.int8_pass(
            np.ascontiguousarray(stored),
            np.ascontiguousarray(inverse),
            queries.codes,
            queries.scales,
            queries.slack,
            threshold,
            k,
        )
        return got, base, len(stored)

    return _search._run(run, tasks, workers, "fse-search")


def _collect(backend, queries, k, state, tasks, workers) -> str | None:
    abort = threading.Event()
    try:
        with contextlib.closing(_passes(backend, queries, k, state, abort, tasks, workers)) as it:
            for result in it:
                reason = None if result is None else state.add(*result)
                if reason is not None:
                    abort.set()
                    return reason
    finally:
        abort.set()
    return None


def _log_entry(width, k, read, pairs, fallback, workers) -> None:
    _log.emit(
        "store.prefilter",
        "fse store: %d queries, prefilter %s, %d survivors of %d rows",
        width,
        fallback,
        pairs,
        read,
        queries=width,
        k=k,
        rows=read,
        pairs=pairs,
        fallback=fallback,
        workers=workers,
    )


def _log_search(width, k, rows, workers) -> None:
    _log.emit(
        "store.search",
        "fse store: %d queries, k %d, exact, over %d rows on %d workers",
        width,
        k,
        rows,
        workers,
        queries=width,
        k=k,
        mode="exact",
        rows=rows,
        workers=workers,
    )


def _rescored(backend, shards, matrix, k, state, selected, workers):
    offsets, positions = state.csr()
    pairs = int(offsets[-1])
    state.release()
    found, _ = _rescore.exact_sets(backend, shards, matrix, offsets, positions, k, workers=workers)
    width = min(k, selected)
    if len(found) != len(matrix) or any(len(item[0]) != width for item in found):
        raise RuntimeError("a row of the prefilter search was lost")
    ids = np.stack([item[0] for item in found])
    scores = np.stack([item[1] for item in found])
    return (ids, scores), pairs


def search(
    backend: Any,
    shards: Any,
    queries: ndarray,
    k: int,
    values: ndarray | None,
    kind: str,
    *,
    workers: int,
    exact: bool,
    grouped: bool,
) -> tuple[ndarray, ndarray] | None:
    """The ids (int64) and scores (float32) of the ``k`` best rows of every query, each
    (queries, k), as ``_search.search`` or ``_filtered.search`` with ``exact``; ``None`` when
    the prefilter does not apply or gives up, and nothing but the log has happened (the
    caller takes the exact path). The caller holds the store's active-use guard."""
    selected = sum(shard.rows for shard, _ in shards) if values is None else len(values)
    if not applies(backend, shards, queries, k, selected, exact, grouped, kind, values is not None):
        return None
    _search.check_queries(backend, queries)
    prepared = _Queries(backend, queries)
    state = _Survivors(len(queries), k, prepared.slack)
    fallback = "query_range"
    if prepared.ok:
        dim = queries.shape[1]
        tasks = (
            _plain_tasks(shards, len(queries))
            if values is None
            else _selected_tasks(shards, values, kind, dim)
        )
        fallback = _collect(backend, prepared, k, state, tasks, workers) or "none"
    result, pairs = None, 0
    if fallback == "none":
        result, pairs = _rescored(backend, shards, queries, k, state, selected, workers)
    state.release()
    _log_entry(len(queries), k, state.read, pairs, fallback, workers)
    if result is not None:
        _log_search(len(queries), k, selected, workers)
    return result
