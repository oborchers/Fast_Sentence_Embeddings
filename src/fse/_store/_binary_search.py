#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The binary search of a store (docs/architecture/store.md, "Binary search").

Stage 1 ranks every selected row by the Hamming distance of its sign bits to the bits of the
query (``fse._reference_bits.hamming``, numpy, no ``np.bitwise_count``) and keeps the best
``m = oversample * k`` rows per query, a tie going to the smaller position. The order is
total, so the candidates depend on neither blocks nor threads. Stage 2 is
``_rescore.exact_sets`` on those candidates: the result is the one of
``search(mode="exact", rows=sorted(C))`` bit for bit, whatever stage 1 found. A shard
without the companion artifact gets its bits derived from its rows for this call (one
WARNING per store).
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _log, _reference_bits

from . import _rescore, _search, _selection

DEFAULT_OVERSAMPLE = 40  # provisional: the one factor measured at 0.95 recall for both models
POSITION_BITS = 41  # a key is (distance << POSITION_BITS) | position; distances stay below 2**22
MAX_ROWS = 1 << POSITION_BITS
_MASK = np.int64(MAX_ROWS - 1)
_HELD_BYTES = 256 << 20  # what the keys of stage 1 and the candidates of stage 2 may hold
_HELD_PER_CANDIDATE = 32  # key, pending copy, merge copy, position in the candidate set
_warn_lock = threading.Lock()


def check_budget(queries: int, k: int, oversample: int, rows: int) -> int:
    """The candidates ``m`` per query, ``min(oversample * k, rows)``; raises when the
    candidates of all queries would outgrow the budget of the call."""
    m = min(oversample * k, rows)
    needed = queries * m * _HELD_PER_CANDIDATE
    if needed > _HELD_BYTES:
        raise ValueError(
            f"{queries} queries with k {k} and oversample {oversample} need about {needed} "
            f"bytes for their candidates; the budget is {_HELD_BYTES} bytes; lower oversample "
            "or ask fewer queries at once"
        )
    return m


def _bits(shard: Any, tensors: dict, selector: Any) -> ndarray:
    if shard.sign_bits is not None:
        return shard.sign_bits[selector]
    return _reference_bits.sign_bits(tensors["vectors"][selector])


def _warn_once(store: Any, shards: Any) -> int:
    missing = sum(shard.sign_bits is None for shard, _ in shards)
    if missing:
        with _warn_lock:
            first = not getattr(store, "_bits_warned", False)
            store._bits_warned = True
        if first:
            _log.emit(
                "store.bits_derived",
                "fse store: %d of %d shards have no sign bits, derived from the rows; "
                "fse.add_sign_bits(path) writes them once",
                missing,
                len(shards),
                shards=len(shards),
                missing=missing,
            )
    return missing


def _tasks(shards: Any, n_queries: int, values: Any, kind: str, count: list[int]):
    """(shard index, slice or local positions) of every block of the selection."""
    if values is None:
        for index, begin, end, _ in _search.blocks(shards, n_queries):
            count[0] += end - begin
            yield index, slice(begin, end)
        return
    size = max(1, min(65_536, (1 << 20) // min(n_queries, 256)))
    for index, local in _selection.blocks(shards, values, kind, size):
        count[0] += len(local)
        yield index, local


class _Keep:
    """The best ``m`` keys of every query so far, per block of queries."""

    def __init__(self, m: int):
        self.m = m
        self.kept: dict[int, ndarray] = {}
        self.pending: dict[int, list[ndarray]] = {}
        self.size: dict[int, int] = {}

    def add(self, found: list[tuple[int, ndarray]]) -> None:
        for start, keys in found:
            self.pending.setdefault(start, []).append(keys)
            self.size[start] = self.size.get(start, 0) + len(keys)
            if self.size[start] >= self.m:
                self._merge(start)

    def _merge(self, start: int) -> None:
        keys = np.concatenate(
            [*([self.kept[start]] if start in self.kept else []), *self.pending[start]]
        )
        if len(keys) > self.m:
            keys = np.partition(keys, self.m - 1, axis=0)[: self.m]
        self.kept[start], self.pending[start], self.size[start] = keys, [], 0

    def positions(self, n_queries: int) -> tuple[ndarray, ndarray]:
        """The candidate set as CSR: positions of every query in query order, and offsets."""
        for start in list(self.pending):
            if self.pending[start]:
                self._merge(start)
        columns: list[ndarray] = []
        for start in sorted(self.kept):
            columns.extend(self.kept[start][:, q] & _MASK for q in range(self.kept[start].shape[1]))
        offsets = np.zeros(n_queries + 1, np.int64)
        np.cumsum([len(c) for c in columns] or [0] * n_queries, out=offsets[1:])
        return (np.concatenate(columns) if columns else np.empty(0, np.int64)), offsets


def candidates(
    store: Any,
    shards: Any,
    queries: ndarray,
    m: int,
    values: Any,
    kind: str,
    *,
    workers: Any,
) -> tuple[ndarray, ndarray, int, int]:
    """Stage 1: the CSR candidates (positions, offsets) of every query, the number of rows
    selected and the number of shards whose bits were derived."""
    derived = _warn_once(store, shards)
    query_bits = _reference_bits.query_bits(queries)
    slices = _search.query_blocks(len(queries))
    keep, count = _Keep(m), [0]

    def run(task):
        index, selector = task
        shard, tensors = shards[index]
        bits = _bits(shard, tensors, selector)
        local = (
            np.arange(selector.start, selector.stop) if isinstance(selector, slice) else selector
        )
        physical = (local + shard.first_row)[:, None]
        found = []
        for qb in slices:
            distance = _reference_bits.hamming(bits, query_bits[qb])
            keys = (distance.astype(np.int64) << POSITION_BITS) | physical
            if len(keys) > m:
                keys = np.partition(keys, m - 1, axis=0)[:m]
            found.append((qb.start, keys))
        return found

    for found in _search._run(
        run, _tasks(shards, len(queries), values, kind, count), workers, "fse-binary"
    ):
        keep.add(found)
    positions, offsets = keep.positions(len(queries))
    return positions, offsets, count[0], derived


def search(
    store: Any,
    shards: Any,
    queries: ndarray,
    k: int,
    oversample: int,
    values: Any,
    kind: str,
    *,
    max_per_doc: int | None,
    workers: Any,
) -> list[tuple[ndarray, ndarray, ndarray | None]]:
    """Both stages: the best ``k`` rows (documents) of every query among its candidates,
    as (positions, scores, document IDs or None) per query, without padding."""
    total = sum(shard.rows for shard, _ in shards)
    if total > MAX_ROWS:
        raise ValueError(f"binary search holds at most {MAX_ROWS} rows, the store has {total}")
    rows = total if values is None or kind != "rows" else len(values)
    m = check_budget(len(queries), k, oversample, rows)
    _search.check_queries(store._backend, queries)
    positions, offsets, selected, derived = candidates(
        store, shards, queries, m, values, kind, workers=workers
    )
    found, _ = _rescore.exact_sets(
        store._backend,
        shards,
        queries,
        offsets,
        positions,
        k,
        one_per_doc=max_per_doc is not None,
        with_doc_ids=store._info.has_doc_ids,
        workers=workers,
    )
    _log.emit(
        "store.binary",
        "fse store: %d queries, k %d, oversample %d, %d candidates of %d selected rows",
        len(queries),
        k,
        oversample,
        len(positions),
        selected,
        queries=len(queries),
        k=k,
        oversample=oversample,
        candidates=len(positions),
        rows=selected,
        derived_shards=derived,
        workers=_search.count(workers),
        **({"max_per_doc": 1} if max_per_doc is not None else {}),
    )
    return found
