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

Memory (docs/architecture/store.md, "Binary search"): a task holds the distances of a block
of rows with a block of queries (at most ``_DISTANCE_ENTRIES`` int32, 1 MiB), the keys it
keeps (at most as many int64, 2 MiB) and the temporaries of one query column. The reference
runs on sub-blocks of at most ``_HAMMING_ENTRIES`` table entries, deriving bits on at most
``_DERIVE_ENTRIES`` values (a bool matrix of 1 MiB), and gathered bits of a selection stay
within ``_GATHER_BYTES``. The candidates of a group of queries are bounded by
``CANDIDATE_BYTES``: a larger call runs in groups of queries one after the other.
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
CANDIDATE_BYTES = 64 << 20  # what the candidates of one group of queries may hold
# Bytes per candidate of a query: while merging, the buffer of ``m + rows`` keys (at most 16)
# and the result being added (at most 8); while the candidate set is written, the buffer (at
# most 16, released block by block) and the positions (8).
CANDIDATE_ENTRY_BYTES = 24
MAX_GROUP = 256  # queries of a group, one block of queries
_DISTANCE_ENTRIES = 1 << 18  # distances of one task: rows times queries (1 MiB of int32)
_HAMMING_ENTRIES = 1 << 18  # table entries of one call of the reference (1 MiB of int32)
_DERIVE_ENTRIES = 1 << 20  # values whose signs are derived at once (a bool matrix of 1 MiB)
_GATHER_BYTES = 16 << 20  # bits gathered for a selection
_MAX_BLOCK_ROWS = 65_536
_warn_lock = threading.Lock()


def group_size(k: int, oversample: int, m: int) -> int:
    """The queries of a group, so its candidates hold at most ``CANDIDATE_BYTES``; raises when
    not even one query fits."""
    per_query = m * CANDIDATE_ENTRY_BYTES
    if per_query > CANDIDATE_BYTES:
        raise ValueError(
            f"k {k} with oversample {oversample} keeps {m} candidates per query, about "
            f"{per_query} bytes; the budget is {CANDIDATE_BYTES} bytes; lower oversample"
        )
    return max(1, min(MAX_GROUP, CANDIDATE_BYTES // max(1, per_query)))


def _derive(vectors: ndarray, selector: Any) -> ndarray:
    """The bits of the selected rows of ``vectors``, derived in chunks of ``_DERIVE_ENTRIES``
    values."""
    chunk = max(1, _DERIVE_ENTRIES // vectors.shape[1])
    if isinstance(selector, slice):
        begin, end = selector.start, selector.stop
        parts = (vectors[b : min(end, b + chunk)] for b in range(begin, end, chunk))
        count = end - begin
    else:
        parts = (vectors[selector[b : b + chunk]] for b in range(0, len(selector), chunk))
        count = len(selector)
    out = np.empty((count, -(-vectors.shape[1] // 8)), np.uint8)
    at = 0
    for part in parts:
        out[at : at + len(part)] = _reference_bits.sign_bits(part)
        at += len(part)
    return out


def _bits(shard: Any, tensors: dict, selector: Any) -> ndarray:
    if shard.sign_bits is not None:
        return shard.sign_bits[selector]
    return _derive(tensors["vectors"], selector)


def _distances(bits: ndarray, query_bits: ndarray) -> ndarray:
    """The Hamming distances, (rows, queries), from calls of the reference on sub-blocks of
    rows whose table stays within ``_HAMMING_ENTRIES``."""
    out = np.empty((len(bits), len(query_bits)), np.int32)
    step = max(1, _HAMMING_ENTRIES // (min(len(query_bits), 8) * min(bits.shape[1], 64)))
    for begin in range(0, len(bits), step):
        out[begin : begin + step] = _reference_bits.hamming(bits[begin : begin + step], query_bits)
    return out


def task(shards: Any, work: tuple, query_bits: ndarray, m: int) -> tuple[int, ndarray]:
    """Stage 1 of one block of rows and one block of queries: the first query and the best
    ``min(m, rows)`` keys ``(distance << POSITION_BITS) | position`` of every query, as a
    compact (keys, queries) array."""
    index, selector, qb = work
    shard, tensors = shards[index]
    distance = _distances(_bits(shard, tensors, selector), query_bits[qb])
    if isinstance(selector, slice):
        local = np.arange(selector.start, selector.stop, dtype=np.int64)
    else:
        local = np.asarray(selector, np.int64)
    physical = local + shard.first_row
    out = np.empty((min(m, len(physical)), distance.shape[1]), np.int64)
    for column in range(distance.shape[1]):
        keys = distance[:, column].astype(np.int64)  # one query: rows * 8 bytes
        keys <<= POSITION_BITS
        keys |= physical
        if len(keys) > m:
            keys.partition(m - 1)  # in place
            keys = keys[:m]
        out[:, column] = keys
    return qb.start, out


def _block_rows(shards: Any, n_queries: int) -> int:
    """The rows of a block: distances within ``_DISTANCE_ENTRIES``, gathered bits within
    ``_GATHER_BYTES``."""
    width = -(-shards[0][1]["vectors"].shape[1] // 8) if shards else 1
    rows = _DISTANCE_ENTRIES // min(max(1, n_queries), MAX_GROUP)
    return max(1, min(_MAX_BLOCK_ROWS, rows, _GATHER_BYTES // width))


def _tasks(shards: Any, n_queries: int, values: Any, kind: str, count: list[int]):
    """(shard index, slice or local positions, block of queries) of the selection."""
    blocks = _search.query_blocks(n_queries)
    rows = _block_rows(shards, n_queries)
    if values is None:
        selections = (
            (index, slice(begin, end))
            for index, begin, end, _ in _search.blocks(shards, 1, rows=rows)
        )
    else:
        selections = _selection.blocks(shards, values, kind, rows)
    for index, selector in selections:
        count[0] += selector.stop - selector.start if isinstance(selector, slice) else len(selector)
        for qb in blocks:
            yield index, selector, qb


class _Keep:
    """The best ``m`` keys of every query so far, per block of queries. One buffer of
    ``m + min(m, rows of a result)`` rows per block of queries receives every result after
    the kept keys and is partitioned in place back to ``m`` rows when it overflows, so
    merging allocates no array of the size of the buffer."""

    def __init__(self, m: int, result_rows: int):
        self.m, self.extra = m, min(m, result_rows)
        self.buffer: dict[int, ndarray] = {}
        self.size: dict[int, int] = {}

    def add(self, start: int, keys: ndarray) -> None:
        buffer = self.buffer.get(start)
        if buffer is None:
            shape = (self.m + self.extra, keys.shape[1])
            # Column-major: every query is contiguous along axis 0, so the in-place partition
            # needs no temporary copy of its column.
            buffer = self.buffer[start] = np.empty(shape, np.int64, order="F")
            self.size[start] = 0
        at = self.size[start]
        buffer[at : at + len(keys)] = keys
        at += len(keys)
        if at > self.m:
            buffer[:at].partition(self.m - 1, axis=0)  # in place
            at = self.m
        self.size[start] = at

    def positions(self, n_queries: int) -> tuple[ndarray, ndarray]:
        """The candidate set as CSR: positions of every query in query order, and offsets;
        each buffer is released as its positions are written."""
        width = next(iter(self.size.values()), 0)
        out = np.empty(width * n_queries, np.int64)
        for start in sorted(self.buffer):
            keys = self.buffer.pop(start)[: self.size.pop(start)]
            for column in range(keys.shape[1]):
                at = (start + column) * width
                np.bitwise_and(keys[:, column], _MASK, out=out[at : at + width])
            del keys
        return out, np.arange(n_queries + 1, dtype=np.int64) * width


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
    selected and the number of shards of the store without sign bits."""
    missing = _warn_once(store, shards)
    query_bits = _reference_bits.query_bits(queries)
    keep, count = _Keep(m, _block_rows(shards, len(queries))), [0]

    def run(work):
        return task(shards, work, query_bits, m)

    for found in _search._run(
        run, _tasks(shards, len(queries), values, kind, count), workers, "fse-binary"
    ):
        keep.add(*found)
        del found
    positions, offsets = keep.positions(len(queries))
    return positions, offsets, count[0], missing


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
    as (positions, scores, document IDs or None) per query, without padding. The queries
    run in groups whose candidates fit ``CANDIDATE_BYTES``."""
    total = sum(shard.rows for shard, _ in shards)
    if total > MAX_ROWS:
        raise ValueError(f"binary search holds at most {MAX_ROWS} rows, the store has {total}")
    rows = total
    if values is not None and (kind == "rows" or not len(values)):
        rows = len(values)
    m = min(oversample * k, rows)
    group = group_size(k, oversample, m)
    _search.check_queries(store._backend, queries)
    found: list[tuple[ndarray, ndarray, ndarray | None]] = []
    pairs = selected = missing = 0
    for first in range(0, len(queries), group):
        part = queries[first : first + group]
        positions, offsets, selected, missing = candidates(
            store, shards, part, m, values, kind, workers=workers
        )
        pairs += len(positions)
        got, _ = _rescore.exact_sets(
            store._backend,
            shards,
            part,
            offsets,
            positions,
            k,
            one_per_doc=max_per_doc is not None,
            with_doc_ids=store._info.has_doc_ids,
            workers=workers,
        )
        found.extend(got)
        del positions, got
    _log.emit(
        "store.binary",
        "fse store: %d queries, k %d, oversample %d, %d candidates of %d selected rows",
        len(queries),
        k,
        oversample,
        pairs,
        selected,
        queries=len(queries),
        k=k,
        oversample=oversample,
        candidates=pairs,
        rows=selected,
        missing_shards=missing,
        workers=_search.count(workers),
        **({"max_per_doc": 1} if max_per_doc is not None else {}),
    )
    return found
