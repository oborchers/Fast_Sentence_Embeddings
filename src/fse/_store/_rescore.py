#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Exact rescoring of given candidate positions (docs/architecture/store.md, "Rescoring").

For every query the best ``min(k, candidates)`` rows of its own set of store positions, by
the arithmetic of ``search(mode="exact")``: the rows of the set are gathered in their stored
dtype, scored with ``backend.cosines`` (the same kernel, one query column at a time, so the
bits of a pair do not depend on the other rows or queries) and ordered by
``_search.order_key`` and the position, the order of ``fse._reference.top_k``. With
``one_per_doc`` the best row of every document is kept and the best ``k`` documents are
returned (``_unique_docs.best_per_doc``).

A set larger than one task may hold (``_TASK_BYTES``) is cut into parts of ascending
positions. Every part keeps its best ``k`` rows (documents); the winners of all parts are
reduced the same way. That is exact: the order is total, and a row (document) outside the
best ``k`` of its part has ``k`` better ones in the part, so it cannot be among the best
``k`` of the set.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _log
from fse._ids import checked_ids

from . import _search, _selection, _take, _unique_docs

F32 = np.float32
_TASK_BYTES = 16 << 20  # the gathered rows of one part of a query's set
_HINT_LIST = (
    "candidates must be a flat list of integers or an int64 array of one or two dimensions; "
    "use np.asarray(candidates, dtype=np.int64) for a list of lists"
)


def _int64_array(value: ndarray) -> ndarray:
    if isinstance(value, np.ma.MaskedArray) or value.dtype != np.dtype("int64"):
        kind = "a masked array" if isinstance(value, np.ma.MaskedArray) else "an array"
        hint = ""
        if value.dtype == np.dtype("uint64"):
            hint = (
                "; a uint64 array may hold keys above the int64 range, so convert it with "
                "an explicit range check first"
            )
        raise TypeError(
            f"candidates must be an int64 array, not {kind} of dtype {value.dtype}{hint}"
        )
    return np.asarray(value)


def _flat_list(value: list) -> ndarray:
    if value and isinstance(value[0], (list, tuple, np.ndarray)):
        raise TypeError(_HINT_LIST)
    checked_ids(value, "candidates")
    return np.array(value, dtype=np.int64)


def candidates(value: Any, queries: int, single: bool) -> ndarray:
    """The int64 candidates, as an array of shape (m,) for one query vector or (queries, m)
    for a matrix of queries: used as it is when it is an int64 array (no copy), converted
    when it is a flat list. Anything else raises before a row is read."""
    if isinstance(value, np.ndarray):
        array = _int64_array(value)
    elif isinstance(value, list):
        array = _flat_list(value)
    else:
        raise TypeError(f"{_HINT_LIST}, not {type(value).__name__}")
    ok = array.ndim == 1 if single else array.ndim == 2 and array.shape[0] == queries
    if not ok:
        want = "(m,) for one query vector" if single else f"({queries}, m) for {queries} queries"
        raise ValueError(f"candidates must have shape {want}, not shape {array.shape}")
    return array


def check_values(array: ndarray, total: int) -> None:
    """Raise for a value below -1 or at or above ``total``, naming the first one."""
    if not array.size or (array.min() >= -1 and array.max() < total):
        return
    at = int(np.argmax((array < -1) | (array >= total)))
    index = ", ".join(str(i) for i in np.unravel_index(at, array.shape))
    value = int(array.reshape(-1)[at])
    hint = (
        "; refresh() the store to see rows published since it was opened" if value >= total else ""
    )
    raise ValueError(
        f"candidates[{index}] is {value}; candidates must be positions from 0 to {total - 1}, "
        f"or -1 for none{hint}"
    )


def csr(array: ndarray) -> tuple[ndarray, ndarray]:
    """The positions of ``array`` ((m,) or (queries, m)) without -1, and the offsets of every
    query in them."""
    keep = array != -1
    rows = array[None, :] if array.ndim == 1 else array
    mask = keep[None, :] if keep.ndim == 1 else keep
    offsets = np.zeros(len(rows) + 1, np.int64)
    np.cumsum(mask.sum(axis=1), out=offsets[1:])
    return array[keep], offsets


def _read_stored(tensors: dict[str, ndarray], local: ndarray) -> ndarray:
    return tensors["vectors"][local]


def _read_inverse(tensors: dict[str, ndarray], local: ndarray) -> ndarray:
    return tensors["inv_norms"][local]


def _best(
    positions: ndarray, scores: ndarray, labels: ndarray | None, k: int
) -> tuple[ndarray, ndarray, ndarray | None]:
    """The best ``k`` rows (documents) of ``positions`` (ascending), best first."""
    if labels is None:
        keys = _search.order_key(scores).astype(np.int64)
        order = np.lexsort((positions, -keys))[:k]
        return positions[order], scores[order], None
    floor = np.full(1, _search.LOWEST, np.int32)
    _, values, local, found = _unique_docs.best_per_doc(scores[:, None], labels, False, k, floor)
    return positions[local], values, found


def _merge(
    left: tuple[ndarray, ndarray, ndarray | None],
    right: tuple[ndarray, ndarray, ndarray | None],
    k: int,
) -> tuple[ndarray, ndarray, ndarray | None]:
    positions = np.concatenate((left[0], right[0]))
    order = np.argsort(positions, kind="stable")
    labels = None if left[2] is None or right[2] is None else np.concatenate((left[2], right[2]))
    return _best(
        positions[order],
        np.concatenate((left[1], right[1]))[order],
        None if labels is None else labels[order],
        k,
    )


class _Run:
    """What the tasks of one call share: the snapshot, the queries and the sizes."""

    def __init__(self, backend, shards, queries, k, one_per_doc, with_doc_ids, offsets, positions):
        self.backend, self.shards, self.queries, self.k = backend, shards, queries, k
        self.one_per_doc, self.with_doc_ids = one_per_doc, with_doc_ids
        self.offsets, self.positions = offsets, positions
        self.total = sum(shard.rows for shard, _ in shards)
        self.dim = queries.shape[1]
        itemsize = shards[0][1]["vectors"].dtype.itemsize if shards else 1
        self.row_bytes = self.dim * itemsize + 4 + 8 + 8
        self.part_rows = max(1, _TASK_BYTES // self.row_bytes)

    def rows(self, wanted: ndarray, read: Any, tail: tuple[int, ...], dtype: Any) -> ndarray:
        return _take.gather(
            self.shards,
            wanted,
            self.total,
            read,
            tail=tail,
            dtype=dtype,
            row_bytes=self.row_bytes,
        )

    def part(self, q: int, wanted: ndarray) -> tuple[ndarray, ndarray, ndarray | None]:
        stored = self.rows(wanted, _read_stored, (self.dim,), self.shards[0][1]["vectors"].dtype)
        inverse = self.rows(wanted, _read_inverse, (), F32)
        try:
            scores = self.backend.cosines(stored, inverse, self.queries[q : q + 1])
        except ValueError as error:
            text = re.sub(r"\brow (\d+)\b", lambda m: f"row {wanted[int(m[1])]}", str(error))
            raise ValueError(f"query {q}: {text}") from None
        labels = self.rows(wanted, _take.read_doc_ids, (), np.int64) if self.one_per_doc else None
        return _best(wanted, np.ascontiguousarray(scores[:, 0]), labels, self.k)

    def query(self, q: int) -> tuple[tuple[ndarray, ndarray, ndarray | None], int]:
        """The best rows of query ``q`` and the number of rows gathered for it."""
        wanted = np.unique(self.positions[self.offsets[q] : self.offsets[q + 1]])
        won = None
        for begin in range(0, len(wanted), self.part_rows):
            found = self.part(q, wanted[begin : begin + self.part_rows])
            won = found if won is None else _merge(won, found, self.k)
        if won is None:
            empty = np.empty(0, np.int64)
            return (empty, np.empty(0, F32), empty if self.with_doc_ids else None), 0
        if self.with_doc_ids and won[2] is None:
            won = (won[0], won[1], self.rows(won[0], _take.read_doc_ids, (), np.int64))
        return won, len(wanted)


def exact_sets(
    backend: Any,
    shards: Any,
    queries: ndarray,
    offsets: ndarray,
    positions: ndarray,
    k: int,
    *,
    one_per_doc: bool = False,
    with_doc_ids: bool = False,
    workers: int,
) -> tuple[list[tuple[ndarray, ndarray, ndarray | None]], int]:
    """The best ``k`` rows (documents) of every query among its own positions
    ``positions[offsets[q]:offsets[q + 1]]`` (already checked against the rows of
    ``shards``; duplicates are dropped; ``with_doc_ids`` adds the document IDs of the winners), as (positions, scores, document IDs or None) per
    query without padding, best first, and the number of rows gathered. The caller holds
    the store's active-use guard."""
    _search.check_queries(backend, queries)
    run = _Run(
        backend, shards, queries, k, one_per_doc, with_doc_ids or one_per_doc, offsets, positions
    )
    found = list(_search._run(run.query, range(len(queries)), workers, "fse-rescore"))
    return [item[0] for item in found], sum(item[1] for item in found)


def pad(
    found: list[tuple[ndarray, ndarray, ndarray | None]], with_labels: bool
) -> tuple[ndarray, ndarray, ndarray | None]:
    """The rectangular result of the queries: ids (-1), scores (-inf) and document IDs (-1)
    of shape (queries, widest hit list), filled up on the right."""
    width = max((len(item[0]) for item in found), default=0)
    ids = np.full((len(found), width), -1, np.int64)
    scores = np.full((len(found), width), -np.inf, F32)
    labels = np.full((len(found), width), -1, np.int64) if with_labels else None
    for row, (i, s, d) in enumerate(found):
        ids[row, : len(i)], scores[row, : len(s)] = i, s
        if labels is not None and d is not None:
            labels[row, : len(d)] = d
    return ids, scores, labels


def check_grouping(has_doc_ids: bool, max_per_doc: int | None) -> None:
    """The ``max_per_doc`` rules of a search, before any row is read."""
    _selection.plan(has_doc_ids, None, None, max_per_doc, 0)


def log(queries: int, k: int, pairs: int, read_rows: int, workers: int, grouped: bool) -> None:
    fields: dict[str, Any] = {"max_per_doc": 1} if grouped else {}
    _log.emit(
        "store.rescore",
        "fse store: %d queries rescored over %d candidate pairs",
        queries,
        pairs,
        queries=queries,
        k=k,
        pairs=pairs,
        read_rows=read_rows,
        workers=workers,
        **fields,
    )
