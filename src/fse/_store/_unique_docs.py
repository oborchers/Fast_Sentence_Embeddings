"""The best row of every document in a block of scores, and the merge of those across blocks.

A search with ``max_per_doc=1`` keeps, of every document ID, the first row in the order of
``fse._reference.top_k``: the larger score first, equal scores (-0.0 equals 0.0) by the
smaller physical row. That order is total, so the best row of a document is the maximum
of a total order, which is associative: merging blocks in any order, with any number of
workers, gives the same rows. Everything here is integer work on ``_search.order_key``, so
no flushing or rounding the thread set can merge a subnormal score with zero.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from . import _search

F32 = np.float32
_NONE = np.iinfo(np.int32).max  # a local row that is no maximum of its group


def best_per_doc(
    scores: ndarray, docs: ndarray, sorted_docs: bool, k: int, floor: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """The documents of a block of ``scores`` (rows, queries) that can still be among the
    best ``k`` of a query, each with its best row: at most ``min(k, documents)`` per query,
    given the ``floor`` of every query (an order_key). ``docs`` are the labels of the rows,
    non-decreasing if ``sorted_docs``. Returns the queries, the scores (read from the
    matrix, so -0.0 keeps its bits), the *local* rows (int32, positions in the block, which
    the caller turns into store rows) and the labels, flat."""
    keep = np.flatnonzero((scores >= _search.from_key(floor)).any(axis=1)).astype(np.int32)
    if not len(keep):
        return _empty()
    labels = docs[keep]
    if not sorted_docs:
        order = np.argsort(labels, kind="stable")  # the rows of a label stay ascending
        keep, labels = keep[order], labels[order]
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    best, rep = _group_maxima(scores[keep], keep, starts)
    queries, groups = _top_groups(best, rep, floor, k)
    local = rep[groups, queries]
    return (
        queries.astype(np.int64),
        np.ascontiguousarray(scores[local, queries]),
        local,
        labels[starts][groups],
    )


def _empty() -> tuple[ndarray, ndarray, ndarray, ndarray]:
    return (
        np.empty(0, np.int64),
        np.empty(0, F32),
        np.empty(0, np.int32),
        np.empty(0, np.int64),
    )


def _group_maxima(scores: ndarray, keep: ndarray, starts: ndarray) -> tuple[ndarray, ndarray]:
    """The best order key of every (group, query) and the smallest local row that has it."""
    keys = _search.order_key(scores)
    best = np.maximum.reduceat(keys, starts, axis=0)
    counts = np.diff(np.append(starts, len(keys)))
    is_best = keys == np.repeat(best, counts, axis=0)
    del keys
    rows = np.where(is_best, keep[:, None], np.int32(_NONE))
    return best, np.minimum.reduceat(rows, starts, axis=0)


def _top_groups(best: ndarray, rep: ndarray, floor: ndarray, k: int) -> tuple[ndarray, ndarray]:
    """The (query, group) pairs at or above the floor, cut for every query to its first
    ``k`` groups in the order (key descending, local row ascending)."""
    picked_q, picked_g = [], []
    for q in range(best.shape[1]):
        groups = np.flatnonzero(best[:, q] >= floor[q])
        if len(groups) > k:  # only the groups at or above the k-th key can stay
            keys = best[groups, q]
            groups = groups[keys >= np.partition(keys, len(groups) - k)[len(groups) - k]]
            keys = best[groups, q].astype(np.int64)
            groups = groups[np.lexsort((rep[groups, q], -keys))[:k]]
        else:
            groups = groups[np.lexsort((rep[groups, q], -best[groups, q].astype(np.int64)))]
        picked_q.append(np.full(len(groups), q, np.int64))
        picked_g.append(groups)
    return np.concatenate(picked_q), np.concatenate(picked_g)


class DocumentCandidates:
    """The best row of every document so far for every query, merged down to the best
    ``k`` documents once the new ones outgrow the merged ones; ``floor`` is the order_key
    of the score of the k-th best document of every query (of -inf before it has ``k``)."""

    def __init__(self, n_queries: int, k: int):
        self.k = k
        self.floor = np.full(n_queries, _search.LOWEST, np.int32)
        self._merged: tuple[ndarray, ...] = (
            np.empty(0, np.int64),
            np.empty(0, F32),
            np.empty(0, np.int64),
            np.empty(0, np.int64),
        )
        self._pending: list[tuple[ndarray, ...]] = []
        self._size = 0

    def add(self, first_query: int, found: tuple[ndarray, ndarray, ndarray, ndarray]) -> None:
        queries, values, rows, labels = found
        if not len(queries):  # nothing at or above the floor: keep no empty arrays
            return
        self._pending.append((queries + first_query, values, rows, labels))
        self._size += len(queries)
        if self._size >= max(1, len(self._merged[0])):
            self._merge()

    def _merge(self) -> None:
        queries, values, rows, labels = (
            np.concatenate(part) for part in zip(self._merged, *self._pending, strict=True)
        )
        keys = _search.order_key(values).astype(np.int64)
        # The best row of every (query, document): the first of its group.
        order = np.lexsort((rows, -keys, labels, queries))
        queries, keys, values, rows, labels = (
            a[order] for a in (queries, keys, values, rows, labels)
        )
        first = np.ones(len(queries), bool)
        first[1:] = (queries[1:] != queries[:-1]) | (labels[1:] != labels[:-1])
        queries, keys, values, rows, labels = (
            a[first] for a in (queries, keys, values, rows, labels)
        )
        # The best k documents of every query.
        order = np.lexsort((rows, -keys, queries))
        queries, keys, values, rows, labels = (
            a[order] for a in (queries, keys, values, rows, labels)
        )
        rank = np.arange(len(queries)) - np.searchsorted(queries, queries)
        keep = rank < self.k
        self._merged = queries[keep], values[keep], rows[keep], labels[keep]
        last = rank == self.k - 1
        floor = self.floor.copy()
        floor[queries[last]] = keys[last]
        self.floor = floor  # a new array: tasks that read the old one keep it
        self._pending, self._size = [], 0

    def result(self) -> tuple[ndarray, ndarray, ndarray]:
        """The rows, scores and labels of every query, (queries, width), the best first;
        the width is ``min(k, documents in the selection)``, which every query has."""
        self._merge()
        queries, values, rows, labels = self._merged
        counts = np.bincount(queries, minlength=len(self.floor))
        if (counts != counts[0]).any():
            raise RuntimeError("a block of the search was lost")
        shape = (len(self.floor), int(counts[0]))
        return rows.reshape(shape), values.reshape(shape), labels.reshape(shape)
