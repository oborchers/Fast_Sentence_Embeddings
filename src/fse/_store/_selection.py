"""Bounded physical row selection without a persistent or in-memory store index."""

from __future__ import annotations

from typing import Any

import numpy as np

from fse._ids import checked_ids

# Admission covers an owned filter, sort/dedup workspace and block-local probes.
# The diagnostic records preparation workspace; caller-owned inputs are excluded.
FILTER_BYTES = 64 << 20
FILTER_ROW_BYTES = 24


def plan(
    has_doc_ids: bool, doc_ids: Any, rows: Any, max_per_doc: int | None, n_rows: int
) -> tuple[np.ndarray | None, str]:
    """Check the filters and ``max_per_doc`` of a search before any row is read; returns the
    prepared filter (None for all rows) and its kind."""
    if max_per_doc not in (None, 1):
        raise ValueError(
            f"max_per_doc must be 1 in this version, not {max_per_doc}; "
            "larger values may be supported in a later release"
        )
    if doc_ids is not None and rows is not None:
        raise ValueError("pass either doc_ids or rows, not both")
    if doc_ids is not None and not has_doc_ids:
        raise ValueError("this store has no doc_ids")
    if max_per_doc is not None and not has_doc_ids:
        raise ValueError("max_per_doc needs a store with doc_ids")
    kind = "doc_ids" if doc_ids is not None else "rows"
    if doc_ids is not None or rows is not None:
        return prepare(doc_ids if doc_ids is not None else rows, kind, n_rows), kind
    return None, kind


def prepare(value: Any, name: str, total: int) -> np.ndarray:
    if not isinstance(value, (list, np.ndarray)):
        raise TypeError(f"{name} must be a list or an int64 array")
    needed = FILTER_ROW_BYTES * len(value)
    if needed > FILTER_BYTES:
        raise ValueError(
            f"{name} needs at least {needed} filter workspace bytes; the budget is "
            f"{FILTER_BYTES} bytes; use smaller filters in separate searches"
        )
    checked_ids(value, name)
    values: np.ndarray = np.array(value, dtype=np.int64, copy=True)
    if name == "rows" and len(values) and (values.min() < 0 or values.max() >= total):
        raise ValueError(f"rows must be positions from 0 to {total - 1}")
    values.sort()
    if len(values) < 2:
        return values
    keep = np.empty(len(values), dtype=bool)
    keep[0], keep[1:] = True, values[1:] != values[:-1]
    return values[keep]


def membership(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Probe the short sorted filter, never sort or copy the store column."""
    where = np.searchsorted(values, ids)
    matched = where < len(values)
    at = np.flatnonzero(matched)
    matched[at] = values[where[at]] == ids[at]
    return np.flatnonzero(matched)


def _ranges(ids: np.ndarray, values: np.ndarray):
    begin = end = -1
    for offset in range(0, len(values), 4096):
        part = values[offset : offset + 4096]
        left, right = np.searchsorted(ids, part, "left"), np.searchsorted(ids, part, "right")
        for lo, hi in zip(left, right, strict=True):
            if lo == hi:
                continue
            if lo == end:
                end = int(hi)
            else:
                if begin >= 0:
                    yield begin, end
                begin, end = int(lo), int(hi)
    if begin >= 0:
        yield begin, end


def _sorted_positions(ids: np.ndarray, values: np.ndarray, size: int):
    pending, count = [], 0
    for lo, hi in _ranges(ids, values):
        while lo < hi:
            end = min(hi, lo + size - count)
            pending.append(np.arange(lo, end, dtype=np.int64))
            count += end - lo
            lo = end
            if count == size:
                yield np.concatenate(pending)
                pending, count = [], 0
    if pending:
        yield np.concatenate(pending)


def _doc_positions(shard: Any, ids: np.ndarray, values: np.ndarray, size: int):
    stats = shard.id_stats
    lo = np.searchsorted(values, stats.minimum, "left")
    hi = np.searchsorted(values, stats.maximum, "right")
    local = values[lo:hi]
    if not len(local):
        return
    if stats.sorted:
        yield from _sorted_positions(ids, local, size)
        return
    for start in range(0, shard.rows, size):
        selected = membership(ids[start : start + size], local)
        if len(selected):
            yield selected + start


def blocks(shards: Any, values: np.ndarray, kind: str, size: int):
    """Yield shard index and strictly ascending local positions in bounded blocks."""
    if not len(values):
        return
    for index, (shard, tensors) in enumerate(shards):
        if kind == "rows":
            lo, hi = np.searchsorted(values, [shard.first_row, shard.first_row + shard.rows])
            for start in range(int(lo), int(hi), size):
                yield index, values[start : min(start + size, int(hi))] - shard.first_row
        else:
            for selected in _doc_positions(shard, tensors["doc_ids"], values, size):
                yield index, selected
