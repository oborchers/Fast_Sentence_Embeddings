#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The numpy reference of the int8 prefilter (docs/architecture/numerical-contract.md, "int8 prefilter (no new arithmetic)").

A cosine search of an int8 store scores ``dot16(codes, u) * inverse norm`` per row (``u`` the
normalized query). The prefilter estimates that score from integers and bounds the error:

    a = (s * D) * inv        D = sum(codes * qcodes), exact; s the scale of the int8 query
    E = 1.003 * eps + 2**-14 eps = ||u - s * qcodes||, the quantization error of the query

A row whose best case ``a + E`` stays below the k-th largest ``a`` minus ``E`` cannot be among
the top k, so only the rows that remain are scored exactly. The bound holds for 1 <= d <= 1024,
stored codes in [-128, 127], query codes in [-127, 127], a query with ``||u||**2 <= 1.002`` and
rows whose inverse norm passes ``row_safe``; the proof is in the contract. The rule adds no
arithmetic to the contract and never changes a result bit: the survivors go through the exact
path. A pass keeps at most ``capacity`` rows per query (the native kernel's buffer) and says
``"overflow"`` when that does not suffice. Anything outside the domain (``bounds`` says ``ok``
false, ``int8_pass`` says ``unsafe``) is for the caller to send to the exact path as a whole.

Everything here is plain float64 and int64 arithmetic in the IEEE standard environment (round
to nearest even, no flush to zero, no contraction), with the grouping written out below, which
a native kernel repeats operation by operation. Constants are fixed; only the maintainer
changes them.
"""

from __future__ import annotations

import heapq
from typing import NamedTuple

import numpy as np
from numpy import ndarray

SLACK = 1.003
ABSOLUTE = 2.0**-14
QUERY_NORM2_MAX = 1.002
ROW_NORM2_MAX = 1.0 + 2.0**-20
MAX_DIM = 1024
CAPACITY_MIN = 4096
_QUERY_CODE_MAX = 127


def default_capacity(k: int) -> int:
    """How many rows a pass keeps per query before it gives up: ``max(4 * k, 4096)``."""
    return max(4 * k, CAPACITY_MIN)


class Pass(NamedTuple):
    """The result of ``int8_pass`` for one block.

    ``status`` is ``"ok"``, ``"overflow"`` or ``"unsafe"``; for ``"unsafe"`` only ``row`` (the
    first unsafe row of the block) is set, for ``"overflow"`` nothing is (see ``int8_pass``). ``heap`` holds the k largest estimates per query, descending, padded
    with ``-inf``; ``rows[j]`` and ``upper[j]`` are the rows (ascending, local to the block) kept
    for query j under the running threshold, and their upper bounds ``fl(a + E)``.
    """

    status: str
    row: int | None
    heap: ndarray | None
    rows: tuple[ndarray, ...] | None
    upper: tuple[ndarray, ...] | None


def _sequential_sum(terms: ndarray) -> ndarray:
    """The float64 sum of every row of ``terms`` in index order, one rounding per addition."""
    total = np.zeros(len(terms))
    for i in range(terms.shape[1]):
        total = total + terms[:, i]
    return total


def query_ok(norm2: ndarray) -> ndarray:
    """Whether a query is inside the proven domain: ``norm2`` is finite and at most 1.002."""
    return np.isfinite(norm2) & (norm2 <= QUERY_NORM2_MAX)


def row_safe(inverse: ndarray, squares: ndarray) -> ndarray:
    """Whether a row is inside the proven domain, from its float64 inverse norm and its integer
    sum of squares as float64: ``inverse`` is finite and not negative, and ``(inv * inv) * S``
    is at most 1 + 2**-20. A NaN fails the comparison, so the row is not safe."""
    with np.errstate(over="ignore", invalid="ignore"):
        norm2 = (inverse * inverse) * squares
        return np.isfinite(inverse) & (inverse >= 0) & (norm2 <= ROW_NORM2_MAX)


def floor_of(threshold, bound):
    """``fl(T - E)``: the value an upper bound must reach to stay."""
    return threshold - bound


def keeps(upper, floor):
    """Whether a row stays: its upper bound reaches the floor, a tie included."""
    return upper >= floor


def _check_queries(qcodes: ndarray, qscales: ndarray, units: ndarray | None = None) -> None:
    if qcodes.dtype != np.int8 or qcodes.ndim != 2 or not qcodes.size:
        raise ValueError("qcodes must be a nonempty int8 matrix")
    if qcodes.shape[1] > MAX_DIM:
        raise ValueError(f"the bound is proven for at most {MAX_DIM} dimensions")
    if qcodes.min() < -_QUERY_CODE_MAX:
        raise ValueError("the codes of a query lie in [-127, 127]")
    if qscales.dtype != np.float32 or qscales.shape != (len(qcodes),):
        raise ValueError("qscales must be one float32 scale per query")
    if not (np.isfinite(qscales).all() and (qscales >= 0).all()):
        raise ValueError("the scales of the queries must be finite and not negative")
    if units is not None and (units.dtype != np.float32 or units.shape != qcodes.shape):
        raise ValueError("units must be the float32 queries, one row per query")


def bounds(units: ndarray, qcodes: ndarray, qscales: ndarray) -> tuple[ndarray, ndarray]:
    """The slack ``E`` (float64) and whether the query is in the domain (bool), per query.

    ``units`` are the normalized float32 queries (``normalize@1``), ``qcodes`` and ``qscales``
    their int8 storage (``quantize_rows@1``). ``n2`` and ``eps`` are float64 sums in index
    order; ``s * code`` is exact, the difference rounds once per value, and
    ``E = fl(fl(1.003 * eps) + 2**-14)``.
    """
    _check_queries(qcodes, qscales, units)
    exact = units.astype(np.float64)
    value = qscales.astype(np.float64)[:, None] * qcodes.astype(np.float64)
    error = exact - value
    with np.errstate(over="ignore", invalid="ignore"):
        norm2 = _sequential_sum(exact * exact)
        eps = np.sqrt(_sequential_sum(error * error))
        slack = SLACK * eps + ABSOLUTE
    return slack, query_ok(norm2)


def estimates(
    stored: ndarray, inverse: ndarray, qcodes: ndarray, qscales: ndarray
) -> tuple[ndarray, ndarray]:
    """The estimates ``a`` (rows, queries) in float64 and which rows are safe (bool, rows).

    ``D`` and ``S`` are exact int64 sums (at most 128 * 127 * 1024 and 128**2 * 1024, below
    2**24 and equal to 2**24, so exact in float64 and int32); ``a = fl(fl(s * D) * inv)``.
    """
    _check_queries(qcodes, qscales)
    if stored.dtype != np.int8 or stored.ndim != 2 or stored.shape[1] != qcodes.shape[1]:
        raise ValueError("stored must be an int8 matrix with the dimension of the queries")
    if inverse.dtype != np.float32 or inverse.shape != (len(stored),):
        raise ValueError("inverse must be one float32 inverse norm per stored row")
    wide = stored.astype(np.int64)
    products = (wide @ qcodes.astype(np.int64).T).astype(np.float64)
    squares = (wide * wide).sum(axis=1).astype(np.float64)
    inv = inverse.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = qscales.astype(np.float64)[None, :] * products
        # + 0.0 turns -0.0 into +0.0: the sign of a zero estimate carries nothing, and equal
        # estimates then have equal bits, whatever layout a heap gives them.
        estimate = scaled * inv[:, None] + 0.0
    return estimate, row_safe(inv, squares)


def _check_pass(
    slack: ndarray, threshold: ndarray, width: int, k: int, capacity: int | None
) -> None:
    if not isinstance(k, int | np.integer) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if capacity is not None and (
        not isinstance(capacity, int | np.integer) or isinstance(capacity, bool) or capacity < 1
    ):
        raise ValueError("capacity must be a positive integer")
    for name, value in (("slack", slack), ("threshold", threshold)):
        if value.dtype != np.float64 or value.shape != (width,):
            raise ValueError(f"{name} must be one float64 value per query")
    if not (np.isfinite(slack).all() and (slack >= 0).all()):
        raise ValueError("the slack of every query must be finite and not negative")
    if np.isnan(threshold).any():
        raise ValueError("a threshold must not be NaN")


def _compact(kept: list[int], upper: list[float], floor: float) -> tuple[list[int], list[float]]:
    """The rows (and bounds) of a full list that still reach the floor, in order."""
    stay = [i for i, top in enumerate(upper) if keeps(top, floor)]
    return [kept[i] for i in stay], [upper[i] for i in stay]


def _scan(estimate: list[float], bound: float, threshold: float, k: int, capacity: int):
    """One query over a block, row by row: the k largest estimates and the rows kept under
    the running threshold ``max(T_in, k-th largest so far)`` (``-inf`` before k rows).

    A row that reaches the floor while ``capacity`` rows are kept first drops those the floor
    now cuts; if the list is still full the scan gives up and returns ``None``."""
    heap: list[float] = []
    kept: list[int] = []
    upper: list[float] = []
    for row, value in enumerate(estimate):
        if len(heap) < k:
            heapq.heappush(heap, value)
        elif value > heap[0]:
            heapq.heapreplace(heap, value)
        running = max(threshold, heap[0]) if len(heap) == k else threshold
        floor = floor_of(running, bound)
        top = value + bound
        if not keeps(top, floor):
            continue
        if len(kept) == capacity:
            kept, upper = _compact(kept, upper, floor)
            if len(kept) == capacity:
                return None
        kept.append(row)
        upper.append(top)
    return sorted(heap, reverse=True), kept, upper


def int8_pass(
    stored: ndarray,
    inverse: ndarray,
    qcodes: ndarray,
    qscales: ndarray,
    slack: ndarray,
    threshold: ndarray,
    k: int,
    capacity: int | None = None,
) -> Pass:
    """One pass over a block of stored rows for every query.

    ``slack`` is ``E`` from ``bounds`` (float64) and ``threshold`` an incoming float64 value per
    query that is at most the k-th largest estimate over all rows (``-inf`` if none). Rows are
    taken in order. A row that is not safe ends the pass with ``"unsafe"`` and its number; it
    never reaches a heap or a list. Otherwise a row's estimate enters the heap of k largest,
    and the row is kept if ``fl(a + E) >= fl(max(T_in, k-th largest so far) - E)``. As the
    running value never exceeds the final k-th largest estimate, the kept rows contain those
    ``survivors`` keeps at the end.

    At most ``capacity`` rows (default ``default_capacity(k)``) are kept per query: when a row
    reaches the floor and the list is full, the rows the floor now cuts are dropped, and if
    it is still full the pass is ``"overflow"``. An unsafe row of the block wins over an
    overflow, whatever their order; both leave only the status (and the row) set.
    """
    estimate, safe = estimates(stored, inverse, qcodes, qscales)
    _check_pass(slack, threshold, len(qcodes), k, capacity)
    if not safe.all():
        return Pass("unsafe", int(np.flatnonzero(~safe)[0]), None, None, None)
    capacity = default_capacity(k) if capacity is None else int(capacity)
    heap = np.full((len(qcodes), k), -np.inf)
    rows, upper = [], []
    for j in range(len(qcodes)):
        scanned = _scan(estimate[:, j].tolist(), float(slack[j]), float(threshold[j]), k, capacity)
        if scanned is None:
            return Pass("overflow", None, None, None, None)
        top, kept, tops = scanned
        heap[j, : len(top)] = top
        rows.append(np.array(kept, dtype=np.int64))
        upper.append(np.array(tops, dtype=np.float64))
    return Pass("ok", None, heap, tuple(rows), tuple(upper))


def merge_heaps(heaps: list[ndarray], k: int) -> ndarray:
    """The k largest estimates per query over several heaps, descending, padded with ``-inf``."""
    merged = np.concatenate(heaps, axis=1)
    return np.sort(merged, axis=1)[:, ::-1][:, :k].copy()


def kth_estimate(heap: ndarray) -> ndarray:
    """The threshold ``T_a`` per query: the k-th largest estimate, ``-inf`` if fewer than k."""
    return heap[:, -1].copy()


def survivors(rows: ndarray, upper: ndarray, threshold: float, slack: float) -> ndarray:
    """The rows of one query that stay against its final threshold ``T_a``: those with
    ``fl(a + E) >= fl(T_a - E)``. ``rows`` and ``upper`` are what ``int8_pass`` kept (every
    block's rows made global); the result depends on the store, the selection and the query,
    not on how the rows were cut into blocks or in which order they came."""
    keep = np.asarray(keeps(upper, floor_of(threshold, slack)))
    return rows[keep]
