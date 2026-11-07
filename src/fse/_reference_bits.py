#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The numpy reference of the sign bits (docs/architecture/numerical-contract.md, "Sign bits (``sign_bits@1``)").

A row of a store gets one bit per dimension: 1 if the stored value is greater than 0, else 0.
The bits are packed eight to a byte, and a search ranks rows by the Hamming distance of their
bits to the bits of the query (the symmetric rule) or by a signed integer sum of the query's
int4 codes over the set bits (the asymmetric variant). Both rules are ranks of integers.

The sign comes from the integer bit pattern of the stored value, never from a floating point
comparison: ``float32`` is read as ``int32`` and ``float16`` as ``int16``, and the bit is 1 if
that signed integer is greater than 0. For a finite value this is "the value is greater than
0" (the sign bit is 0 and the value is not +0.0), whatever the flush-to-zero and
denormals-are-zero state of the thread: a positive subnormal gives 1 where a float comparison
under DAZ would give 0. An int8 code is compared as it is. The rule has no floating point
operation, so it has no ``transform_version``.

Everything works in bounded blocks of rows, queries and bytes, so no step holds an array of
``rows * queries * bytes`` or a bool matrix of more than ``BLOCK_BYTES`` entries.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

BLOCK_BYTES = 1 << 22
INT4_MAX = 7
_VIEWS = {np.dtype(np.float32): np.int32, np.dtype(np.float16): np.int16, np.dtype(np.int8): None}
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _check_matrix(values: ndarray, name: str) -> None:
    if not isinstance(values, ndarray) or values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{name} must be a two dimensional array with at least one column")


def _check_packed(bits: ndarray, name: str) -> None:
    if not isinstance(bits, ndarray) or bits.dtype != np.uint8 or bits.ndim != 2:
        raise ValueError(f"{name} must be a uint8 matrix of packed bits")


def _positive(values: ndarray) -> ndarray:
    """The bool matrix ``values > 0`` from the integer bit pattern of ``values``."""
    view = _VIEWS[values.dtype]
    patterns = values if view is None else np.ascontiguousarray(values).view(view)
    return patterns > 0


def _pack(flags: ndarray) -> ndarray:
    """Eight flags to a byte, dimension 0 in the highest bit, the padding bits 0."""
    return np.packbits(flags, axis=1, bitorder="big")


def _round(values: ndarray) -> ndarray:
    """Round half to even."""
    return np.rint(values)


def _row_block(width: int) -> int:
    return max(1, BLOCK_BYTES // max(1, width))


def sign_bits(stored: ndarray) -> ndarray:
    """The packed bits of the rows of ``stored`` (float32, float16 or int8), shape
    ``(rows, ceil(d / 8))``, uint8. Dimension 0 is the highest bit of byte 0 and the padding
    bits of the last byte are 0."""
    _check_matrix(stored, "stored")
    if stored.dtype not in _VIEWS:
        raise ValueError("stored must be float32, float16 or int8")
    step = _row_block(stored.shape[1])
    out = np.empty((len(stored), -(-stored.shape[1] // 8)), dtype=np.uint8)
    for start in range(0, len(stored), step):
        out[start : start + step] = _pack(_positive(stored[start : start + step]))
    return out


def query_bits(queries: ndarray) -> ndarray:
    """The packed bits of float32 queries, by the rule of ``sign_bits`` on the float32 pattern."""
    _check_matrix(queries, "queries")
    if queries.dtype != np.float32:
        raise ValueError("queries must be float32")
    return sign_bits(queries)


def hamming(bits: ndarray, query: ndarray) -> ndarray:
    """The Hamming distances, shape ``(rows, queries)``, int32: the popcount of
    ``bits[r] XOR query[q]``, from a table of 256 entries and summed in int32, in blocks of
    rows, queries and bytes. Needs no ``np.bitwise_count`` (NumPy 1.26 has none)."""
    _check_packed(bits, "bits")
    _check_packed(query, "query")
    if bits.shape[1] != query.shape[1]:
        raise ValueError("bits and query must have the same number of bytes per row")
    width = bits.shape[1]
    out = np.zeros((len(bits), len(query)), dtype=np.int32)
    step_bytes = min(width, 64)
    step_queries = 8
    step_rows = _row_block(step_queries * step_bytes)
    for r in range(0, len(bits), step_rows):
        for q in range(0, len(query), step_queries):
            for b in range(0, width, step_bytes):
                rows = bits[r : r + step_rows, None, b : b + step_bytes]
                cols = query[None, q : q + step_queries, b : b + step_bytes]
                out[r : r + step_rows, q : q + step_queries] += np.add.reduce(
                    _POPCOUNT[rows ^ cols], axis=2, dtype=np.int32
                )
    return out


def physical(rows: ndarray, local: ndarray) -> ndarray:
    """The global positions of local row numbers: ``rows[local]``, never ``first + local``,
    because a selection is not contiguous."""
    return rows[local]


def _order(rank: ndarray, rows: ndarray, m: int) -> ndarray:
    """The first ``m`` local rows by ascending ``rank`` (one column), a tie going to the
    smaller position. The order is total."""
    local = np.lexsort((rows, rank))[:m]
    return physical(rows, local)


def _select(rank: ndarray, m: int, rows: ndarray | None) -> list[ndarray]:
    if not isinstance(m, int | np.integer) or isinstance(m, bool) or m < 1:
        raise ValueError("m must be a positive integer")
    rows = np.arange(len(rank), dtype=np.int64) if rows is None else rows
    if rows.shape != (len(rank),) or (len(rows) > 1 and not (np.diff(rows) > 0).all()):
        raise ValueError("rows must be one ascending position per row of bits")
    return [_order(rank[:, q], rows, int(m)) for q in range(rank.shape[1])]


def candidates(bits: ndarray, query: ndarray, m: int, rows: ndarray | None = None) -> list[ndarray]:
    """Per query the global positions of the ``min(m, rows)`` rows with the smallest Hamming
    distance, nearest first, a tie going to the smaller position.

    ``rows`` holds the ascending global positions of the rows of ``bits`` (default
    ``0 .. n - 1``), as a filtered search has them."""
    return _select(hamming(bits, query), m, rows)


def candidates_signed(
    bits: ndarray, codes: ndarray, m: int, rows: ndarray | None = None
) -> list[ndarray]:
    """Like ``candidates`` for variant B: the largest ``signed_sums`` first, a tie going to
    the smaller position."""
    return _select(-signed_sums(bits, codes), m, rows)


def pad(found: list[ndarray]) -> ndarray:
    """The rectangular id matrix of the queries, padded with -1 on the right to the widest
    list of the whole call, never per block of queries."""
    width = max((len(item) for item in found), default=0)
    out = np.full((len(found), width), -1, dtype=np.int64)
    for row, item in enumerate(found):
        out[row, : len(item)] = item
    return out


def int4_codes(queries: ndarray) -> ndarray:
    """The int4 codes of float32 queries (variant B), int8 in -7 to 7: with ``s = max |q_j|``
    in float32, ``u_j = rint(float64(q_j) * 7 / float64(s))``, ties to even. A query of zeros
    or with a value that is not finite has no scale and raises."""
    _check_matrix(queries, "queries")
    if queries.dtype != np.float32:
        raise ValueError("queries must be float32")
    if not np.isfinite(queries).all():
        raise ValueError("queries must be finite")
    scale = np.abs(queries).max(axis=1)
    if not (scale > 0).all():
        raise ValueError("a query of zeros has no int4 codes")
    codes = _round(queries.astype(np.float64) * INT4_MAX / scale.astype(np.float64)[:, None])
    return codes.astype(np.int8)


def signed_sums(bits: ndarray, codes: ndarray) -> ndarray:
    """The ranks of variant B, shape ``(rows, queries)``, int32: ``S = sum_j u_j * b_j`` over
    the first ``d = codes.shape[1]`` bits of every row. A larger ``S`` ranks first; it ranks
    like ``sum_j u_j * (2 * b_j - 1)`` because ``sum_j u_j`` is constant per query."""
    _check_packed(bits, "bits")
    if not isinstance(codes, ndarray) or codes.dtype != np.int8 or codes.ndim != 2:
        raise ValueError("codes must be an int8 matrix")
    dim = codes.shape[1]
    if bits.shape[1] != -(-dim // 8) or dim < 1:
        raise ValueError("bits must have ceil(d / 8) bytes per row for d codes per query")
    if np.abs(codes.astype(np.int16)).max(initial=0) > INT4_MAX:
        raise ValueError("the codes of a query lie in [-7, 7]")
    wide = codes.astype(np.int32).T
    step = _row_block(dim)
    out = np.empty((len(bits), len(codes)), dtype=np.int32)
    for start in range(0, len(bits), step):
        unpacked = np.unpackbits(bits[start : start + step], axis=1, count=dim, bitorder="big")
        out[start : start + step] = unpacked.astype(np.int32) @ wide
    return out
