#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The numpy reference of the 2.0 numerics (``transform_version`` 2).

Every function here computes exactly what ARCHITECTURE.md, "Numerical contract",
prescribes, bit for bit, in plain numpy: the oracle for the compiled and device
backends. It is written for clarity, not speed, and loops over tokens in Python.

numpy has no float32 fused multiply-add, so ``fma32`` emulates it: the product of two
float32 values is exact in float64, the sum is rounded to odd in float64 (53 bits) and
then rounded to nearest float32 (24 bits). Round-to-odd with at least two extra bits
(53 >= 24 + 2) makes the second rounding give the correctly rounded result.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from fse._contract import (
    OOV_TOKEN,
    Batch,
    Normalize,
    Project,
    RemoveComponents,
    Step,
    check_steps,
)

_F32 = np.float32
_LANES = 16
_INT8_TOP = 127
_FLOAT16_MAX = float(np.finfo(np.float16).max)


def fma32(a, b, c) -> ndarray:
    """``a * b + c`` with a single rounding to float32, elementwise."""
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.asarray(a, dtype=_F32).astype(np.float64)
        y = np.asarray(b, dtype=_F32).astype(np.float64)
        z = np.asarray(c, dtype=_F32).astype(np.float64)
        p = x * y  # exact: 24 + 24 bits fit into 53
        s = p + z
        # TwoSum: the exact rounding error of s, for finite operands.
        bp = s - z
        err = (p - bp) + (z - (s - bp))
        bits = s.view(np.uint64) if s.ndim else np.array(s).view(np.uint64)
        even = (bits & np.uint64(1)) == 0
        inexact = np.isfinite(s) & (err != 0)
        toward = np.where(err > 0, np.inf, -np.inf)
        s = np.where(inexact & even, np.nextafter(s, toward), s)
        return np.asarray(s.astype(_F32))


def dot16(rows: ndarray, other: ndarray) -> ndarray:
    """The dot product of every row with ``other``, in the fixed order of the contract.

    ``other`` is one vector for all rows or one row per row (``dot16(rows, rows)`` gives
    the squared norms).

    16 partial sums take the products of the indices congruent to their lane, by fused
    multiply-adds in index order; they are then added pairwise, 8, 4, 2 and 1 lanes apart.
    """
    rows = np.asarray(rows, dtype=_F32)
    other = np.broadcast_to(np.asarray(other, dtype=_F32), rows.shape)
    partial = np.zeros((rows.shape[0], _LANES), dtype=_F32)
    for i in range(rows.shape[1]):
        lane = i % _LANES
        partial[:, lane] = fma32(rows[:, i], other[:, i], partial[:, lane])
    width = _LANES // 2
    with np.errstate(over="ignore"):
        while width:
            partial[:, :width] = partial[:, :width] + partial[:, width : 2 * width]
            width //= 2
    return partial[:, 0].copy()


def _stored(rows: ndarray, index: int) -> ndarray:
    """Row ``index`` of stored float32, float16 or int8 values, exactly as float32."""
    return np.asarray(rows[index], dtype=_F32)


def _factor(weight, scales: ndarray | None, index: int) -> np.float32:
    weight = _F32(weight)
    return weight if scales is None else _F32(weight * _F32(scales[index]))


def accumulate(
    batch: Batch,
    vectors: ndarray,
    weights: ndarray,
    scales: ndarray | None = None,
    ngram_vectors: ndarray | None = None,
    ngram_scales: ndarray | None = None,
    oov_weight: float | None = None,
) -> tuple[ndarray, ndarray]:
    """The accumulators of ``weighted_mean@1`` and the token counts of every sentence.

    ``vectors`` (and ``ngram_vectors``) hold the stored word (and ngram) rows as float32,
    float16 or int8 codes, with ``scales`` (``ngram_scales``) for int8. ``weights`` holds
    one float32 weight per word, ``oov_weight`` the weight of unknown FastText words.
    Raises a ValueError naming the row if an accumulator overflows.
    """
    dim = vectors.shape[1]
    accs = np.zeros((batch.n_sentences, dim), dtype=_F32)
    counts = np.diff(batch.offsets)
    k = 0  # the next unknown word of the batch
    with np.errstate(over="ignore", invalid="ignore"):
        for s in range(batch.n_sentences):
            acc = np.zeros(dim, dtype=_F32)
            for token in batch.tokens[batch.offsets[s] : batch.offsets[s + 1]]:
                if token == OOV_TOKEN:
                    acc = _add_unknown(acc, batch, k, ngram_vectors, ngram_scales, oov_weight)
                    k += 1
                else:
                    factor = _factor(weights[token], scales, int(token))
                    acc = fma32(factor, _stored(vectors, int(token)), acc)
            if not np.isfinite(acc).all():
                raise ValueError(f"row {batch.first_row + s} contains NaN or infinite values")
            accs[s] = acc
    return accs, counts


def _add_unknown(
    acc: ndarray,
    batch: Batch,
    k: int,
    ngram_vectors: ndarray | None,
    ngram_scales: ndarray | None,
    oov_weight: float | None,
) -> ndarray:
    """``acc`` plus the ngram rows of unknown word ``k`` of the batch, each with weight
    oov_weight / (its number of buckets)."""
    if batch.ngrams is None or batch.ngram_offsets is None:
        raise ValueError("an unknown FastText word needs ngram buckets")
    if ngram_vectors is None:
        raise ValueError("an unknown FastText word needs ngram vectors")
    if oov_weight is None:
        raise ValueError("an unknown FastText word needs oov_weight")
    buckets = batch.ngrams[batch.ngram_offsets[k] : batch.ngram_offsets[k + 1]]
    if len(buckets) == 0:
        return acc  # counts for the length, adds nothing
    shared = _F32(_F32(_F32(1) / _F32(len(buckets))) * _F32(oov_weight))
    for bucket in buckets:
        factor = _factor(shared, ngram_scales, int(bucket))
        acc = fma32(factor, _stored(ngram_vectors, int(bucket)), acc)
    return acc


def pool_weighted_mean(
    batch: Batch,
    vectors: ndarray,
    weights: ndarray,
    scales: ndarray | None = None,
    ngram_vectors: ndarray | None = None,
    ngram_scales: ndarray | None = None,
    oov_weight: float | None = None,
) -> ndarray:
    """``weighted_mean@1`` of every sentence of ``batch``, as float32 rows.

    Takes the arguments of ``accumulate``; an empty sentence is the zero row.
    """
    accs, counts = accumulate(
        batch, vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight
    )
    out = np.zeros_like(accs)
    for s, (acc, n) in enumerate(zip(accs, counts, strict=True)):
        if n:
            out[s] = _F32(_F32(1) / _F32(n)) * acc
    return out


def inverse_norms(stored: ndarray, first_row: int = 0) -> ndarray:
    """``float32(1 / sqrt(s))`` per stored row, s the float64 sum of squares; 0 for zero rows.

    Raises a ValueError naming the row (from ``first_row``) if an inverse norm is beyond
    float32 (a nonzero row too small).
    """
    values = np.asarray(stored, dtype=np.float64)
    squares = np.zeros(len(values))
    for i in range(values.shape[1]):
        squares += values[:, i] * values[:, i]
    out = np.zeros(len(values), dtype=_F32)
    nonzero = squares > 0
    inverse = 1 / np.sqrt(squares[nonzero])
    with np.errstate(over="ignore"):
        out[nonzero] = inverse.astype(_F32)
    bad = np.flatnonzero(~np.isfinite(out))
    if len(bad):
        raise ValueError(f"the inverse norm of row {first_row + int(bad[0])} is beyond float32")
    return out


def score(stored: ndarray, scales: ndarray | None, weights: ndarray, bias=0.0) -> ndarray:
    """The linear score ``dot16(stored, w) * scale + bias`` of every stored row.

    Raises a ValueError naming the row if a score overflows.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        dots = dot16(np.asarray(stored, dtype=_F32), np.asarray(weights, dtype=_F32))
        if scales is not None:
            dots = dots * np.asarray(scales, dtype=_F32)
        out = (dots + _F32(bias)).astype(_F32)
    bad = np.flatnonzero(~np.isfinite(out))
    if len(bad):
        raise ValueError(f"the score of row {int(bad[0])} overflows float32")
    return out


def cosine(stored: ndarray, inverse: ndarray, query: ndarray) -> ndarray:
    """The cosine of every stored row with ``query``: ``dot16(stored, q) * inverse norm``.

    ``query`` is normalized by ``normalize@1`` first, so a zero query or one whose norm
    underflows or overflows raises; the int8 scale cancels and is not used.
    """
    query = np.asarray(query, dtype=_F32)
    if not (query != 0).any():
        raise ValueError("the query is the zero vector and has no direction")
    unit = postprocess(query[None, :], [Normalize()])[0]
    with np.errstate(over="ignore", invalid="ignore"):
        out = (dot16(np.asarray(stored, dtype=_F32), unit) * inverse).astype(_F32)
    bad = np.flatnonzero(~np.isfinite(out))
    if len(bad):
        raise ValueError(f"the cosine of row {int(bad[0])} is not finite")
    return out


def cosines(stored: ndarray, inverse: ndarray, queries: ndarray) -> ndarray:
    """``cosine`` of every stored row with every query, as a (rows, queries) array: column
    j is ``cosine(stored, inverse, queries[j])`` bit for bit.

    Every query is checked before any is used: the first zero query raises, then the
    first one ``normalize@1`` cannot normalize; then the first query with a non-finite
    cosine raises, naming its first such row.
    """
    queries = _matrix(queries, "queries")
    for j, query in enumerate(queries):
        if not (query != 0).any():
            raise ValueError(f"query {j} is the zero vector and has no direction")
    for j, query in enumerate(queries):
        try:
            postprocess(query[None, :], [Normalize()])
        except ValueError as error:
            raise ValueError(f"query {j}: {error}") from None
    out = np.empty((len(stored), len(queries)), dtype=_F32)
    for j, query in enumerate(queries):
        try:
            out[:, j] = cosine(stored, inverse, query)
        except ValueError as error:
            raise ValueError(f"query {j}: {error}") from None
    return out


def scores(stored: ndarray, scales: ndarray | None, weights: ndarray, bias=0.0) -> ndarray:
    """``score`` of every stored row with every weight vector, as a (rows, vectors) array:
    column j is ``score(stored, scales, weights[j], bias[j])`` bit for bit, ``bias`` one
    number for all or one per vector. The first vector with an overflow raises."""
    weights = _matrix(weights, "weights")
    biases = np.broadcast_to(np.asarray(bias, dtype=_F32), (len(weights),))
    out = np.empty((len(stored), len(weights)), dtype=_F32)
    for j, w in enumerate(weights):
        try:
            out[:, j] = score(stored, scales, w, biases[j])
        except ValueError as error:
            raise ValueError(f"weights {j}: {error}") from None
    return out


def _matrix(value: ndarray, name: str) -> ndarray:
    value = np.asarray(value, dtype=_F32)
    if value.ndim != 2:
        raise ValueError(
            f"{name} must be a matrix with one vector per row, not shape {value.shape}"
        )
    return value


def top_k(scores: ndarray, k: int) -> ndarray:
    """The rows of the k largest scores; equal scores go to the smaller row id."""
    order = np.lexsort((np.arange(len(scores)), -scores.astype(np.float64)))
    return order[:k]


def _check_finite(rows: ndarray, first_row: int):
    bad = np.flatnonzero(~np.isfinite(rows).all(axis=1))
    if len(bad):
        raise ValueError(f"row {first_row + int(bad[0])} contains NaN or infinite values")


def postprocess(rows: ndarray, steps: Sequence[Step], first_row: int = 0) -> ndarray:
    """Apply ``steps`` to every float32 row; returns new rows and checks they are finite.

    A step whose result overflows raises a ValueError naming the row, as the contract
    requires, instead of a numpy warning.
    """
    out = np.array(rows, dtype=_F32, copy=True)
    _check_finite(out, first_row)
    with np.errstate(over="ignore", invalid="ignore"):
        for step in check_steps(steps, out.shape[1]):
            if isinstance(step, RemoveComponents):
                projection = np.zeros_like(out)
                for j, component in enumerate(step.components):
                    coefficient = dot16(out, component)
                    if step.weights is not None:
                        coefficient = coefficient * step.weights[j]
                    projection = fma32(coefficient[:, None], component[None, :], projection)
                out = out - projection
            elif isinstance(step, Project):
                out = np.stack([dot16(out, axis) for axis in step.basis], axis=1).astype(_F32)
            elif isinstance(step, Normalize):
                squares = dot16(out, out)
                nonzero = (out != 0).any(axis=1)
                bad = np.flatnonzero(nonzero & ((squares == 0) | ~np.isfinite(squares)))
                if len(bad):
                    raise ValueError(
                        f"row {first_row + int(bad[0])} cannot be normalized: the sum of its "
                        "squares underflows or overflows float32"
                    )
                inverse = np.where(nonzero, _F32(1) / np.sqrt(np.where(nonzero, squares, 1)), 0)
                out = out * inverse.astype(_F32)[:, None]
            _check_finite(out, first_row)
    return out


def quantize_rows(rows: ndarray, dtype: str) -> tuple[ndarray, ndarray | None]:
    """Store finite float32 rows as ``dtype`` codes and, for int8, one scale per row."""
    rows = np.asarray(rows, dtype=_F32)
    if not np.isfinite(rows).all():
        raise ValueError("rows to store must be finite")
    if dtype == "float32":
        return rows.copy(), None
    if dtype == "float16":
        if (np.abs(rows) > _FLOAT16_MAX).any():
            raise ValueError(f"values beyond the float16 range of +-{_FLOAT16_MAX:g}")
        return rows.astype(np.float16), None
    if dtype != "int8":
        raise ValueError(f"unknown storage dtype {dtype!r}")
    absmax = np.abs(rows).max(axis=1) if rows.shape[1] else np.zeros(len(rows), dtype=_F32)
    scales = (absmax / _F32(_INT8_TOP)).astype(_F32)
    with np.errstate(over="ignore"):
        beyond = ~np.isfinite(scales * _F32(_INT8_TOP))
    scales[beyond] = np.nextafter(scales[beyond], _F32(0))
    divisor = np.where(scales > 0, scales, _F32(1))[:, None]
    codes = np.clip(np.rint(rows / divisor), -_INT8_TOP, _INT8_TOP).astype(np.int8)
    return codes, scales


def dequantize(codes: ndarray, scales: ndarray | None) -> ndarray:
    """The float32 rows of ``codes``: ``code * scale`` with one rounding for int8."""
    rows = np.array(codes, dtype=_F32, copy=True)
    if scales is not None:
        rows *= np.asarray(scales, dtype=_F32)[:, None]
    return rows
