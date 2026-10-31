#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Conservative per-row error bounds of the mlx backend and the checks they drive
(docs/architecture/backends.md, "Tolerance contract of device backends" and "MLX").

The kernel gives per row ``M = sum |f| ||x||`` over its nonzero factors, an upper bound
of every partial sum of every value (``|x_i| <= ||x||``) and, divided by n, of m (the
triangle inequality). From it: E, the bound of the row's distance to the cpu row, ``(tau
+ 2 gamma_K) m + delta``; the rows whose sums could overflow on the cpu although the
device's did not; before ``normalize`` the rows that could be zero or tiny on either side
(``||g|| <= 2 E``, or ``||g|| - E`` below 2**-50, where a sum of squares may be
subnormal on the cpu or below 2**-100 on the device) or whose squares could overflow;
and before float16 storage the rows whose values could reach its range on the cpu. All in
float64, rounded towards the safe side; a bound that is not finite sends its row to the
host.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

TAU = 1e-5
U = 2.0**-24
DELTA = 2.0**-125  # per contribution and value
LOW = 2.0**-50  # a row norm from which no sum of squares is near the subnormal range
HIGH = 2.0**60  # a row norm below which no square or sum of squares overflows
OVERFLOW = 2.0**127  # partial sums below it cannot round to infinity on the cpu
FLOAT16_MAX = 65504.0
SAFE = 1 + 2.0**-40  # a margin for the float64 arithmetic of these bounds
F64 = np.float64


def gamma(k: ndarray | float) -> ndarray | float:
    """``k u / (1 - k u)``, the bound of a float32 sum of k terms (k u < 1)."""
    return k * U / (1 - k * U)


def magnitude(sums: ndarray, k: ndarray) -> ndarray:
    """An upper bound of the exact ``sum |f| ||x||`` from the kernel's float32 sum of K
    nonnegative terms, which is at least ``(1 - K u)`` times the exact one (one rounding
    per term) less 2**-126 per term it flushes."""
    k = k.astype(F64)
    with np.errstate(invalid="ignore", over="ignore"):
        return (sums.astype(F64) + k * 2.0**-126) / (1 - k * U) * SAFE


def pooling(sums: ndarray, n: ndarray, k: ndarray, dim: int) -> ndarray:
    """E of every row (0 for an empty one), ``n`` its tokens, ``k`` its contributions."""
    k = k.astype(F64)
    m = np.divide(magnitude(sums, k), n, out=np.zeros(len(k)), where=n > 0)
    with np.errstate(invalid="ignore", over="ignore"):
        error = ((TAU + 2 * gamma(k)) * m + DELTA * k * np.sqrt(dim)) * SAFE
    return np.where(n > 0, error, 0.0)


def overflows(sums: ndarray, k: ndarray) -> ndarray:
    """Rows whose partial sums could reach the float32 overflow on the cpu (or whose bound
    is not finite): the device's finite accumulator does not exclude a cpu overflow."""
    bound = magnitude(sums, k)
    return ~np.isfinite(bound) | (bound >= OVERFLOW)


def norms(rows: ndarray) -> ndarray:
    """The Euclidean norms of float32 rows, in float64 (a relative error of about 2**-50,
    which SAFE covers)."""
    return np.sqrt(np.einsum("ij,ij->i", rows, rows, dtype=F64))


def device_norms(squares: ndarray, dim: int) -> tuple[ndarray, ndarray]:
    """Lower and upper bounds of a row's norm from the kernel's dot16 of the row with
    itself, which may flush products below 2**-126 and round each of its operations."""
    s = squares.astype(F64)
    with np.errstate(invalid="ignore", over="ignore"):  # a nonfinite sum: flagged below
        slack = gamma(dim + 4) * np.abs(s) + dim * 2.0**-125
        low = np.sqrt(np.maximum(s - slack, 0.0)) / SAFE
        high = np.sqrt(s + slack) * SAFE
    bad = ~np.isfinite(s)
    return np.where(bad, 0.0, low), np.where(bad, np.inf, high)


def unnormalizable(low: ndarray, high: ndarray, error: ndarray) -> ndarray:
    """Rows normalize must not take from the device: zero or tiny on either side, with
    squares that may overflow, or without a finite bound."""
    with np.errstate(invalid="ignore"):
        near = (low <= 2 * error) | (low - error < LOW) | (high + error >= HIGH)
    return near | ~np.isfinite(error)


def beyond_float16(rows: ndarray, error: ndarray) -> ndarray:
    """Rows whose values could reach the float16 range on the cpu (within E of the
    device's), or without a finite bound: storing them, the cpu may raise."""
    with np.errstate(invalid="ignore"):
        near = np.abs(rows).max(axis=1, initial=0).astype(F64) + error >= FLOAT16_MAX
    return near | ~np.isfinite(error)
