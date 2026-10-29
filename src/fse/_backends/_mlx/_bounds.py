#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Conservative per-row error bounds of the mlx backend and the checks before normalize
(ARCHITECTURE.md, "Tolerance contract of device backends" and "MLX").

After pooling a row's distance to the cpu row is at most ``E = (tau + 2 gamma_K) m + delta``;
the kernel gives an upper bound of m, ``(1/n) sum |f| ||x||`` (the triangle inequality),
from the norms of the stored rows. A linear step with matrix A, run on the host on both
sides, makes it ``||A|| E + R_cpu + R_device``, R from the rounding of its products and
sums. Before ``normalize`` a row goes to the host when it could be zero or tiny on the
cpu (``||g|| <= 2 E``, or ``||g|| - E`` below 2**-50, where a sum of squares may be
subnormal on either side) or its squares could overflow. All in float64, rounded towards
the safe side; no floating-point environment of the worker changes float64 values this
far from its subnormal range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from fse._contract import Normalize, Project, RemoveComponents, Step

TAU = 1e-5
U = 2.0**-24
DELTA = 2.0**-125  # per contribution and value
LOW = 2.0**-50  # a row norm from which no sum of squares is near the subnormal range
HIGH = 2.0**60  # a row norm below which no square or sum of squares overflows
SAFE = 1 + 2.0**-40  # a margin for the float64 arithmetic of these bounds
F64 = np.float64


def gamma(k: ndarray | float) -> ndarray | float:
    """``k u / (1 - k u)``, the bound of a float32 sum of k terms (k u < 1)."""
    return k * U / (1 - k * U)


def pooling(magnitude: ndarray, n: ndarray, k: ndarray, dim: int) -> ndarray:
    """E of every row: ``magnitude`` is the kernel's float32 sum of ``|f| ||x||`` (each
    norm rounded up), ``n`` its tokens and ``k`` its contributions. The float32 sum of K
    nonnegative terms is at least ``(1 - gamma_K)`` times the exact one, less 2**-126 per
    flushed term; from K u = 1/4 on no useful bound follows (infinite: the host)."""
    k = k.astype(F64)
    usable = (n > 0) & (k * U < 0.25)
    shrink = np.where(usable, 1 - gamma(np.minimum(k, 2.0**22)), 1.0)
    exact = (magnitude.astype(F64) + k * 2.0**-126) / shrink
    m = np.divide(exact, n, out=np.zeros_like(exact), where=n > 0) * SAFE
    error = ((TAU + 2 * gamma(np.minimum(k, 2.0**22))) * m + DELTA * k * np.sqrt(dim)) * SAFE
    return np.where(n > 0, np.where(usable, error, np.inf), 0.0)


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
    """Rows normalize must not take from the device: zero or tiny on either side, or
    with squares that may overflow, given the bounds of the device row's norm."""
    return (low <= 2 * error) | (low - error < LOW) | (high + error >= HIGH)


@dataclass(frozen=True)
class Linear:
    """The bound of a linear step: ``||A||``, the norm of the matrix of absolute values
    that bounds its rounding, the length of its reductions and its output dimension."""

    norm: float
    absolute: float
    length: int
    dim: int

    def propagate(self, error: ndarray, g: ndarray) -> ndarray:
        """E after the step, ``g`` the norms of the device rows before it (the cpu rows'
        are at most ``g + E``); with the absolute errors of underflow in its operations."""
        rounding = gamma(self.length) * self.absolute * (2 * g + error)
        underflow = 2 * (self.length + 1) * 2.0**-149 * np.sqrt(self.dim)
        return (self.norm * error + rounding + underflow) * SAFE

    def overflows(self, error: ndarray, g: ndarray) -> ndarray:
        """Rows whose values could overflow in the step on either side."""
        return (g + error) * max(self.norm, self.absolute) * SAFE >= 2.0**100


def linear(step: Step, dim: int) -> Linear | None:
    """The bound of ``step`` on rows of ``dim`` values (None for normalize). For
    remove_components the cpu computes ``x - sum_j (dot16(x, u_j) w_j) u_j``, for
    project ``dot16(x, b_j)``: the rounding of every output value is at most
    ``gamma_L`` times that of ``(I + |U|^T |W| |U|) |x|`` or ``|B| |x|``."""
    if isinstance(step, Normalize):
        return None
    if isinstance(step, RemoveComponents):
        u = np.asarray(step.components, F64)
        w = np.ones(len(u)) if step.weights is None else np.asarray(step.weights, F64)
        matrix = np.eye(dim) - u.T @ np.diag(w) @ u
        absolute = np.eye(dim) + np.abs(u).T @ np.diag(np.abs(w)) @ np.abs(u)
        return Linear(_norm(matrix), _norm(absolute), dim + len(u) + 2, dim)
    assert isinstance(step, Project)
    basis = np.asarray(step.basis, F64)
    return Linear(_norm(basis), _norm(np.abs(basis)), dim, len(basis))


def _norm(matrix: ndarray) -> float:
    """The spectral norm, with a margin for the error of its computation."""
    return float(np.linalg.norm(matrix, 2)) * (1 + 2.0**-30)


def after_normalize(error: ndarray, g: ndarray, dim: int) -> ndarray:
    """E after normalize for rows it takes from the device: ``2 E / r`` (r at least
    ``g - E``) plus the rounding of both normalizations (the dot product, the square
    root, the reciprocal and one product per value)."""
    r = np.maximum(g - error, LOW)
    return (2 * error / r + 2 * gamma(dim + 4)) * SAFE
