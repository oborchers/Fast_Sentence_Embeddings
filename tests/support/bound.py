"""The bound of the tolerance contract of device backends (ARCHITECTURE.md, "Tolerance
contract of device backends"), computed from a conformance case, independently of the
backend's own bounds.

After pooling, per row ``||g - c|| <= (TAU + 2 gamma_K) m + DELTA K sqrt(d)``: g the
device's row, c the cpu row (the reference's bits), m the norm of ``(1/n) sum |f| |x|`` in
float64 from the cpu's float32 factors (``weight``, ``weight * scale``, ``(1/m) *
oov_weight`` times the ngram scale, each rounded as the cpu rounds it) and the stored
values widened exactly, K the tokens plus ngram buckets, d the dimension. A linear step
with matrix A makes E ``||A|| E + R_cpu + R_device``, each R from the magnitudes of the
step's products and the lengths of its reductions on that side's input (the device's
within E of the cpu's in every value), with the absolute error of underflow; normalize
makes it ``2 E / r`` plus the rounding of both normalizations, r the norm of the cpu row
before it. Rows of empty sentences, and before normalize rows that could be zero or whose
squared norm could be subnormal on either side, must be the cpu's bits: no bound holds
there, so the host must have computed them.
"""

from __future__ import annotations

import numpy as np

from fse import _reference
from fse._contract import OOV_TOKEN, Normalize, Project, RemoveComponents

TAU = 1e-5
DELTA = 2.0**-125
U = 2.0**-24
UNDERFLOW = 2.0**-149  # the absolute error of an operation in the subnormal range
F32, F64 = np.float32, np.float64


def gamma(k):
    return k * U / (1 - k * U)


def contributions(case) -> np.ndarray:
    """K of every sentence: its tokens plus the ngram buckets of its unknown words."""
    unknown = iter(case.ngrams or ())
    return np.array(
        [len(s) + sum(len(next(unknown)) for t in s if t == OOV_TOKEN) for s in case.sentences],
        F64,
    )


def _terms(case, sentence, unknown):
    """The cpu's float32 factors and stored rows of a sentence, in order."""
    a = case.arrays
    for t in sentence:
        if t != OOV_TOKEN:
            yield _reference._factor(case.weights[t], a["scales"], t), a["rows"][t]
            continue
        buckets = next(unknown)
        if not buckets:
            continue
        shared = F32(F32(F32(1) / F32(len(buckets))) * F32(case.oov_weight))
        for b in buckets:
            yield _reference._factor(shared, a.get("ngram_scales"), b), a["ngram_rows"][b]


def magnitudes(case) -> np.ndarray:
    """m of every sentence, in float64 from the factors and the exactly widened values."""
    dim = case.base.shape[1]
    sums = np.zeros((len(case.sentences), dim))
    unknown = iter(case.ngrams or ())
    for s, sentence in enumerate(case.sentences):
        for factor, row in _terms(case, sentence, unknown):
            sums[s] += abs(float(factor)) * np.abs(np.asarray(row, F32).astype(F64))
        if sentence:
            sums[s] /= len(sentence)
    return np.linalg.norm(sums, axis=1)


def _remove_rounding(x: np.ndarray, step: RemoveComponents) -> np.ndarray:
    """R of remove_components on inputs whose values are at most ``x`` in magnitude:
    coefficients ``dot16(x, u_j) * w_j``, their k-term fma sum per value, the subtraction."""
    u = np.asarray(step.components, F64)
    k, d = u.shape
    w = np.ones(k) if step.weights is None else np.abs(np.asarray(step.weights, F64))
    sums = x @ np.abs(u).T  # sum_l |x_l u_jl|
    coef = sums * w  # bounds |coefficient|
    coef_err = gamma(d) * sums * w + U * coef + d * UNDERFLOW
    projection = coef @ np.abs(u)
    error = coef_err @ np.abs(u) + gamma(k) * projection + k * UNDERFLOW
    error += U * (x + projection) + UNDERFLOW
    return np.linalg.norm(error, axis=1)


def _project_rounding(x: np.ndarray, step: Project) -> np.ndarray:
    """R of project: ``dot16(x, b_j)`` for every j, d products each."""
    basis = np.abs(np.asarray(step.basis, F64))
    error = gamma(basis.shape[1]) * (x @ basis.T) + basis.shape[1] * UNDERFLOW
    return np.linalg.norm(error, axis=1)


def _linear(rows: np.ndarray, error: np.ndarray, step):
    """E after a linear step on the cpu rows ``rows`` (the device's within E per value)."""
    c = np.abs(rows.astype(F64))
    g = c + error[:, None]
    if isinstance(step, RemoveComponents):
        u = np.asarray(step.components, F64)
        w = np.ones(len(u)) if step.weights is None else np.asarray(step.weights, F64)
        norm = np.linalg.norm(np.eye(u.shape[1]) - u.T @ np.diag(w) @ u, 2)
        return norm * error + _remove_rounding(c, step) + _remove_rounding(g, step)
    assert isinstance(step, Project)
    norm = np.linalg.norm(np.asarray(step.basis, F64), 2)
    return norm * error + _project_rounding(c, step) + _project_rounding(g, step)


def bounds(case, steps=(), rounding=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cpu rows of ``case`` after ``steps``, the bound of every row and the rows that
    must be the cpu's bits; without ``rounding``, the pooling bound of the first contract,
    without the term ``2 gamma_K`` (for the witness that needs it)."""
    rows = _reference.pool_weighted_mean(case.batch, *case.args())
    k = contributions(case)
    exact = k == 0
    pool = (TAU + 2 * gamma(k) if rounding else TAU) * magnitudes(case)
    error = np.where(exact, 0.0, pool + DELTA * k * np.sqrt(rows.shape[1]))
    for step in steps:
        after = _reference.postprocess(rows, (step,), case.first_row)
        if isinstance(step, Normalize):
            r = np.linalg.norm(rows.astype(F64), axis=1)
            exact |= (r <= error) | (r - error < 2.0**-63)  # zero, or squares subnormal
            r = np.maximum(r, 2.0**-63)
            error = 2 * error / r + 2 * (gamma(rows.shape[1]) / 2 + 4 * U)
        else:
            error = _linear(rows, error, step)
        rows = after
    return rows, error, exact


def assert_within_bound(got: np.ndarray, case, steps=(), name: str = "", rounding=True) -> float:
    """``got`` (a device's rows of ``case`` after ``steps``) keeps the contract: the shape
    and dtype of the cpu rows, their bits where no bound holds and every other row within
    its bound. Gives the largest ratio of error to bound (0 where both are zero)."""
    want, bound, exact = bounds(case, steps, rounding)
    assert got.shape == want.shape, (name, got.shape)
    assert got.dtype == want.dtype, (name, got.dtype)
    assert np.isfinite(got).all(), f"{name}: nonfinite rows"
    assert np.isfinite(want).all(), f"{name}: nonfinite cpu rows"
    g, w = got.view(np.uint32), want.view(np.uint32)
    differs = np.nonzero(exact & (g != w).any(axis=1))[0]
    assert not len(differs), f"{name}: row {differs[0]} must be the cpu's (no bound holds)"
    error = np.linalg.norm(got.astype(F64) - want.astype(F64), axis=1)
    beyond = np.nonzero(~exact & (error > bound))[0]
    assert not len(beyond), (
        f"{name}: row {beyond[0]} misses the bound: {error[beyond[0]]!r} > {bound[beyond[0]]!r}"
    )
    ratios = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
    return float(ratios.max(initial=0))
