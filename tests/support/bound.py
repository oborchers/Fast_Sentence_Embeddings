"""The bound of the tolerance contract of device backends (ARCHITECTURE.md, "Tolerance
contract of device backends"), computed from a conformance case, independently of the
backend's own bounds.

After pooling, per row ``||g - c|| <= (TAU + 2 gamma_K) m + DELTA K sqrt(d)``: g the
device's row, c the cpu row (the reference's bits), m the norm of ``(1/n) sum |f| |x|`` in
float64 from the cpu's float32 factors (``weight``, ``weight * scale``, ``(1/m) *
oov_weight`` times the ngram scale, each rounded as the cpu rounds it) and the stored
values widened exactly, K the tokens plus ngram buckets, d the dimension. Normalize makes
it ``2 E / r`` plus the rounding of both normalizations, r the norm of the cpu row before
it. Rows of empty sentences, before normalize rows that could be zero or whose squared
norm could be subnormal on either side (or is below 2**-100 on the device), and every
row after a linear step (which a device backend leaves to its host) must be the cpu's
bits: no bound holds there, so the host must have computed them.
"""

from __future__ import annotations

import numpy as np

from fse import _reference
from fse._contract import OOV_TOKEN, Normalize

TAU = 1e-5
DELTA = 2.0**-125
U = 2.0**-24
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


def bounds(case, steps=(), rounding=True, device=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cpu rows of ``case`` after ``steps``, the bound of every row and the rows that
    must be the cpu's bits; without ``rounding``, the pooling bound of the first contract,
    without the term ``2 gamma_K`` (for the witness that needs it). ``device``, the
    device's pooled rows, is needed before normalize: a row whose squared norm there is
    below 2**-100 must be the cpu's too. Linear steps run on the host for a device
    backend (fse 2.0 folds them), so after them every row must be the cpu's."""
    rows = _reference.pool_weighted_mean(case.batch, *case.args())
    k = contributions(case)
    exact = k == 0
    pool = (TAU + 2 * gamma(k) if rounding else TAU) * magnitudes(case)
    error = np.where(exact, 0.0, pool + DELTA * k * np.sqrt(rows.shape[1]))
    if any(not isinstance(step, Normalize) for step in steps):
        exact = np.ones(len(rows), bool)
    elif steps:
        assert device is not None, "the device's pooled rows are needed before normalize"
        r = np.linalg.norm(rows.astype(F64), axis=1)
        squares = np.linalg.norm(device.astype(F64), axis=1) ** 2
        exact |= (r <= error) | (r - error < 2.0**-63) | (squares < 2.0**-100)
        r = np.maximum(r, 2.0**-63)
        error = 2 * error / r + 2 * (gamma(rows.shape[1]) / 2 + 4 * U)
    rows = _reference.postprocess(rows, steps, case.first_row) if steps else rows
    return rows, error, exact


def assert_within_bound(
    got: np.ndarray, case, steps=(), name: str = "", rounding=True, device=None
) -> float:
    """``got`` (a device's rows of ``case`` after ``steps``; ``device`` its pooled rows,
    for normalize) keeps the contract: the shape and dtype of the cpu rows, their bits
    where no bound holds and every other row within its bound. Gives the largest ratio of
    error to bound (0 where both are zero)."""
    want, bound, exact = bounds(case, steps, rounding, device)
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
