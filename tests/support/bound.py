"""The bound of the tolerance contract of device backends (ARCHITECTURE.md, "Tolerance
contract of device backends"), computed from a conformance case.

Per row ``||g - c|| <= TAU m``: g the device's row, c the cpu row (the reference's bits),
m the norm of ``(1/n) sum |f| |x|`` over the stored inputs in float64 (the int8 scale in
the factor, every ngram bucket a contribution with its factor ``(1/m) oov_weight`` times
its scale, n the tokens, an unknown word without buckets counting for n only). A linear
step with matrix A multiplies the bound by the spectral norm of A and adds 4 ulp of the
largest value of the cpu row after it; ``normalize`` makes it ``2 E / r`` plus 4 ulp, r the
norm of the cpu row before it. The rows of empty sentences must be the cpu's bits, and
nonfinite rows are refused before any comparison.
"""

from __future__ import annotations

import numpy as np

from fse import _reference
from fse._contract import OOV_TOKEN, Normalize, Project, RemoveComponents

TAU = 1e-5
F64 = np.float64


def magnitudes(case) -> np.ndarray:
    """m of every sentence of ``case``, from its stored arrays."""
    a = case.arrays
    rows = np.abs(a["rows"].astype(F64))
    factors = np.abs(case.weights.astype(F64))
    if a["scales"] is not None:
        factors = factors * np.abs(a["scales"].astype(F64))
    sums = np.zeros((len(case.sentences), rows.shape[1]))
    unknown = iter(case.ngrams or ())
    for s, sentence in enumerate(case.sentences):
        for t in sentence:
            if t != OOV_TOKEN:
                sums[s] += factors[t] * rows[t]
                continue
            buckets = next(unknown)
            for b in buckets:
                sums[s] += _ngram_factor(case, b, len(buckets)) * np.abs(a["ngram_rows"][b])
        if sentence:
            sums[s] /= len(sentence)
    return np.linalg.norm(sums, axis=1)


def _ngram_factor(case, bucket: int, count: int) -> float:
    g = abs(float(np.float32(case.oov_weight))) / count
    scales = case.arrays.get("ngram_scales")
    return g if scales is None else g * abs(float(scales[bucket]))


def _matrix(step, dim: int) -> np.ndarray:
    if isinstance(step, RemoveComponents):
        u = np.asarray(step.components, F64)
        w = np.ones(len(u)) if step.weights is None else np.asarray(step.weights, F64)
        return np.eye(dim) - u.T @ np.diag(w) @ u
    assert isinstance(step, Project)
    return np.asarray(step.basis, F64)


def _ulps(rows: np.ndarray) -> np.ndarray:
    return 4 * np.spacing(np.abs(rows).max(axis=1, initial=0)).astype(F64)


def bounds(case, steps=()) -> tuple[np.ndarray, np.ndarray]:
    """The cpu rows of ``case`` after ``steps`` and the bound of every row."""
    rows = _reference.pool_weighted_mean(case.batch, *case.args())
    error = TAU * magnitudes(case)
    for step in steps:
        after = _reference.postprocess(rows, (step,), case.first_row)
        if isinstance(step, Normalize):
            norm = np.linalg.norm(rows.astype(F64), axis=1)
            scaled = np.divide(2 * error, norm, out=np.zeros_like(error), where=norm > 0)
            error = scaled + _ulps(after)
        else:
            error = error * np.linalg.norm(_matrix(step, rows.shape[1]), 2) + _ulps(after)
        rows = after
    return rows, error


def assert_within_bound(got: np.ndarray, case, steps=(), name: str = "") -> float:
    """``got`` (a device's rows of ``case`` after ``steps``) keeps the contract: the shape
    and dtype of the cpu rows, their bits for empty sentences and every row within its
    bound. Gives the largest ratio of error to bound (0 where both are zero)."""
    want, bound = bounds(case, steps)
    assert got.shape == want.shape, (name, got.shape)
    assert got.dtype == want.dtype, (name, got.dtype)
    assert np.isfinite(got).all(), f"{name}: nonfinite rows"
    assert np.isfinite(want).all(), f"{name}: nonfinite cpu rows"
    empty = np.array([not s for s in case.sentences], bool)
    g, w = got.view(np.uint32), want.view(np.uint32)
    assert (g[empty] == w[empty]).all(), f"{name}: the row of an empty sentence differs"
    error = np.linalg.norm(got.astype(F64) - want.astype(F64), axis=1)
    beyond = np.nonzero(error > bound)[0]
    assert not len(beyond), (
        f"{name}: row {beyond[0]} misses the bound: {error[beyond[0]]!r} > {bound[beyond[0]]!r}"
    )
    ratios = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
    return float(ratios.max(initial=0))
