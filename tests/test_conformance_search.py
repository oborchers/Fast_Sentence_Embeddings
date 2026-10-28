"""The conformance suite for the search: inverse norms, cosines and scores.

Every case runs for every backend in BACKENDS and must give the bits of fse._reference.
The meta-tests show that the cases tell the contract from plausible other arithmetic:
float32 BLAS products, zero-padded tails, a float32 sum of the norm squares and a fused
``dot * scale + bias``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fse import _reference
from support.conformance import BACKENDS, F32, TINY, _rng, assert_bits_equal

DTYPES = ["float32", "float16", "int8"]
DIMS = [1, 7, 16, 17, 33, 300]


def _stored(dtype, dim, seed=0):
    rows = _rng(dim + seed).standard_normal((40, dim)).astype(F32)
    rows[4] = 0
    rows[7] *= F32(1e-3)
    if dim > 1:
        rows[9, : dim // 2] = -0.0
    return _reference.quantize_rows(rows, dtype)


def _queries(dim, q, seed=1):
    return _rng(1000 * dim + q + seed).standard_normal((q, dim)).astype(F32)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS)
def test_inverse_norms(backend, dtype, dim):
    stored, _ = _stored(dtype, dim)
    want = _reference.inverse_norms(stored)
    assert_bits_equal(backend.inverse_norms(stored)[None, :], want[None, :], f"{dtype} {dim}")


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("q", [1, 3, 17])
def test_cosines_and_scores(backend, dtype, dim, q):
    stored, scales = _stored(dtype, dim)
    inverse = _reference.inverse_norms(stored)
    queries = _queries(dim, q)
    want = _reference.cosines(stored, inverse, queries)
    assert_bits_equal(backend.cosines(stored, inverse, queries), want, "cosines")
    biases = _queries(1, q, 2)[:, 0]
    want = _reference.scores(stored, scales, queries, biases)
    assert_bits_equal(backend.scores(stored, scales, queries, biases), want, "scores")


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_errors_like_the_reference(backend):
    big = F32(3e38)
    stored = np.array([[1, 0], [big, big]], F32)
    with pytest.raises(ValueError, match="query 1 is the zero vector"):
        backend.cosines(stored, np.ones(2, F32), np.array([[1, 0], [0, 0]], F32))
    with pytest.raises(ValueError, match=r"query 1: the cosine of row 1 is not finite"):
        backend.cosines(stored, np.ones(2, F32), np.array([[1, 0], [1, 1]], F32))
    with pytest.raises(ValueError, match=r"weights 0: the score of row 1 overflows"):
        backend.scores(stored, None, np.array([[big, big]], F32), 0.0)
    with pytest.raises(ValueError, match="inverse norm of row 0 is beyond float32"):
        backend.inverse_norms(np.array([[1e-45, 0]], F32))


def _negative_zero_lanes():
    """Products that round to -0.0 in every lane: a zero-padded tail would turn them +0.0."""
    return np.full((1, 17), -TINY, dtype=F32), np.full((2, 17), 0.5, dtype=F32)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_scores_keep_negative_zero_lanes(backend):
    stored, weights = _negative_zero_lanes()
    want = _reference.scores(stored, None, weights, -0.0)  # -0.0 + -0.0 keeps the sign
    assert_bits_equal(backend.scores(stored, None, weights, -0.0), want)
    assert want.view(np.uint32)[0, 0] == 0x80000000  # the case keeps a -0.0


# --- meta-tests: the cases see other arithmetic ----------------------------------------


def _cases_fail(fn, op):
    """Whether some case of the suite gives other bits than the reference for ``fn``."""
    for dtype in DTYPES:
        for dim in DIMS:
            stored, scales = _stored(dtype, dim)
            inverse = _reference.inverse_norms(stored)
            queries = _queries(dim, 3)
            if op == "inverse_norms":
                got, want = fn(stored), _reference.inverse_norms(stored)
            elif op == "cosines":
                got, want = (
                    fn(stored, inverse, queries),
                    _reference.cosines(stored, inverse, queries),
                )
            else:
                got, want = fn(stored, scales, queries), _reference.scores(stored, scales, queries)
            if not np.array_equal(np.asarray(got, F32).view(np.uint32), want.view(np.uint32)):
                return True
    return False


def _decoded(stored, scales=None):
    return _reference.dequantize(stored, None).astype(F32)


def test_blas_products_are_not_the_contract():
    def blas(stored, inverse, queries):
        units = _reference.postprocess(queries, [])
        units = units / np.sqrt((units * units).sum(axis=1, dtype=F32))[:, None]
        return (_decoded(stored) @ units.T.astype(F32)) * inverse[:, None]

    assert _cases_fail(blas, "cosines")


def test_a_zero_padded_tail_is_not_the_contract():
    def padded(stored, scales, weights):
        x = _decoded(stored)
        pad = (-x.shape[1]) % 16
        x = np.pad(x, ((0, 0), (0, pad)))
        w = np.pad(weights, ((0, 0), (0, pad)))
        lanes = np.zeros((len(x), len(w), 16), F32)
        for i in range(0, x.shape[1], 16):
            for j in range(16):
                lanes[:, :, j] = _reference.fma32(
                    x[:, None, i + j], w[None, :, i + j], lanes[:, :, j]
                )
        for width in (8, 4, 2, 1):
            lanes[:, :, :width] = lanes[:, :, :width] + lanes[:, :, width : 2 * width]
        return lanes[:, :, 0] + F32(-0.0)

    stored, weights = _negative_zero_lanes()
    got = padded(stored, None, weights)
    assert not np.array_equal(
        got.view(np.uint32), _reference.scores(stored, None, weights, -0.0).view(np.uint32)
    )


def test_a_float32_sum_of_squares_is_not_the_contract():
    def float32_sum(stored):
        x = _decoded(stored)
        squares = (x * x).sum(axis=1, dtype=F32)
        with np.errstate(divide="ignore"):
            return np.where(squares > 0, F32(1) / np.sqrt(squares), F32(0)).astype(F32)

    assert _cases_fail(float32_sum, "inverse_norms")


def test_a_fused_scale_and_bias_is_not_the_contract():
    def fused(stored, scales, weights):
        dots = np.stack([_reference.dot16(_decoded(stored), w) for w in weights], axis=1)
        if scales is None:
            return dots
        return _reference.fma32(dots, scales[:, None], np.full_like(dots, F32(0.1)))

    def contract(stored, scales, weights):
        return _reference.scores(stored, scales, weights, F32(0.1))

    # The fused form differs from the contract (two roundings) on some int8 case.
    for dim in DIMS:
        stored, scales = _stored("int8", dim)
        queries = _queries(dim, 3)
        got, want = fused(stored, scales, queries), contract(stored, scales, queries)
        if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
            return
    pytest.fail("no case tells a fused scale and bias from the contract")
