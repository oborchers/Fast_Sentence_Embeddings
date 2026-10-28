"""The conformance suite for dot16, the postprocessing steps and score.

Every case runs for every backend in BACKENDS and must give the bits of fse._reference;
tests/test_conformance.py holds the weighted mean cases and the meta-tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from fse import _reference
from fse._contract import Normalize, Project, RemoveComponents
from support.conformance import BACKENDS, BIG, F32, _rng, assert_bits_equal


def _basis(dim, k, seed):
    q = np.linalg.qr(_rng(seed).standard_normal((dim, k)))[0]
    return np.ascontiguousarray(q.T, dtype=F32)


def _step_lists(dim):
    k = min(3, dim)
    removal = RemoveComponents(_basis(dim, k, dim), np.array([1, 0.5, 0.25], F32)[:k])
    plain = RemoveComponents(_basis(dim, 1, dim + 1))
    project = Project(_basis(dim, max(1, dim // 2), dim + 2))
    return {
        "none": [],
        "normalize": [Normalize()],
        "remove weighted": [removal],
        "remove one": [plain],
        "project": [project],
        "all three": [removal, project, Normalize()],
    }


STEP_CASES = [
    (dim, name) for dim in (1, 3, 4, 7, 15, 16, 17, 31, 33, 300) for name in _step_lists(dim)
]


def _step_rows(dim):
    rows = _rng(dim).standard_normal((12, dim)).astype(F32)
    rows[3] = 0  # zero rows stay zero through normalize
    rows[5] *= F32(1e-20)
    return rows


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize(("dim", "steps"), STEP_CASES, ids=[f"{d}-{n}" for d, n in STEP_CASES])
def test_steps(backend, dim, steps):
    rows = _step_rows(dim)
    lists = _step_lists(dim)
    want = _reference.postprocess(rows, lists[steps])
    assert_bits_equal(backend.postprocess(rows, lists[steps]), want, f"{dim} {steps}")


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_step_errors_like_the_reference(backend):
    rows = np.array([[1, 1], [1e-30, 0], [3e38, 3e38]], dtype=F32)
    with pytest.raises(ValueError, match="row 11 cannot be normalized"):
        backend.postprocess(rows, [Normalize()], first_row=10)
    big = np.array([[1, 0], [np.finfo(F32).max, 0]], dtype=F32)
    step = RemoveComponents(np.array([[1, 0]], dtype=F32), np.array([-2], dtype=F32))
    with pytest.raises(ValueError, match="row 8 contains NaN or infinite"):
        backend.postprocess(big, [step], first_row=7)
    with pytest.raises(ValueError, match="row 4 contains NaN"):
        backend.postprocess(np.array([[1, 0], [np.inf, 0]], dtype=F32), [], first_row=3)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_dot16_keeps_negative_zero_lanes(backend):
    rows = np.full((1, 17), -(2.0**-149), dtype=F32)
    vector = np.full(17, 0.5, dtype=F32)
    assert_bits_equal(backend.dot16(rows, vector)[None, :], _reference.dot16(rows, vector)[None, :])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dim", [1, 7, 16, 17, 300])
def test_score(backend, dtype, dim):
    rng = _rng(dim)
    stored, scales = _reference.quantize_rows(rng.standard_normal((20, dim)).astype(F32), dtype)
    weights = rng.standard_normal(dim).astype(F32)
    want = _reference.score(stored, scales, weights, 0.25)
    assert_bits_equal(backend.score(stored, scales, weights, 0.25)[None, :], want[None, :])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_score_overflow_raises(backend):
    with pytest.raises(ValueError, match="score of row 0 overflows"):
        backend.score(np.array([[BIG, 0]], dtype=F32), None, np.array([2, 0], F32), 0.0)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_a_bias_beyond_float32_raises_the_overflow(backend):
    with pytest.raises(ValueError, match="score of row 0 overflows"):
        backend.score(np.zeros((1, 2), F32), None, np.ones(2, F32), 1e100)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_score_weights_beyond_float32_raise_the_overflow(backend):
    with pytest.raises(ValueError, match="score of row 0 overflows"):
        backend.score(np.ones((1, 2), F32), None, np.array([1e100, 0.0]), 0.0)
