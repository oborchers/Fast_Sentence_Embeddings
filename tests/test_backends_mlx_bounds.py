"""fse._backends._mlx._bounds: the backend's own bounds are at least those of the
contract (support.bound) and its checks before normalize catch zero, tiny and huge rows.
Plain numpy, so these run on every machine."""

import numpy as np
import pytest

from fse._backends._mlx import _bounds
from support import bound
from support.cases import random_case

F32 = np.float32


def _magnitude(case):
    """What the kernel sums per sentence: |f| times the row's norm rounded up, float32."""
    rows = case.arrays["rows"].astype(np.float64)
    norms = np.linalg.norm(rows, axis=1).astype(F32) * F32(1 + 2**-22)
    out = []
    for sentence in case.sentences:
        m = F32(0)
        for t in sentence:
            factor = abs(
                F32(case.weights[t])
                * (F32(1) if case.arrays["scales"] is None else case.arrays["scales"][t])
            )
            m = F32(m + F32(factor) * norms[t])
        out.append(m)
    return np.array(out, F32)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_the_pooling_bound_covers_the_contract_s(dtype):
    case = random_case("b", dtype, 3, dim=9, n=40, empty=0.2)
    n = np.array([len(s) for s in case.sentences])
    k = bound.contributions(case)
    got = _bounds.pooling(_magnitude(case), n, k, 9)
    _, want, exact = bound.bounds(case)
    assert (got[~exact] >= want[~exact]).all()
    assert (got[n == 0] == 0).all()


def test_the_pooling_bound_is_finite_below_2_24_contributions():
    k = np.array([1, (1 << 24) - 1])
    got = _bounds.pooling(np.ones(2, F32), k, k, 4)
    assert np.isfinite(got).all()
    assert got[1] > 1  # gamma_K about 2**24: no useful bound, before normalize the host


def test_rows_whose_sums_could_overflow_on_the_cpu():
    """R4 of review 3: M bounds every partial sum; from 2**127 (or not finite) the host."""
    sums = np.array([1.0, 2.0**126, 2.0**127, np.inf, np.nan], F32)
    got = _bounds.overflows(sums, np.full(5, 4))
    assert got.tolist() == [False, False, True, True, True]


def test_a_nonfinite_bound_sends_the_row_to_the_host():
    """R1 of review 3: a NaN bound made every comparison false."""
    g = np.array([1.0, 1.0])
    assert _bounds.unnormalizable(g, g, np.array([np.nan, np.inf])).tolist() == [True, True]
    rows = np.ones((2, 3), F32)
    assert _bounds.beyond_float16(rows, np.array([np.nan, 0.0])).tolist() == [True, False]


def test_rows_near_the_float16_range():
    rows = np.array([[65504.0, 0], [-65000.0, 1], [65000.0, 0]], F32)
    got = _bounds.beyond_float16(rows, np.array([0.0, 600.0, 1.0]))
    assert got.tolist() == [True, True, False]


@pytest.mark.parametrize("dim", [1, 16, 300])
def test_device_norms_bracket_the_norm(dim):
    rng = np.random.default_rng(dim)
    rows = rng.standard_normal((50, dim)).astype(F32)
    squares = np.einsum("ij,ij->i", rows, rows).astype(F32)  # another rounding than dot16
    low, high = _bounds.device_norms(squares, dim)
    exact = _bounds.norms(rows)
    assert (low <= exact).all()
    assert (exact <= high).all()
    low, high = _bounds.device_norms(np.array([np.inf, np.nan], F32), dim)
    assert low.tolist() == [0, 0]
    assert high.tolist() == [np.inf, np.inf]


def test_the_rows_normalize_must_not_take_from_the_device():
    g = np.array([1.0, 1e-6, 2.0**-51, 1.0, 2.0**61, 0.0])
    e = np.array([1e-5, 1e-6, 1e-30, 0.4, 1.0, 0.0])
    assert _bounds.unnormalizable(g, g, e).tolist() == [False, True, True, False, True, True]


def test_after_normalize():
    got = _bounds.after_normalize(np.array([1e-6]), np.array([2.0]), 300)
    assert got[0] >= 2 * 1e-6 / 2
