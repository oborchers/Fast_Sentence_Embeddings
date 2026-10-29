"""fse._backends._mlx._bounds: the backend's own bounds are at least those of the
contract (support.bound) and its checks before normalize catch zero, tiny and huge rows.
Plain numpy, so these run on every machine."""

import numpy as np
import pytest

from fse._backends._mlx import _bounds
from fse._contract import Normalize, Project, RemoveComponents
from support import bound
from support.cases import random_case

F32 = np.float32


def _basis(dim, k, seed):
    q = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, k)))[0]
    return np.ascontiguousarray(q.T, dtype=F32)


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


def test_no_pooling_bound_for_a_quarter_of_2_24_contributions():
    got = _bounds.pooling(np.ones(2, F32), np.array([1, 1 << 22]), np.array([1, 1 << 22]), 4)
    assert np.isfinite(got[0])
    assert got[1] == np.inf


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


@pytest.mark.parametrize(
    "step",
    [
        RemoveComponents(_basis(12, 2, 1), np.array([1.0, 0.5], F32)),
        RemoveComponents(_basis(12, 3, 2)),
        Project(_basis(12, 5, 3)),
    ],
    ids=["remove weighted", "remove", "project"],
)
def test_a_linear_bound_covers_the_contract_s(step):
    """After pooling then the step: at least the helper's E for the same rows."""
    case = random_case("l", "float32", 5, dim=12, n=30, empty=0)
    rows, pool, _ = bound.bounds(case)
    _, want, _ = bound.bounds(case, (step,))
    got = _bounds.linear(step, 12).propagate(pool, _bounds.norms(rows))
    assert (got >= want).all()


def test_the_linear_bound_of_a_projection_and_its_overflow():
    linear = _bounds.linear(Project(np.array([[0.0, 1.0]], F32)), 2)
    assert linear.norm == pytest.approx(1)
    assert linear.dim == 1
    over = linear.overflows(np.array([0.0, 0.0]), np.array([1.0, 2.0**100]))
    assert over.tolist() == [False, True]
    assert _bounds.linear(Normalize(), 2) is None


def test_after_normalize():
    got = _bounds.after_normalize(np.array([1e-6]), np.array([2.0]), 300)
    assert got[0] >= 2 * 1e-6 / 2
