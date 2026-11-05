"""The float64 arithmetic of the import of a head: checks, and the cpu backend against numpy,
also under a foreign floating-point environment."""

import numpy as np
import pytest

from fse import _native
from fse._backends import get_backend
from fse._backends._base_head import check_affine64_input, check_quotient64_input
from support.head_cases import BACKENDS

SUBNORMAL = np.nextafter(0.0, 1.0)  # 5e-324, flushed by DAZ


def _head():
    weight = np.array([[1.5, -2.25, SUBNORMAL], [0.1, 0.2, 0.3]])
    return weight, np.array([0.5, -SUBNORMAL]), np.array([-1.7, 0.3]), np.array([-0.2, 4.0])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
def test_affine_and_quotient_are_the_numpy_expressions(backend):
    name, isa = backend
    host = get_backend(name, isa=isa)
    weight, bias, factor, shift = _head()
    got_w, got_b = host.affine64(weight, bias, factor, shift)
    np.testing.assert_array_equal(got_w.view(np.uint64), (factor[:, None] * weight).view(np.uint64))
    np.testing.assert_array_equal(got_b.view(np.uint64), (factor * bias + shift).view(np.uint64))
    q_w, q_b = host.quotient64(weight, bias, 1.7)
    np.testing.assert_array_equal(q_w.view(np.uint64), (weight / 1.7).view(np.uint64))
    np.testing.assert_array_equal(q_b.view(np.uint64), (bias / 1.7).view(np.uint64))
    assert got_w.dtype == got_b.dtype == q_w.dtype == q_b.dtype == np.float64


@pytest.mark.parametrize("isa", _native.available_isas())
@pytest.mark.parametrize(
    "env", [{"flush": True}, {"round_up": True}, {"flush": True, "round_up": True}]
)
def test_the_cpu_backend_ignores_and_restores_a_foreign_environment(isa, env):
    host = get_backend("cpu", isa=isa)
    weight, bias, factor, shift = _head()
    want = (factor[:, None] * weight, factor * bias + shift)
    want_q = (weight / 1.7, bias / 1.7)
    saved = _native._test_fpenv_set(**env)
    try:
        before = _native._test_fpenv_get()
        got = host.affine64(weight, bias, factor, shift)
        got_q = host.quotient64(weight, bias, 1.7)
        assert _native._test_fpenv_get() == before
    finally:
        _native._test_fpenv_restore(saved)
    for have, expect in zip((*got, *got_q), (*want, *want_q), strict=True):
        np.testing.assert_array_equal(have.view(np.uint64), expect.view(np.uint64))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
def test_overflow_gives_infinity_without_a_warning(backend):
    name, isa = backend
    weight = np.full((1, 2), 1e308)
    got_w, got_b = get_backend(name, isa=isa).affine64(
        weight, np.zeros(1), np.array([10.0]), np.zeros(1)
    )
    assert np.isinf(got_w).all()
    assert got_b.tolist() == [0.0]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ("weight_dtype", "weight must be a float64 matrix"),
        ("weight_ndim", "weight must be a float64 matrix"),
        ("weight_empty", "must not be empty"),
        ("bias", "bias must be a float64 vector of 2 values"),
        ("factor", "factor must be a float64 vector of 2 values"),
        ("shift", "shift must be a float64 vector of 2 values"),
    ],
)
def test_affine_inputs_are_checked(backend, bad, match):
    name, isa = backend
    weight, bias, factor, shift = _head()
    if bad == "weight_dtype":
        weight = weight.astype(np.float32)
    elif bad == "weight_ndim":
        weight = weight[0]
    elif bad == "weight_empty":
        weight = np.empty((0, 3))
    elif bad == "bias":
        bias = bias[:1]
    elif bad == "factor":
        factor = factor.astype(np.float32)
    else:
        shift = shift[None, :]
    with pytest.raises(ValueError, match=match):
        get_backend(name, isa=isa).affine64(weight, bias, factor, shift)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda item: item[1] or item[0])
@pytest.mark.parametrize("divisor", [0.0, -1.0, float("inf"), float("nan"), True, 2, "2", None])
def test_quotient_divisor_is_a_finite_positive_float(backend, divisor):
    name, isa = backend
    weight, bias, _, _ = _head()
    with pytest.raises(ValueError, match="divisor"):
        get_backend(name, isa=isa).quotient64(weight, bias, divisor)


def test_quotient_bias_and_weight_are_checked():
    weight, bias, _, _ = _head()
    with pytest.raises(ValueError, match="bias must be a float64 vector of 2 values"):
        check_quotient64_input(weight, bias[:1], 2.0)
    with pytest.raises(ValueError, match="weight"):
        check_quotient64_input(weight.astype(np.float32), bias, 2.0)
    check_affine64_input(weight, bias, bias, bias)
