"""The folds of Platt scaling and of a softmax temperature, against independent float64
formulas, and the refusals of the temperature."""

import numpy as np
import pytest

from fse._head import _calibration
from fse._head import _content as content

F64 = np.float64


def _rng(seed=5):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("sign", [-1.0, 1.0, 0.0])
def test_platt_folds_to_the_negated_product_bits(sign):
    rng = _rng()
    weight, bias = rng.standard_normal((3, 7)), rng.standard_normal(3)
    a = sign * np.abs(rng.standard_normal(3)) * [1, 10, 1e-3]
    c = rng.standard_normal(3)
    got_w, got_b = _calibration.platt(weight, bias, a, c)
    want_w = np.stack([-(a[j] * weight[j]) for j in range(3)])
    want_b = np.array([-(a[j] * bias[j] + c[j]) for j in range(3)])
    np.testing.assert_array_equal(got_w.view(np.uint64), want_w.view(np.uint64))
    np.testing.assert_array_equal(got_b.view(np.uint64), want_b.view(np.uint64))
    if sign == 0.0:
        assert not got_w.any()
        np.testing.assert_array_equal(got_b, -c + 0.0)


def test_platt_rounds_once_to_float32_in_the_head():
    rng = _rng()
    weight, bias = rng.standard_normal((2, 5)), rng.standard_normal(2)
    a, c = np.array([-1.3, -0.7]), np.array([0.2, -0.1])
    folded_w, folded_b = _calibration.platt(weight, bias, a, c)
    w32, b32 = content.weights(folded_w, folded_b, 5)
    np.testing.assert_array_equal(w32, folded_w.astype(np.float32))
    np.testing.assert_array_equal(b32, folded_b.astype(np.float32))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int8, np.int64])
def test_temperature_divides_exactly_in_float64(dtype):
    weight = (np.arange(12).reshape(3, 4) - 5).astype(dtype)
    bias = np.array([1, -2, 3]).astype(dtype)
    got_w, got_b = _calibration.temperature(weight, bias, 1.7, "softmax")
    np.testing.assert_array_equal(got_w, weight.astype(F64) / 1.7)
    np.testing.assert_array_equal(got_b, bias.astype(F64) / 1.7)
    assert got_w.dtype == got_b.dtype == F64


def test_temperature_takes_vectors_scalars_and_no_bias():
    w, b = _calibration.temperature(np.array([2.0, 4.0]), None, 2.0, "softmax")
    assert w.tolist() == [[1.0, 2.0]]
    assert b.tolist() == [0.0]
    w, b = _calibration.temperature(np.ones((2, 2)), 3, 3.0, "auto")
    assert b.tolist() == [1.0, 1.0]


def test_a_shape_that_is_no_head_is_left_for_the_head_checks():
    weight, bias = _calibration.temperature(np.ones((2, 3, 4)), None, 2.0, "softmax")
    assert weight.shape == (2, 3, 4)
    assert bias is None
    with pytest.raises(ValueError, match="bias must hold 2 values"):
        content.weights(*_calibration.temperature(np.ones((2, 3)), np.ones(3), 2.0, "softmax"), 3)


@pytest.mark.parametrize(
    ("weight", "link"),
    [(np.ones((3, 2)), "sigmoid"), (np.ones((3, 2)), "ovr"), (np.ones((3, 2)), "none"),
     (np.ones((3, 2)), "multilabel"), (np.ones((1, 2)), "auto"), (np.ones(2), "auto")],
)  # fmt: skip
def test_only_the_softmax_takes_a_temperature(weight, link):
    with pytest.raises(
        ValueError, match=r"temperature applies only to the link softmax|sigmoid takes one output"
    ):
        _calibration.temperature(weight, None, 2.0, link)


@pytest.mark.parametrize(
    ("weight", "error", "match"),
    [
        (np.array([[True, False]]), TypeError, "must be numbers"),
        (np.array([["1", "2"]]), TypeError, "must be numbers"),
        (np.array([[2**53 + 1, 1]], np.int64), ValueError, "beyond 2\\*\\*53"),
        (np.array([[-(2**53) - 1, 1]], np.int64), ValueError, "beyond 2\\*\\*53"),
        (np.array([[2**63 + 5, 1]], np.uint64), ValueError, "beyond 2\\*\\*53"),
    ],
)
def test_the_type_is_checked_before_any_conversion(weight, error, match):
    with pytest.raises(error, match=match):
        _calibration.temperature(weight, None, 2.0, "softmax")


@pytest.mark.parametrize(
    ("weight", "error", "match"),
    [
        ([[2**53 + 1, 1.0]], ValueError, "beyond 2\\*\\*53"),
        ([[1.0, -(2**53) - 1]], ValueError, "beyond 2\\*\\*53"),
        ([[True, 1.0]], TypeError, "not bool"),
        ([[np.True_, 1]], TypeError, "not bool"),
    ],
)
def test_mixed_python_lists_cannot_hide_a_large_integer_or_a_bool(weight, error, match):
    with pytest.raises(error, match=match):
        _calibration.temperature(weight, None, 2.0, "softmax")


def test_the_largest_exact_integers_are_accepted():
    weight = np.array([[2**53, -(2**53)]], np.int64)
    got, _ = _calibration.temperature(weight, None, 4.0, "softmax")
    assert got.tolist() == [[2.0**51, -(2.0**51)]]


def test_the_note_names_the_temperature():
    assert (
        _calibration.note(1.7) == "Softmax temperature T = 1.7, folded into the weights and bias."
    )


@pytest.mark.parametrize("field", ["weight", "bias"])
def test_public_temperature_import_refuses_masked_arrays(field):
    import fse
    from support.encoder import finished

    enc = finished(normalize=False)
    values = {"weight": np.ones((2, enc.output_dim)), "bias": np.zeros(2)}
    values[field] = np.ma.array(values[field], mask=False)
    with pytest.raises(TypeError, match="masked"):
        fse.Classifier.from_arrays(
            enc,
            values["weight"],
            values["bias"],
            labels=["a", "b"],
            link="softmax",
            temperature=1.7,
        )
