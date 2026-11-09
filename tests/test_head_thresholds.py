"""Thresholds preserve class identity and validate values before rounding."""

import numpy as np
import pytest

from fse._head._thresholds import thresholds


@pytest.mark.parametrize("value", [None, 0.5, [0.5, 0.5], {}, {"sport": 0.5}])
def test_default_and_named_thresholds(value):
    got = thresholds(value, ("sport", "news"))
    np.testing.assert_array_equal(got, np.array([0.5, 0.5], np.float32))
    assert got.dtype == np.float32
    assert not got.flags.writeable


def test_partial_named_thresholds_use_class_order_and_own_the_data():
    given = {"news": 0.3, "sport": 0.4}
    got = thresholds(given, ("sport", "news", "other"))
    given["news"] = 0.8
    np.testing.assert_array_equal(got, np.array([0.4, 0.3, 0.5], np.float32))
    np.testing.assert_array_equal(thresholds({7: 0.2}, (3, 7)), np.array([0.5, 0.2], np.float32))


@pytest.mark.parametrize(
    ("given", "error", "match"),
    [
        ({"missing": 0.4}, ValueError, "unknown threshold label"),
        ({0: 0.4}, ValueError, "unknown threshold label"),
        ({True: 0.4}, TypeError, "not a bool"),
        (True, TypeError, "number"),
        ([True, 0.3], TypeError, "number"),
        ({"sport": "0.3"}, TypeError, "number"),
        ([0.5], ValueError, "2 values"),
        ([[0.5, 0.5]], ValueError, "shape"),
        (float("nan"), ValueError, "finite"),
        (float("inf"), ValueError, "finite"),
        (-1e-100, ValueError, "finite"),
        (1 + 2**-30, ValueError, "finite"),
    ],
)
def test_invalid_thresholds_are_not_coerced(given, error, match):
    with pytest.raises(error, match=match):
        thresholds(given, ("sport", "news"))


def test_updated_keeps_what_a_partial_mapping_does_not_name():
    from fse._head._thresholds import updated

    labels = ("a", "b", "c")
    current = {"a": 0.1, "b": 0.2, "c": 0.3}
    assert updated(current, {"b": 0.9}, labels).tolist() == pytest.approx([0.1, 0.9, 0.3])
    assert updated(current, 0.4, labels).tolist() == pytest.approx([0.4] * 3)
    assert updated(current, [0.6, 0.7, 0.8], labels).tolist() == pytest.approx([0.6, 0.7, 0.8])
    assert updated(current, None, labels).tolist() == [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="unknown threshold label"):
        updated(current, {"z": 0.5}, labels)


@pytest.mark.parametrize("key", [True, np.bool_(True), 1.0])
def test_updated_checks_keys_before_equal_keys_can_merge(key):
    from fse._head._thresholds import updated

    current = {1: 0.1, 2: 0.2}
    with pytest.raises(TypeError, match="a label"):
        updated(current, {key: 0.9}, (1, 2))
    assert current == {1: 0.1, 2: 0.2}


def test_updated_numpy_integer_keys_keep_valid_float32_bits():
    from fse._head._thresholds import updated

    got = updated({1: 0.1, 2: 0.2}, {np.int64(2): 0.9}, (1, 2))
    want = thresholds([0.1, 0.9], (1, 2))
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("mask", [False, [True, False]])
def test_masked_threshold_vectors_are_refused_before_conversion(mask):
    with pytest.raises(TypeError, match="masked"):
        thresholds(np.ma.array([0.3, 0.7], mask=mask), (1, 2))
