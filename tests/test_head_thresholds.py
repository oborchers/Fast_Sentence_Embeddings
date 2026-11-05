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
