"""The runtime rejects invalid worker, row and sink inputs before conversion."""

import numpy as np
import pytest

from fse._runtime._checks import _non_negative, check_sink_rows, check_workers
from fse._runtime._pool import Workers
from fse._runtime._sinks import ArrayRows


@pytest.mark.parametrize("value", [True, -1, 1.5, "2", {}, set()])
def test_invalid_counts_fail_in_the_original_form(value):
    with pytest.raises(ValueError, match="positive int"):
        check_workers(value)
    with pytest.raises(ValueError, match="non-negative int"):
        _non_negative(value, "first_row")


def test_valid_counts_and_workers_keep_their_meaning():
    assert check_workers(None) is None
    assert check_workers(np.int64(3)) == 3
    assert check_workers(Workers(2, None)) == 2
    assert _non_negative(np.int64(0), "first_row") == 0
    with pytest.raises(ValueError, match="positive int"):
        check_workers(0)


def test_sink_shape_is_checked():
    check_sink_rows(ArrayRows(2, 3), 3)
    with pytest.raises(ValueError, match="rows of 4 values"):
        check_sink_rows(ArrayRows(2, 3), 4)
    with pytest.raises(TypeError, match="target and commit"):
        check_sink_rows(object(), 3)
