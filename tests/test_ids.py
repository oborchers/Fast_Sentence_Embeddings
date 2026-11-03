"""Strict ID input and owned task snapshots."""

import numpy as np
import pytest

from fse._ids import checked_ids, id_block


@pytest.mark.parametrize(
    "value",
    [
        [True],
        [1.0],
        [None],
        [2**63],
        [-(2**63) - 1],
        np.array([1], dtype="int32"),
        np.array([1], dtype="uint64"),
        "id",
    ],
)
def test_refuse_coercion(value):
    with pytest.raises((TypeError, ValueError)):
        checked_ids(value)


def test_full_domain_and_owned_snapshots():
    values = np.array([-(2**63), -1, 0, 2**63 - 1], dtype="int64")
    assert checked_ids(values) is values
    copy = id_block(values[::-1], 0, 4)
    values[:] = 0
    assert list(copy) == [2**63 - 1, 0, -1, -(2**63)]
    assert not copy.flags.writeable
    with pytest.raises(ValueError, match="required"):
        id_block(values, 0, 5)
