"""Sorted stored-row reads preserve bits and validate before touching tensors."""

from types import SimpleNamespace

import numpy as np
import pytest

from fse._store import _take
from support.take_store import critical_rows


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("name", ["vectors", "inv_norms", "doc_ids"])
@pytest.mark.parametrize("wanted", [[], [0], [12], [0, 2, 5, 6, 12]])
def test_sorted_gather_matches_general_gather(dtype, name, wanted, monkeypatch):
    codes, _ = critical_rows(13, dtype)
    columns = {
        "vectors": codes,
        "inv_norms": np.arange(13, dtype=np.float32),
        "doc_ids": np.arange(13, dtype=np.int64) * -17,
    }
    shards = [
        (SimpleNamespace(first_row=lo), {k: v[lo:hi] for k, v in columns.items()})
        for lo, hi in ((0, 3), (3, 6), (6, 13))
    ]
    wanted = np.array(wanted, np.int64)
    column = columns[name]
    monkeypatch.setattr(_take, "_BLOCK_BYTES", 10)
    got = _take.gather_sorted(shards, wanted, 13, name)
    expected = _take.gather(
        shards,
        wanted,
        13,
        lambda tensors, local: tensors[name][local],
        tail=column.shape[1:],
        dtype=column.dtype,
        row_bytes=10,
    )
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize(("wanted", "total"), [([-1, 2], 3), ([0, 3], 3), ([0], 0)])
def test_invalid_positions_fail_before_tensor_access(wanted, total):
    class Refuse(dict):
        def __getitem__(self, _key):
            raise AssertionError("tensor accessed before validation")

    shards = [(SimpleNamespace(first_row=0), Refuse())]
    with pytest.raises(ValueError, match="rows"):
        _take.gather_sorted(shards, np.array(wanted, np.int64), total, "vectors")
