"""Primitive Arrow IDs and caller-provided NumPy values share one finite interface."""

import numpy as np
import pytest

from fse._text._ids import IdSource


def test_array_snapshot():
    with IdSource([1, -1]) as ids:
        assert len(ids) == 2
        assert list(ids.block(0, 2)) == [1, -1]


def test_arrow_chunk_offsets_and_nulls():
    pa = pytest.importorskip("pyarrow")
    chunks = pa.chunked_array(
        [pa.array([99, -2, -1]).slice(1), pa.array([], type=pa.int64()), pa.array([4, 5])]
    )
    with IdSource(chunks) as ids:
        assert len(ids) == 4
        assert list(ids.block(1, 4)) == [-1, 4, 5]
        assert ids.block(0, 0).dtype == np.int64
    for value in (
        pa.array([1, None]),
        pa.array([1], type=pa.int32()),
        pa.table({"id": [1]}),
        pa.array([1]).dictionary_encode(),
    ):
        with pytest.raises((ValueError, TypeError)):
            IdSource(value)
