"""The columns of a shard: streamed to a file, kept in memory, or constant."""

import numpy as np
import pytest

from fse._container import checksum
from fse._store._columns import ConstantColumn, MemoryColumn, StreamColumn


def test_a_stream_column_writes_at_offsets_and_cuts_to_the_rows(tmp_path):
    column = StreamColumn(tmp_path, "scores", np.dtype("<f4"), (3,), 10)
    first = np.arange(6, dtype="<f4").reshape(2, 3)
    second = np.arange(6, 12, dtype="<f4").reshape(2, 3)
    column.write(2, second)  # positional: the order of the writes is free
    column.write(0, first)
    column.commit(first)
    column.commit(second)
    rows = np.concatenate([first, second])
    assert column.finish(4, synced=True) == ("F32", (4, 3), checksum(rows))
    assert column.fd < 0
    np.testing.assert_array_equal(np.load(tmp_path / "scores.npy"), rows)


def test_a_stream_column_that_was_not_written_refuses_to_finish(tmp_path):
    with pytest.raises(RuntimeError, match="scores were not written"):
        StreamColumn(tmp_path, "scores", np.dtype("<f4"), (), 4).finish(1, synced=False)


def test_an_always_column_exists_without_a_write(tmp_path):
    column = StreamColumn(tmp_path, "vectors", np.dtype("<f4"), (2,), 4, always=True)
    column.finish(0, synced=False)
    assert np.load(tmp_path / "vectors.npy").shape == (0, 2)


def test_closing_a_stream_column_twice_is_harmless(tmp_path):
    column = StreamColumn(tmp_path, "x", np.dtype("<i8"), (), 2)
    column.write(0, np.zeros(2, "<i8"))
    column.close()
    column.close()
    assert column.fd < 0


def test_a_memory_column_hands_out_views_and_writes_the_first_rows(tmp_path):
    column = MemoryColumn(tmp_path, "inv_norms", np.dtype("<f4"), 5)
    column.view(0, 3)[...] = 0.5
    column.view(3, 5)[...] = 2.0
    entry = column.finish(4, synced=False)
    want = np.array([0.5, 0.5, 0.5, 2.0], "<f4")
    assert entry == ("F32", (4,), checksum(want))
    np.testing.assert_array_equal(np.load(tmp_path / "inv_norms.npy"), want)
    assert len(column.array) == 0  # a long stream keeps no arrays of published shards


def test_a_constant_column_is_not_cut_with_the_rows(tmp_path):
    values = np.array([0.25, 0.75], "<f4")
    column = ConstantColumn(tmp_path, "thresholds", values)
    assert column.finish(1, synced=True) == ("F32", (2,), checksum(values))
    np.testing.assert_array_equal(np.load(tmp_path / "thresholds.npy"), values)
