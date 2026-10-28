"""One shard being written: reserve, commit in order, publish whole, or abort."""

import numpy as np
import pytest

from fse._store._layout import StoreInfo, create_store, shard_name, shards
from fse._store._shard import ShardWriter

INFO = StoreInfo("sha256:" + "cd" * 32, "float16", 4)


@pytest.fixture
def store(tmp_path):
    create_store(tmp_path / "s", INFO)
    return tmp_path / "s"


def _fill(writer, n, value):
    codes, _, norms = writer.reserve(n)
    codes[...] = value
    norms[...] = 0.5
    return codes


def test_a_shard_is_published_whole_and_only_then(store):
    writer = ShardWriter(store, INFO, 0, 8, "test")
    first = _fill(writer, 3, 1)
    second = _fill(writer, 5, 2)
    writer.commit(0, first)
    del first
    assert not writer.can_finish()  # rows 3 to 7 are not committed
    writer.commit(3, second)
    assert not writer.can_finish()  # the view of rows 3 to 7 is still held
    del second
    assert writer.can_finish()
    assert not shards(store, INFO)
    assert writer.finish() == 8
    ((shard, tensors),) = shards(store, INFO)
    assert (shard.first_row, shard.rows) == (0, 8)
    np.testing.assert_array_equal(tensors["vectors"][:3], 1)
    np.testing.assert_array_equal(tensors["vectors"][3:], 2)


def test_a_sealed_shard_ends_at_its_reserved_rows(store):
    writer = ShardWriter(store, INFO, 0, 100, "test")
    writer.commit(0, _fill(writer, 7, 3))
    writer.seal()
    assert writer.full
    assert writer.can_finish()
    assert writer.finish() == 7
    assert shards(store, INFO)[0][0].rows == 7


def test_commits_follow_the_reservations(store):
    writer = ShardWriter(store, INFO, 0, 8, "test")
    a = _fill(writer, 2, 1)
    b = _fill(writer, 2, 1)
    with pytest.raises(
        ValueError, match="rows from 2 of the shard committed, expected rows from 0"
    ):
        writer.commit(2, b)
    with pytest.raises(ValueError, match="not the target reserved"):
        writer.commit(0, np.ones((2, 4), np.float16))
    with pytest.raises(ValueError, match="beyond the 8 rows"):
        writer.reserve(5)
    writer.commit(0, a)
    with pytest.raises(RuntimeError, match="2 reserved rows were not committed"):
        writer.finish()
    writer.abort()
    assert not list(store.glob(".shard-*"))


def test_dropping_the_uncommitted_rows_keeps_the_committed_ones(store):
    writer = ShardWriter(store, INFO, 0, 10, "test")
    writer.commit(0, _fill(writer, 4, 5))
    _fill(writer, 3, 6)  # reserved, never committed (a task that failed)
    writer.drop_uncommitted()
    assert writer.can_finish()
    assert writer.finish() == 4
    np.testing.assert_array_equal(shards(store, INFO)[0][1]["vectors"], 5)


def test_a_published_name_is_never_replaced(store):
    first = ShardWriter(store, INFO, 0, 2, "test")
    second = ShardWriter(store, INFO, 0, 2, "test")
    first.commit(0, _fill(first, 2, 1))
    second.commit(0, _fill(second, 2, 2))
    first.finish()
    with pytest.raises(FileExistsError):
        second.finish()
    second.abort()
    np.testing.assert_array_equal(shards(store, INFO)[0][1]["vectors"], 1)
    assert sorted(p.name for p in store.iterdir()) == [shard_name(0), "store.json"]
