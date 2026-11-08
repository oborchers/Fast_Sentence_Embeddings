"""The bits column of a shard being written: derived from the committed codes, published for
the shard as it is on disk."""

import numpy as np
import pytest

from fse._store._bits_format import bits_name, open_bits
from fse._store._bits_writer import BitsColumn
from fse._store._layout import read_store
from fse._store._shard import ShardBase
from support.bits import ENCODER, expected_bits, signed_store, tree


def column(path, info, capacity):
    return BitsColumn(path, info, 0, capacity, "test")


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_the_commits_in_order_give_the_bits_of_the_rows(tmp_path, dtype):
    path = tmp_path / "store"
    info, rows_of = signed_store(path, dtype=dtype, sizes=(9,))
    bits = column(path, info, 9)
    vectors = rows_of[0]
    for begin, end in ((0, 4), (4, 5), (5, 9)):
        bits.commit(begin, vectors[begin:end])
    assert np.array_equal(bits.bits, expected_bits(vectors))
    assert bits.publish(9, synced=True) is True
    found = open_bits(path, info, _shard(path, info), verify=True)
    assert np.array_equal(found[0], expected_bits(vectors))


def _shard(path, info):
    from fse._store._layout import open_shard

    return open_shard(path, info, 0)[0]


def test_a_shard_that_ends_early_publishes_the_bits_of_its_rows(tmp_path):
    path = tmp_path / "store"
    info, rows_of = signed_store(path, sizes=(5,))
    bits = column(path, info, 16)  # the capacity is larger than the rows written
    bits.commit(0, rows_of[0])
    assert bits.publish(5, synced=True) is True
    assert open_bits(path, info, _shard(path, info), verify=True)[0].shape == (5, 2)


def test_the_bits_are_bound_to_the_shard_on_disk_and_the_row_count_must_agree(tmp_path):
    path = tmp_path / "store"
    info, rows_of = signed_store(path, sizes=(5,))
    bits = column(path, info, 5)
    bits.commit(0, rows_of[0])
    with pytest.raises(RuntimeError, match="holds 5 rows, not the 4 written"):
        bits.publish(4, synced=True)
    assert not list(path.glob("*sign_bits-*"))


def test_a_valid_artifact_there_first_is_kept(tmp_path):
    path = tmp_path / "store"
    info, rows_of = signed_store(path, sizes=(5,))
    first = column(path, info, 5)
    first.commit(0, rows_of[0])
    second = BitsColumn(path, info, 0, 5, "other")
    second.commit(0, rows_of[0])
    assert first.publish(5, synced=True) is True
    kept = tree(path / bits_name(0))
    assert second.publish(5, synced=True) is False
    assert tree(path / bits_name(0)) == kept


def test_release_frees_the_bits(tmp_path):
    path = tmp_path / "store"
    info, _ = signed_store(path, sizes=(5,))
    bits = column(path, info, 5)
    bits.release()
    assert bits.bits is None


def test_a_shard_of_another_kind_publishes_no_extras():
    assert ShardBase.publish_extras(object.__new__(ShardBase)) is False


def test_the_encoder_of_the_test_stores_is_the_one_the_artifact_is_bound_to(tmp_path):
    path = tmp_path / "store"
    info, _ = signed_store(path, sizes=(3,))
    assert read_store(path).encoder == info.encoder == ENCODER.fingerprint
