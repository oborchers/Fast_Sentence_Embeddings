"""Reading document IDs back from a store: ``has_doc_ids`` and ``get_doc_ids``."""

import numpy as np
import pytest

import fse
from support.document_ids import write_store
from support.encoder import finished

IDS = [5, 3, 3, 9, -(2**63), 7, 2**63 - 1, 0, 3, 4]


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_all_ids_in_order(tmp_path, dtype):
    write_store(tmp_path / "s", IDS, dtype=dtype, shard_rows=4)
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert store.has_doc_ids
        out = store.get_doc_ids()
        assert out.dtype == np.int64
        assert out.tolist() == IDS


def test_ranges_across_shards_and_empty(tmp_path):
    write_store(tmp_path / "s", IDS, shard_rows=4)
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert store.get_doc_ids(start=2, stop=9).tolist() == IDS[2:9]
        assert store.get_doc_ids(start=4, stop=8).tolist() == IDS[4:8]
        assert store.get_doc_ids(start=3).tolist() == IDS[3:]
        for point in (0, 5, len(IDS)):
            empty = store.get_doc_ids(start=point, stop=point)
            assert empty.dtype == np.int64 and len(empty) == 0
        with pytest.raises(ValueError, match="not within"):
            store.get_doc_ids(start=5, stop=3)
        with pytest.raises(ValueError, match="not within"):
            store.get_doc_ids(stop=len(IDS) + 1)
        with pytest.raises(ValueError):
            store.get_doc_ids(start=-1)


def test_result_is_a_copy_that_outlives_the_store(tmp_path):
    write_store(tmp_path / "s", IDS, shard_rows=4)
    store = fse.VectorStore.open(tmp_path / "s")
    out = store.get_doc_ids(start=1, stop=3)
    store.close()
    assert out.tolist() == IDS[1:3]
    out[:] = 0
    with fse.VectorStore.open(tmp_path / "s") as again:
        assert again.get_doc_ids(start=1, stop=3).tolist() == IDS[1:3]


def test_store_without_ids_and_closed_store(tmp_path):
    encoder = finished(dim=17)
    with fse.ShardSink(tmp_path / "s", encoder) as sink:
        encoder.encode(["w1", "w2"], sink=sink)
    store = fse.VectorStore.open(tmp_path / "s")
    assert not store.has_doc_ids
    with pytest.raises(ValueError, match="this store has no doc_ids"):
        store.get_doc_ids()
    store.close()
    with pytest.raises(RuntimeError):
        store.get_doc_ids()


def test_append_shows_in_a_new_open_not_the_old_snapshot(tmp_path):
    encoder = write_store(tmp_path / "s", IDS[:4], shard_rows=4)
    with fse.VectorStore.open(tmp_path / "s") as old:
        with fse.ShardSink(tmp_path / "s", encoder, doc_ids=True, shard_rows=4) as sink:
            encoder.encode(["w1", "w2"], doc_ids=[11, 12], sink=sink)
        assert old.get_doc_ids().tolist() == IDS[:4]
        with fse.VectorStore.open(tmp_path / "s") as new:
            assert new.get_doc_ids().tolist() == [*IDS[:4], 11, 12]
