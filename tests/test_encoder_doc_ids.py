"""ID batches fill shards across input boundaries and reject unaligned columns."""

import json

import numpy as np
import pytest

import fse
from support.document_ids import column, write_store
from support.encoder import finished


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_parts_fill_shards_and_append(tmp_path, dtype):
    ids = [5, 5, -1, 10, 10, 3, 3, 3, 0, 7, 8]
    encoder = write_store(tmp_path / "s", ids, dtype=dtype, shard_rows=4, batches=True)
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert [s.rows for s, _ in store._shards] == [4, 4, 3]
        assert list(column(store, "doc_ids")) == ids
        store.verify()
    with fse.ShardSink(tmp_path / "s", encoder, dtype=dtype, doc_ids=True) as sink:
        assert sink.first_row == len(ids)
        encoder.encode(["w1"], doc_ids=[-(2**63)], sink=sink)
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert list(column(store, "doc_ids")) == [*ids, -(2**63)]
    header = json.loads((tmp_path / "s" / "store.json").read_text())
    assert header["store_version"] == 2
    assert header["metadata"]["fse.format_version"] == "2"
    with pytest.raises(ValueError, match="has_doc_ids"):
        fse.ShardSink(tmp_path / "s", encoder, dtype=dtype)


def test_lengths_modes_and_batch_errors(tmp_path):
    encoder = finished()
    with pytest.raises(ValueError, match="doc_ids"):
        encoder.encode(["w1"], doc_ids=[1])
    with fse.ShardSink(tmp_path / "s", encoder, doc_ids=True, shard_rows=8) as sink:
        parts = [(["w1", "w2"], [1, 2]), (["w3"], [])]
        it = encoder.encode_batches(parts, sink=sink)
        assert next(it).durable_rows == 0
        with pytest.raises(ValueError, match="doc_ids has 0 rows for 1 texts"):
            next(it)
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert list(column(store, "doc_ids")) == [1, 2]


def test_parquet_recipe(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "input.parquet"
    ids = np.arange(20, dtype="int64") // 3
    pq.write_table(pa.table({"text": ["w1"] * 20, "ticket_id": ids}), source)
    encoder = finished()
    with (
        pq.ParquetFile(source) as parquet,
        fse.ShardSink(tmp_path / "s", encoder, doc_ids=True, shard_rows=7) as sink,
    ):
        parts = ((b["text"], b["ticket_id"]) for b in parquet.iter_batches(batch_size=3))
        list(encoder.encode_batches(parts, sink=sink))
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert [s.rows for s, _ in store._shards] == [7, 7, 6]
        assert np.array_equal(column(store, "doc_ids"), ids)
