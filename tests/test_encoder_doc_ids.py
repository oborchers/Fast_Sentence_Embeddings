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
        it = encoder.encode_batches(parts, sink=sink, doc_ids=True)
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
        list(encoder.encode_batches(parts, sink=sink, doc_ids=True))
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert [s.rows for s, _ in store._shards] == [7, 7, 6]
        assert np.array_equal(column(store, "doc_ids"), ids)


def _poisoned():
    yield from ()
    raise AssertionError("a batch was read")


def test_the_mode_is_checked_when_called_before_any_batch(tmp_path):
    encoder = finished()
    with fse.ShardSink(tmp_path / "ids", encoder, doc_ids=True) as ids_sink:
        with pytest.raises(ValueError, match="pass doc_ids=True"):
            encoder.encode_batches(_poisoned(), sink=ids_sink)
        with pytest.raises(TypeError, match="bool"):
            encoder.encode_batches(_poisoned(), sink=ids_sink, doc_ids=1)
        with pytest.raises(TypeError, match="bool"):
            encoder.encode_batches(_poisoned(), sink=ids_sink, doc_ids=None)
    with (
        fse.ShardSink(tmp_path / "plain", encoder) as plain_sink,
        pytest.raises(ValueError, match="needs a sink with doc_ids"),
    ):
        encoder.encode_batches(_poisoned(), sink=plain_sink, doc_ids=True)
    with pytest.raises(ValueError, match="needs a sink with doc_ids"):
        encoder.encode_batches(_poisoned(), doc_ids=True)


def test_a_pair_of_strings_is_two_sentences_without_the_mode():
    batches = list(finished().encode_batches([("w1", "w2")]))
    assert batches[0].rows.shape[0] == 2


class _IdsRecorder(fse.MemorySink):
    has_doc_ids = True

    def __init__(self, dim):
        super().__init__(dim)
        self.seen = []

    def commit(self, first_row, rows, doc_ids=None):
        self.seen.append(doc_ids)
        super().commit(first_row, rows)


def _flagged(value):
    class Flagged(fse.MemorySink):
        has_doc_ids = value

    return Flagged


def test_commit_gets_read_only_int64_ids_of_the_commit_length():
    encoder = finished()
    sink = _IdsRecorder(encoder.output_dim)
    encoder.encode(["w1", "w2", "w3"], sink=sink, doc_ids=[4, 5, 6])
    ids = np.concatenate(sink.seen)
    assert list(ids) == [4, 5, 6]
    assert all(s.dtype == np.int64 and not s.flags.writeable for s in sink.seen)


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_other_values_of_has_doc_ids_keep_the_2_0_behavior_without_ids(value):
    encoder = finished()
    texts = ["w1", "w2"]
    plain = fse.MemorySink(encoder.output_dim)
    encoder.encode(texts, sink=plain)
    sink = _flagged(value)(encoder.output_dim)
    encoder.encode(texts, sink=sink)
    assert np.array_equal(sink.result().view("u4"), plain.result().view("u4"))
    batched = _flagged(value)(encoder.output_dim)
    assert len(list(encoder.encode_batches([texts], sink=batched))) == 1
    assert np.array_equal(batched.result().view("u4"), plain.result().view("u4"))


@pytest.mark.parametrize("value", [1, "yes", None])
def test_ids_need_a_sink_with_exactly_true(value):
    encoder = finished()
    sink = _flagged(value)(encoder.output_dim)
    with pytest.raises(TypeError, match="must be True"):
        encoder.encode(["w1"], sink=sink, doc_ids=[1])
    with pytest.raises(TypeError, match="must be True"):
        encoder.encode_batches(_poisoned(), sink=sink, doc_ids=True)


def test_encode_refuses_a_bool_for_the_ids(tmp_path):
    encoder = finished()
    with (
        fse.ShardSink(tmp_path / "s", encoder, doc_ids=True) as sink,
        pytest.raises(TypeError, match="not a bool"),
    ):
        encoder.encode(["w1"], sink=sink, doc_ids=True)
    with pytest.raises(TypeError, match="not a bool"):
        encoder.encode(["w1"], doc_ids=True)
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert len(store) == 0
    with pytest.raises(ValueError, match="doc_ids"):
        encoder.encode(["w1"], doc_ids=[1])
