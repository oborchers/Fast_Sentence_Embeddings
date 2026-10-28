"""Encoding into a sink and in batches: encode(sink=) and encode_batches."""

import numpy as np
import pytest

import fse
from fse import _reference
from fse._store._layout import StoreInfo, shards
from support.encoder import unfolded
from support.runtime import texts

ENCODER = unfolded()
F32 = np.float32


def _polls(sizes, seed=0):
    """Finite batches as a queue consumer polls them."""
    rng = np.random.default_rng(seed)
    sentences = texts(sum(sizes), seed=int(rng.integers(100)))
    out, begin = [], 0
    for n in sizes:
        out.append(sentences[begin : begin + n])
        begin += n
    return out, sentences


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_batches_without_a_sink_hold_their_rows(dtype):
    polls, sentences = _polls([7, 0, 130, 1])
    got = list(ENCODER.encode_batches(polls, dtype=dtype, workers=2))
    assert [(b.first_row, b.n_rows, b.durable_rows) for b in got] == [
        (0, 7, None),
        (7, 0, None),
        (7, 130, None),
        (137, 1, None),
    ]
    want = ENCODER.encode(sentences, dtype=dtype)
    if dtype == "int8":
        assert isinstance(got[0].rows, fse.QuantizedRows)
        codes = np.concatenate([b.rows.codes for b in got])
        np.testing.assert_array_equal(codes, want.codes)
    else:
        np.testing.assert_array_equal(
            np.concatenate([b.rows for b in got]).view(np.uint8), want.view(np.uint8)
        )


def test_batches_into_a_shard_sink(tmp_path):
    polls, sentences = _polls([60, 70, 80])
    with fse.ShardSink(tmp_path / "s", ENCODER, dtype="float16", shard_rows=64) as sink:
        got = list(ENCODER.encode_batches(polls, sink=sink))
    assert [(b.first_row, b.n_rows, b.rows) for b in got] == [
        (0, 60, None),
        (60, 70, None),
        (130, 80, None),
    ]
    assert [b.durable_rows for b in got] == [0, 128, 192]  # full shards of 64 only
    assert sink.durable_rows == 210  # the last shard at the end of the stream
    info = StoreInfo(ENCODER.fingerprint, "float16", ENCODER.output_dim)
    codes = np.concatenate([t["vectors"] for _, t in shards(tmp_path / "s", info)])
    np.testing.assert_array_equal(codes, ENCODER.encode(sentences, dtype="float16"))


def test_arguments_are_checked_before_a_batch_is_read():
    read = []

    def polls():
        read.append(True)
        yield ["w1"]

    with pytest.raises(ValueError, match=r"unknown storage dtype|dtype"):
        ENCODER.encode_batches(polls(), dtype="float64")
    with pytest.raises(ValueError, match="backend must be one of"):
        ENCODER.encode_batches(polls(), backend="gpu")
    assert not read
    with pytest.raises(TypeError, match="batch 1 is a str; every batch is a list"):
        list(ENCODER.encode_batches([["w1"], "w2 w3"]))


def test_closing_early_reads_no_further_batch():
    read = []

    def polls():
        for k in range(5):
            read.append(k)
            yield [f"w{k}"]

    batches = ENCODER.encode_batches(polls())
    assert next(batches).n_rows == 1
    batches.close()
    assert read == [0]


class _Database:
    """A vector database: rows arrive by commit, are acknowledged later in bulk."""

    def __init__(self, dim):
        self.dim = dim
        self.points = {}
        self.pending = []
        self.durable_rows = 0

    def target(self, _first_row, n_rows):
        return np.empty((n_rows, self.dim), F32)

    def commit(self, first_row, rows):
        self.pending.append((first_row, rows.copy()))

    def publish(self):  # an upsert with wait=True: acknowledged once it returns
        for first_row, rows in self.pending:
            for i, row in enumerate(rows):
                self.points[first_row + i] = row
            self.durable_rows = first_row + len(rows)
        self.pending.clear()


def test_a_database_sink_and_a_consumer_that_commits_only_durable_offsets():
    polls, sentences = _polls([40, 25, 0, 60])
    offsets = []  # the offset of every message, by row of the run
    committed = []
    database = _Database(ENCODER.output_dim)

    def consumer():
        offset = 1000
        for poll in polls:
            offsets.extend(range(offset, offset + len(poll)))
            offset += len(poll)
            yield poll

    for batch in ENCODER.encode_batches(consumer(), sink=database):
        assert batch.durable_rows <= batch.first_row + batch.n_rows
        if batch.durable_rows:  # nothing durable: nothing to confirm
            committed.append(offsets[batch.durable_rows - 1] + 1)
    assert committed == [1040, 1065, 1065, 1125]
    want = ENCODER.encode(sentences)
    np.testing.assert_array_equal(
        np.stack([database.points[i] for i in range(len(sentences))]), want
    )


def test_encode_into_a_sink_returns_nothing_and_flushes_it():
    class Flushing:
        dim = ENCODER.output_dim
        flushed = 0

        def target(self, _first_row, n_rows):
            return np.empty((n_rows, self.dim), F32)

        def commit(self, _first_row, rows):
            pass

        def flush(self):
            self.flushed += 1

    sink = Flushing()
    assert ENCODER.encode(texts(20), sink=sink) is None
    assert sink.flushed == 1
    with pytest.raises(ValueError, match="must take rows of 6 values"):
        ENCODER.encode(["w1"], sink=fse.MemorySink(5))
    with pytest.raises(TypeError, match="target and commit"):
        ENCODER.encode(["w1"], sink=object())


def test_the_memory_sink_is_the_sink_of_encode():
    sink = fse.MemorySink(ENCODER.output_dim, "float16")
    ENCODER.encode(texts(40), sink=sink)
    np.testing.assert_array_equal(sink.result(), ENCODER.encode(texts(40), dtype="float16"))
    assert _reference.inverse_norms(sink.result()).shape == (40,)
