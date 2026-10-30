"""Encoding into a sink and in batches: encode(sink=) and encode_batches."""

import numpy as np
import pytest

import fse
from fse import _reference
from fse._store._layout import StoreInfo, shards
from support.encoder import finished
from support.runtime import texts

ENCODER = finished()
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
        assert sink.durable_rows == 210  # flushed by the full iteration, before close
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
    with pytest.raises(ValueError, match="workers must be a positive int"):
        ENCODER.encode_batches(polls(), workers=0)
    sink = fse.MemorySink(ENCODER.output_dim)
    sink.inverse_norms = "yes"
    with pytest.raises(TypeError, match="inverse_norms of a sink must be a bool"):
        ENCODER.encode_batches(polls(), sink=sink)
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


def test_errors_name_the_rows_of_the_run():
    wv = fse.WordVectors.from_arrays(
        np.array([[1, 0], [1e-30, 0]], np.float32), words=["big", "tiny"]
    )
    encoder = fse.Encoder.from_word_vectors(wv, tokenizer="external")
    polls = [["big", "big"], ["big", "tiny"]]  # "tiny" cannot be normalized
    with pytest.raises(ValueError, match=r"^batch 1 \(rows 2 to 3 of the run\): row 3 cannot"):
        list(encoder.encode_batches(polls, workers=1, sink=fse.MemorySink(2)))
    with pytest.raises(TypeError, match=r"^batch 1 \(rows 1 to 2 of the run\): .*str"):
        list(encoder.encode_batches([["big"], ["big", 123]], workers=1))  # in its source
    with pytest.raises(ValueError, match=r"^batch 1 \(rows 2 to 3 of the run\): row 3 cannot"):
        list(encoder.encode_batches(polls, workers=1))  # the row of the run without a sink too


class _Odd(fse.MemorySink):
    def __init__(self, **attributes):
        super().__init__(ENCODER.output_dim)
        self.__dict__.update(attributes)


@pytest.mark.parametrize(
    ("attributes", "error", "match"),
    [
        ({"publish": 42}, TypeError, "publish of the sink must be a method"),
        ({"flush": "x"}, TypeError, "flush of the sink must be a method"),
        ({"durable_rows": "100"}, TypeError, "durable_rows of the sink must be an int"),
        ({"durable_rows": True}, TypeError, "must be an int, not bool"),
        ({"durable_rows": 10**6}, ValueError, "must be between 0"),
        ({"durable_rows": -1}, ValueError, "must be between 0"),
    ],
)
def test_the_optional_parts_of_a_sink_are_checked(attributes, error, match):
    with pytest.raises(error, match=match):
        list(ENCODER.encode_batches([["w1 w2"], ["w3"]], sink=_Odd(**attributes)))


def test_durable_rows_never_shrink():
    class Shrinking(fse.MemorySink):
        calls = 0

        def publish(self):
            self.calls += 1
            self.durable_rows = 1 if self.calls == 1 else 0  # 1 after the first, then 0

    sink = Shrinking(ENCODER.output_dim)
    with pytest.raises(ValueError, match="between 1"):
        list(ENCODER.encode_batches([["w1"], ["w2"]], sink=sink))


def test_a_unicode_error_keeps_its_type_and_names_its_batch():
    with pytest.raises(UnicodeEncodeError) as raised:
        list(ENCODER.encode_batches([["w1"], ["w2", "bad \udcff"]]))
    assert "in batch 1 (rows 1 to 2 of the run)" in raised.value.__notes__


def test_a_sink_is_checked_before_a_batch_is_read():
    read = []

    def polls():
        read.append(True)
        yield ["w1"]

    for attributes, match in [
        ({"flush": 1}, "flush of the sink must be a method"),
        ({"publish": "x"}, "publish of the sink must be a method"),
        ({"durable_rows": "0"}, "durable_rows of the sink must be an int"),
    ]:
        with pytest.raises(TypeError, match=match):
            ENCODER.encode_batches(polls(), sink=_Odd(**attributes))
    assert not read


def test_a_batch_is_a_list_or_tuple():
    polls, sentences = _polls([5, 0, 40])
    got = list(ENCODER.encode_batches([tuple(p) for p in polls], workers=2))
    want = list(ENCODER.encode_batches(polls, workers=2))
    assert [b.n_rows for b in got] == [5, 0, 40]
    for a, b in zip(got, want, strict=True):
        assert a.first_row == b.first_row
        assert a.rows.tobytes() == b.rows.tobytes()
    with pytest.raises(TypeError, match=r"batch 0 is a list_iterator; every batch is a list"):
        list(ENCODER.encode_batches([iter(sentences)]))


def test_stopping_early_flushes_nothing_until_the_sink_is_closed(tmp_path):
    polls, _ = _polls([30, 30, 30])
    sink = fse.ShardSink(tmp_path / "s", ENCODER, shard_rows=64)
    batches = ENCODER.encode_batches(polls, sink=sink)
    assert next(batches).durable_rows == 0
    batches.close()  # the next batches are never read
    assert sink.durable_rows == 0
    sink.close()  # publishes the rows committed so far
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert len(store) == 30


def test_a_sink_takes_one_run(tmp_path):
    with fse.ShardSink(tmp_path / "s", ENCODER, shard_rows=64) as sink:
        ENCODER.encode(texts(100), sink=sink)
        with pytest.raises(ValueError, match="expected rows from 100"):
            ENCODER.encode(texts(50), sink=sink)
    memory = fse.MemorySink(ENCODER.output_dim)
    ENCODER.encode(texts(10), sink=memory)
    with pytest.raises(ValueError, match="committed already"):
        ENCODER.encode(texts(5), sink=memory)


def test_a_sink_names_the_encoder_it_takes_by_encoder_fingerprint(tmp_path):
    other = finished(dim=12)
    with fse.ShardSink(tmp_path / "s", ENCODER, shard_rows=64) as sink:
        assert sink.encoder_fingerprint == ENCODER.fingerprint
        assert not hasattr(sink, "encoder")
        with pytest.raises(ValueError, match="takes the rows of encoder"):
            other.encode(texts(10), sink=sink)

    class Custom(fse.MemorySink):
        encoder_fingerprint = ENCODER.fingerprint

    with pytest.raises(ValueError, match="takes the rows of encoder"):
        other.encode(texts(10), sink=Custom(other.output_dim))
    ENCODER.encode(texts(10), sink=Custom(ENCODER.output_dim))


@pytest.mark.parametrize("sink", [False, True])
def test_errors_of_a_later_batch_name_the_rows_of_the_run(sink):
    words = ["ok", "big"]
    encoder = fse.Encoder.from_arrays(
        np.array([[1, 2, 3, 4], [1e5] * 4], np.float32), words=words, normalize=False
    )
    batches = [["ok"] * 1000, ["ok"] * 5 + ["big"] + ["ok"] * 4]
    target = fse.MemorySink(4, "float16") if sink else None
    with pytest.raises(ValueError, match=r"batch 1 .*row 1005 has values beyond the float16"):
        list(encoder.encode_batches(batches, dtype=None if sink else "float16", sink=target))
