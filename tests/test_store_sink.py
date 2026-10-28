"""The ShardSink: the rows of an encoder written straight into the shards of a store."""

import os

import numpy as np
import pytest

import fse
from fse import _reference
from fse._container import NpyContainer
from fse._store import ShardSink
from fse._store._layout import StoreInfo, read_store, shard_name, shards
from support.encoder import unfolded
from support.runtime import texts

ENCODER = unfolded()  # remove components, project to 6, normalize


def _store_rows(path, dtype):
    """codes, scales, inverse norms and the shards of the store, joined."""
    info = StoreInfo(ENCODER.fingerprint, dtype, ENCODER.output_dim)
    found = shards(path, info)
    codes = np.concatenate([t["vectors"] for _, t in found])
    scales = np.concatenate([t["scales"] for _, t in found]) if dtype == "int8" else None
    norms = np.concatenate([t["inv_norms"] for _, t in found])
    return codes, scales, norms, [s for s, _ in found]


def _want(dtype, sentences):
    rows = ENCODER.encode(sentences, workers=1)
    return _reference.quantize_rows(rows, dtype)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 3])
def test_a_run_into_shards_gives_the_rows_of_encode(tmp_path, dtype, workers):
    sentences = texts(1000)
    with ShardSink(tmp_path / "store", ENCODER, dtype=dtype, shard_rows=128) as sink:
        ENCODER.encode(sentences, workers=workers, sink=sink)
        assert sink.durable_rows == sink.committed_rows == 1000
    codes, scales, norms, found = _store_rows(tmp_path / "store", dtype)
    want_codes, want_scales = _want(dtype, sentences)
    np.testing.assert_array_equal(codes.view(np.uint8), want_codes.view(np.uint8))
    if dtype == "int8":
        np.testing.assert_array_equal(scales.view(np.uint32), want_scales.view(np.uint32))
    np.testing.assert_array_equal(
        norms.view(np.uint32), _reference.inverse_norms(want_codes).view(np.uint32)
    )
    assert [s.rows for s in found] == [128] * 7 + [104]  # shards of 128, the last shorter
    assert [s.first_row for s in found] == list(range(0, 1000, 128))
    for shard in found:
        NpyContainer().verify(tmp_path / "store" / shard.name)
        np.load(tmp_path / "store" / shard.name / "vectors.npy", mmap_mode="r")
    assert not [p for p in (tmp_path / "store").iterdir() if p.name.startswith(".")]


def test_appends_continue_the_store(tmp_path):
    first, second = texts(300, seed=1), texts(200, seed=2)
    with ShardSink(tmp_path / "s", ENCODER, dtype="float16", shard_rows=256) as sink:
        ENCODER.encode(first, sink=sink)
    with ShardSink(tmp_path / "s", ENCODER, dtype="float16", shard_rows=256) as sink:
        assert sink.first_row == 300
        ENCODER.encode(second, sink=sink)
    codes, _, _, found = _store_rows(tmp_path / "s", "float16")
    np.testing.assert_array_equal(codes, _want("float16", first + second)[0])
    assert [(s.first_row, s.rows) for s in found] == [(0, 256), (256, 44), (300, 200)]


def test_seal_and_publish_make_rows_durable_before_a_shard_is_full(tmp_path):
    with ShardSink(tmp_path / "s", ENCODER, shard_rows=1024) as sink:
        for k, batch in enumerate(
            ENCODER.encode_batches([texts(50, seed=s) for s in range(3)], sink=sink)
        ):
            assert batch.durable_rows == 50 * k  # nothing full yet: only sealed shards
            sink.seal()
            assert sink.publish() == 50 * (k + 1)
    _, _, _, found = _store_rows(tmp_path / "s", "float32")
    assert [s.rows for s in found] == [50, 50, 50]


def test_the_store_must_match_the_encoder(tmp_path):
    with ShardSink(tmp_path / "s", ENCODER, dtype="int8") as sink:
        ENCODER.encode(texts(10), sink=sink)
    with pytest.raises(ValueError, match="holds rows of dtype 'int8', not 'float16'"):
        ShardSink(tmp_path / "s", ENCODER, dtype="float16")
    other = unfolded(steps="normalize")
    with pytest.raises(ValueError, match="holds rows of encoder"):
        ShardSink(tmp_path / "s", other, dtype="int8")
    with pytest.raises(TypeError, match="encoder must be an Encoder"):
        ShardSink(tmp_path / "t", "encoder")
    assert read_store(tmp_path / "s").dtype == "int8"


def test_encode_checks_the_sink(tmp_path):
    with ShardSink(tmp_path / "s", ENCODER, dtype="float16") as sink:
        with pytest.raises(ValueError, match="dtype 'int8' differs from the dtype 'float16'"):
            ENCODER.encode(["w1"], dtype="int8", sink=sink)
        with pytest.raises(ValueError, match="takes the rows of encoder"):
            unfolded(steps="none").encode(["w1"], sink=sink)
        assert ENCODER.encode(["w1 w2"], dtype="float16", sink=sink) is None


def test_after_an_error_the_committed_rows_are_published(tmp_path):
    def sentences():
        yield from texts(600)
        raise OSError("the stream broke")

    with (
        pytest.raises(OSError, match="the stream broke"),
        ShardSink(tmp_path / "s", ENCODER, shard_rows=128) as sink,
    ):
        ENCODER.encode(sentences(), workers=2, sink=sink)
    assert sink.durable_rows == 600  # every row before the error is committed first
    codes = _store_rows(tmp_path / "s", "float32")[0]
    np.testing.assert_array_equal(codes, _want("float32", texts(600))[0])
    assert not [p for p in (tmp_path / "s").iterdir() if p.name.startswith(".")]


def test_a_sink_takes_one_run(tmp_path):
    sink = ShardSink(tmp_path / "s", ENCODER)
    ENCODER.encode(["w1"], sink=sink)
    sink.close()
    with pytest.raises(RuntimeError, match="the ShardSink is closed"):
        ENCODER.encode(["w1"], sink=sink)
    assert (tmp_path / "s" / shard_name(0)).is_dir()


def test_concurrent_appenders_never_replace_a_shard(tmp_path):
    with ShardSink(tmp_path / "s", ENCODER) as sink:
        ENCODER.encode(texts(5), sink=sink)
    a = ShardSink(tmp_path / "s", ENCODER, shard_rows=16)
    b = ShardSink(tmp_path / "s", ENCODER, shard_rows=16)
    assert a.first_row == b.first_row == 5
    ENCODER.encode(texts(16, seed=7), sink=a)  # a publishes shard-5 first
    with pytest.raises(FileExistsError, match=r"another writer appended .* at row 5 first"):
        ENCODER.encode(texts(16, seed=8), sink=b)
    b.close()
    codes, _, _, found = _store_rows(tmp_path / "s", "float32")
    assert [(s.first_row, s.rows) for s in found] == [(0, 5), (5, 16)]
    np.testing.assert_array_equal(codes[5:], _want("float32", texts(16, seed=7))[0])
    assert not [p for p in (tmp_path / "s").iterdir() if p.name.startswith(".")]


def test_a_wrong_commit_changes_nothing(tmp_path):
    with ShardSink(tmp_path / "s", ENCODER, shard_rows=8) as sink:
        target = sink.target(0, 5)
        with pytest.raises(ValueError, match="not the target reserved"):
            sink.commit(0, tuple(np.empty_like(a) for a in target))
        target[0][...] = 0
        target[1][...] = 0
        sink.commit(0, target)  # the reservation is still there
        del target
        assert sink.flush() == 5


def test_after_a_failed_write_the_sink_takes_no_more_rows(tmp_path, monkeypatch):
    from fse._store import _shard

    def full(*_args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_shard, "write_at", full)
    sink = ShardSink(tmp_path / "s", ENCODER, shard_rows=64)
    with pytest.raises(OSError, match="No space left"):
        ENCODER.encode(texts(300), workers=2, sink=sink)
    with pytest.raises(RuntimeError, match="failed earlier"):
        ENCODER.encode(texts(10), sink=sink)
    sink.close()  # no second error; nothing more is published
    assert sink.durable_rows == 0
    assert not [p for p in (tmp_path / "s").iterdir() if p.name.startswith(".")]


def test_close_raises_a_write_error_nothing_raised_before(tmp_path, monkeypatch):
    from fse._store import _shard

    def full(*_args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_shard, "write_at", full)
    sink = ShardSink(tmp_path / "s", ENCODER, shard_rows=8)
    target = sink.target(0, 3)
    target[0][...] = 0
    target[1][...] = 0
    sink.commit(0, target)  # handed to the writer threads; their error comes later
    with pytest.raises(OSError, match="No space left"):
        sink.close()
    sink.close()  # closed: no second error
    assert sink.durable_rows == 0
    assert not [p for p in (tmp_path / "s").iterdir() if p.name.startswith(".")]


def test_close_raises_a_publish_error_another_thread_met_first(tmp_path, monkeypatch):
    import threading
    import time

    from fse._store import _shard

    failed, resume = threading.Event(), threading.Event()
    real_commit = _shard.ShardWriter.commit

    def publish(*_args):
        failed.set()
        raise OSError(5, "Input/output error")

    def commit(self, begin, codes):
        if self.first_row == 8:
            resume.wait(10)  # the hash thread hands shard 2 on after the publisher failed
        real_commit(self, begin, codes)

    monkeypatch.setattr(_shard, "publish_directory", publish)
    monkeypatch.setattr(_shard.ShardWriter, "commit", commit)
    sink = ShardSink(tmp_path / "s", ENCODER, shard_rows=8)
    for first in (0, 8):
        target = sink.target(first, 8)
        target[0][...] = 0
        target[1][...] = 0
        sink.commit(first, target)
    assert failed.wait(10)
    deadline = time.monotonic() + 10
    while not sink._publisher.failed and time.monotonic() < deadline:
        time.sleep(0.001)  # the error is kept just after publish raised
    resume.set()  # now the hash thread meets the publisher's error handing shard 2 on
    sink._hash.settle()
    assert sink._hash.failed
    with pytest.raises(OSError, match="Input/output error"):
        sink.close()


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="no /dev/fd to count open files")
def test_reserving_many_small_shards_opens_no_file(tmp_path):
    sink = ShardSink(tmp_path / "s", ENCODER, shard_rows=1)
    before = len(os.listdir("/dev/fd"))
    target = sink.target(0, 1000)  # 1,000 shards ahead of their writes
    assert len(os.listdir("/dev/fd")) <= before + 2
    del target
    sink.close()  # drops the rows never committed
    assert sink.durable_rows == 0
    assert not [p for p in (tmp_path / "s").iterdir() if p.name.startswith(".")]


def test_two_writers_meet_at_the_rename_and_one_wins(tmp_path, monkeypatch):
    import threading

    from fse._store import _shard

    with ShardSink(tmp_path / "s", ENCODER) as sink:
        ENCODER.encode(texts(5), sink=sink)
    meet, real = threading.Barrier(2, timeout=10), _shard.publish_directory
    seen = []

    def publish(tmp, path):
        meet.wait()  # both shards are written and synced; both rename now
        before = len(fse.VectorStore.open(tmp_path / "s"))
        real(tmp, path)
        seen.append((before, len(fse.VectorStore.open(tmp_path / "s"))))

    monkeypatch.setattr(_shard, "publish_directory", publish)
    errors = []

    def append(seed):
        try:
            with ShardSink(tmp_path / "s", ENCODER, shard_rows=16) as sink:
                ENCODER.encode(texts(16, seed=seed), sink=sink)
        except FileExistsError as error:
            errors.append(error)

    threads = [threading.Thread(target=append, args=(s,)) for s in (7, 8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert len(errors) == 1  # exactly one writer published shard-5
    assert seen == [(5, 21)]  # a reader sees the shard whole or not at all
    codes, _, _, found = _store_rows(tmp_path / "s", "float32")
    assert [(s.first_row, s.rows) for s in found] == [(0, 5), (5, 16)]
    assert any(np.array_equal(codes[5:], _want("float32", texts(16, seed=s))[0]) for s in (7, 8))
    assert not [p for p in (tmp_path / "s").iterdir() if p.name.startswith(".")]
