"""Failures crossing the sink's write, hash and publication threads."""

import threading

import pytest

from fse import ShardSink
from fse._store import _shard
from support.encoder import finished


@pytest.mark.parametrize("durability", ["full", "none"])
@pytest.mark.parametrize("call", ["target", "commit", "publish", "flush", "close"])
def test_write_failure_survives_publication_and_hash_handover(
    tmp_path, monkeypatch, durability, call
):
    release = threading.Event()
    publishing = threading.Event()
    root = OSError(28, "No space left on device")
    real_commit = _shard.ShardWriter.commit
    real_wait = _shard.ShardWriter.wait_written

    def fail_write(*_args):
        assert release.wait(10)
        raise root

    def commit(writer, begin, codes):
        if writer.first_row == 8:
            assert release.wait(10)
            sink._io.settle()
            sink._publisher.settle()
        real_commit(writer, begin, codes)

    def wait_written(writer, failed):
        publishing.set()
        real_wait(writer, failed)

    monkeypatch.setattr(_shard, "write_at", fail_write)
    monkeypatch.setattr(_shard.ShardWriter, "commit", commit)
    monkeypatch.setattr(_shard.ShardWriter, "wait_written", wait_written)
    sink = ShardSink(tmp_path / "store", finished(), shard_rows=8, durability=durability)
    try:
        for first in (0, 8):
            target = sink.target(first, 8)
            for array in target:
                array[...] = 0
            sink.commit(first, target)
        assert publishing.wait(10)
        release.set()
        sink._settle()
        assert sink._io.failed
        assert sink._publisher.failed
        assert sink._hash.failed
        args = {"target": (16, 1), "commit": (16, target)}.get(call, ())
        with pytest.raises(OSError, match="No space left") as raised:
            getattr(sink, call)(*args)
        assert raised.value is root
        if call != "close":
            with pytest.raises(RuntimeError, match="failed earlier") as repeated:
                sink.target(16, 1)
            assert repeated.value.__cause__ is root
    finally:
        release.set()
        sink.close()
    assert sink.published_rows == sink.durable_rows == 0
    assert not list(sink.path.glob(".shard-*"))


@pytest.mark.parametrize("durability", ["full", "none"])
@pytest.mark.parametrize("stage", ["write", "hash", "publish"])
def test_first_failure_is_shared_without_marking_internal_handover_reported(
    tmp_path, durability, stage
):
    sink = ShardSink(tmp_path / "store", finished(), durability=durability)
    root = OSError(f"{stage} failed")

    def fail():
        raise root

    thread = {"write": sink._io, "hash": sink._hash, "publish": sink._publisher}[stage]
    try:
        thread.submit(fail)
        thread.settle()
        with pytest.raises(OSError, match=f"{stage} failed") as internal:
            sink._hand_on(None)
        assert internal.value is root
        assert not any(t.reported for t in (sink._io, sink._hash, sink._publisher))
        with pytest.raises(OSError, match=f"{stage} failed") as raised:
            sink.publish()
        assert raised.value is root
        with pytest.raises(RuntimeError, match="failed earlier") as repeated:
            sink.flush()
        assert repeated.value.__cause__ is root
    finally:
        sink.close()
    assert sink.published_rows == sink.durable_rows == 0
