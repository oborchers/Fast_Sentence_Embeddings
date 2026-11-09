"""Sinks start only the writers needed by committed rows, including failed starts."""

import threading

import pytest

import fse
from support import score_heads as sh
from support.encoder import finished
from support.logs import captured
from support.resume import ENCODER, SHARD_ROWS, sentences, write


def _threads(sink):
    return [worker._thread for worker in (sink._io, sink._hash, sink._publisher)]


def _assert_stopped(sink):
    for thread in _threads(sink):
        if thread.ident is not None:
            thread.join(10)
        assert not thread.is_alive()


@pytest.fixture(params=["vectors", "scores"])
def sink(tmp_path, request):
    if request.param == "vectors":
        result = fse.ShardSink(tmp_path / "s", finished(), shard_rows=4)
    else:
        result = fse.ScoreSink(tmp_path / "s", sh.group(), shard_rows=4)
        result._attach({"quality": "folded", "toxic": "folded", "topic": "folded"})
    try:
        yield result
    finally:
        result.close()
        _assert_stopped(result)


def _commit(sink, count):
    target = sink.target(0, count)
    for array in target:
        array[...] = 0
    sink.commit(0, target)


def _commit_and_flush(sink):
    _commit(sink, 4)
    sink.flush()  # the publish start happens in the hash thread


def test_partial_shard_starts_only_write_and_hash_then_flush_starts_publish(sink):
    assert all(t.ident is None for t in _threads(sink))
    _commit(sink, 1)
    sink._settle()
    assert [t.is_alive() for t in _threads(sink)] == [True, True, False]
    assert sink._publisher._thread.ident is None
    assert sink.flush() == 1
    assert all(t.is_alive() for t in _threads(sink))
    sink.close()
    _assert_stopped(sink)


@pytest.mark.parametrize("stage", ["write", "hash", "publish"])
def test_each_start_failure_reaches_caller_and_close_cleans_shards(sink, monkeypatch, stage):
    root = RuntimeError(f"cannot start {stage}")
    worker = {"write": sink._io, "hash": sink._hash, "publish": sink._publisher}[stage]

    def fail():
        raise root

    monkeypatch.setattr(worker._thread, "start", fail)
    with captured() as records:
        with pytest.raises(RuntimeError, match=f"cannot start {stage}") as raised:
            _commit_and_flush(sink)
        assert raised.value is root
        assert worker.failed
        assert sink._committed == (4 if stage == "publish" else 0)
        with pytest.raises(RuntimeError, match="failed earlier") as repeated:
            sink.publish()
        assert repeated.value.__cause__ is root
        sink.close()
    (closed,) = [r for r in records if r.fse_event.endswith("sink_closed")]
    assert closed.fse_outcome == "error"
    assert closed.fse_dropped_rows == (4 if stage == "publish" else 0)
    assert sink.published_rows == sink.durable_rows == 0
    assert not list(sink.path.glob(".shard-*"))
    assert not list(sink.path.glob("shard-*"))
    _assert_stopped(sink)


def test_publisher_start_and_close_from_another_thread(sink, monkeypatch):
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    real = sink._publisher._thread.start
    real_settle = sink._hash.settle
    errors = []

    def start():
        entered.set()
        assert release.wait(10)
        real()

    def settle():
        if threading.current_thread().name == "closer":
            closing.set()
        real_settle()

    def close():
        try:
            sink.close()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(sink._publisher._thread, "start", start)
    monkeypatch.setattr(sink._hash, "settle", settle)
    closer = threading.Thread(target=close, name="closer")
    try:
        _commit(sink, 4)
        assert entered.wait(10)
        closer.start()
        assert closing.wait(10)
        release.set()
        closer.join(10)
        assert not closer.is_alive()
        assert errors == []
        assert sink.published_rows == sink.durable_rows == 4
        _assert_stopped(sink)
    finally:
        release.set()


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("sign_bits", [False, True])
def test_constructing_and_closing_vector_sink_start_nothing(tmp_path, resume, sign_bits):
    with fse.ShardSink(tmp_path / "s", finished(), resume=resume, sign_bits=sign_bits) as sink:
        assert all(t.ident is None for t in _threads(sink))
    assert all(t.ident is None for t in _threads(sink))


def test_score_schema_property_without_close_starts_nothing(tmp_path, monkeypatch):
    starts = []
    real = threading.Thread.start

    def start(thread):
        starts.append(thread.name)
        real(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    assert fse.ScoreSink(tmp_path / "s", sh.group())._score_schema.dim == 7
    assert starts == []


def test_resume_backfills_bits_on_attach_without_starting_sink_threads(tmp_path, monkeypatch):
    path = tmp_path / "s"
    source = sentences(SHARD_ROWS)
    write(path, source)
    assert not list(path.glob("sign_bits-*"))
    with fse.ShardSink(path, ENCODER, shard_rows=SHARD_ROWS, resume=True, sign_bits=True) as sink:
        real = sink._attach_source

        def attach(*args, **kwargs):
            result = real(*args, **kwargs)
            assert all(t.ident is None for t in _threads(sink))
            assert list(path.glob("sign_bits-*"))
            return result

        monkeypatch.setattr(sink, "_attach_source", attach)
        ENCODER.encode(source, sink=sink, workers=1)
        assert all(t.ident is None for t in _threads(sink))


def test_uncertain_start_can_close_before_a_late_bootstrap_without_taking_work(sink, monkeypatch):
    import signal

    worker = sink._io
    thread = worker._thread
    entered, release = threading.Event(), threading.Event()
    real_bootstrap = thread._bootstrap
    took = []

    def bootstrap():
        entered.set()
        assert release.wait(10)
        real_bootstrap()

    def interrupted_wait(*_args, **_kwargs):
        assert entered.wait(10)  # OS thread exists but has not set ident
        signal.raise_signal(signal.SIGINT)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    def take():
        took.append(True)
        raise AssertionError("a late bootstrap must never take work")

    monkeypatch.setattr(thread, "_bootstrap", bootstrap)
    monkeypatch.setattr(thread._started, "wait", interrupted_wait)
    monkeypatch.setattr(worker, "_take", take)
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            _commit(sink, 1)
        assert thread.ident is None
        assert worker._launch_state == "launching"
        assert not worker._ended.is_set()
        assert worker._abandoned
        assert not worker._accepting
        sink.close()
        sink.close()
        assert not worker._ended.is_set()
        assert not list(sink.path.glob(".shard-*"))
        assert sink.published_rows == sink.durable_rows == 0
    finally:
        signal.signal(signal.SIGINT, previous)
        release.set()
        assert worker._ended.wait(10)
        thread.join(10)
    assert worker._launch_state == "started"
    assert not thread.is_alive()
    assert took == []
