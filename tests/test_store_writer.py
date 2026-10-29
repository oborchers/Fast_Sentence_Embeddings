"""The writer thread of a ShardSink: in order, bounded, errors never lost."""

import threading

import pytest

from fse._store import _writer
from fse._store._writer import WriterThread


def test_work_runs_in_order_in_one_thread():
    done, names = [], set()
    writer = WriterThread()

    def work(k):
        done.append(k)
        names.add(threading.current_thread().name)

    for k in range(100):
        writer.submit(work, k, nbytes=10)
    writer.drain()
    writer.stop()
    assert done == list(range(100))
    assert names == {"fse-shard-writer"}


def test_the_bytes_waiting_are_bounded(monkeypatch):
    monkeypatch.setattr(_writer, "MAX_PENDING_BYTES", 100)
    writer = WriterThread()
    gate = threading.Event()
    writer.submit(gate.wait, nbytes=60)
    submitted = threading.Event()

    def second():
        writer.submit(lambda: None, nbytes=60)  # 120 > 100: waits for the first
        submitted.set()

    thread = threading.Thread(target=second)
    thread.start()
    assert not submitted.wait(0.2)
    gate.set()
    assert submitted.wait(5)
    thread.join()
    writer.drain()
    writer.stop()
    big = WriterThread()
    big.submit(lambda: None, nbytes=10_000)  # one piece larger than the budget still passes
    big.drain()
    big.stop()


def test_an_error_is_raised_once_and_then_named():
    writer = WriterThread()
    done, release = [], threading.Event()

    def fail():
        release.wait()  # fails only after the next piece was handed over
        raise OSError("disk full")

    try:
        writer.submit(fail, nbytes=5)
        writer.submit(done.append, 1, nbytes=5)  # dropped after the error
        release.set()
        with pytest.raises(OSError, match="disk full"):
            writer.drain()
        assert writer.failed
        assert writer.reported
        with pytest.raises(RuntimeError, match="failed earlier: OSError"):
            writer.submit(done.append, 2)
        writer.settle()
    finally:
        release.set()
        writer.stop()
    assert done == []


def test_an_interrupt_while_stopping_drops_the_rest_and_still_ends_the_thread():
    writer = WriterThread()
    started, release, done = threading.Event(), threading.Event(), []

    def block():
        started.set()
        release.wait(10)

    writer.submit(block)
    writer.submit(done.append, 1)
    assert started.wait(10)
    real_join, real_abandon, joins = writer._thread.join, writer.abandon, []

    def join(*args):
        joins.append(args)
        if len(joins) == 1:
            raise KeyboardInterrupt  # Ctrl+C while stop waits for the thread
        return real_join(*args)

    def abandon(error):
        real_abandon(error)
        release.set()  # the piece running finishes only after the rest is dropped

    writer._thread.join = join
    writer.abandon = abandon
    with pytest.raises(KeyboardInterrupt):
        writer.stop()
    assert not writer._thread.is_alive()
    assert len(joins) == 2  # waited again after the interrupt
    assert done == []  # the work after the piece running was dropped
    writer.stop()  # stopped: nothing to do
