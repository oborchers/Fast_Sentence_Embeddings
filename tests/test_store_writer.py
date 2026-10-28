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
    done = []

    def fail():
        raise OSError("disk full")

    writer.submit(fail, nbytes=5)
    writer.submit(done.append, 1, nbytes=5)  # dropped after the error
    with pytest.raises(OSError, match="disk full"):
        writer.drain()
    assert writer.failed
    with pytest.raises(RuntimeError, match="failed earlier: OSError"):
        writer.submit(done.append, 2)
    writer.settle()
    writer.stop()
    assert done == []
