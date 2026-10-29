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


def test_abandon_finishes_the_running_piece_drops_the_rest_and_ends_the_thread():
    writer = WriterThread()
    started, release, finished, done = threading.Event(), threading.Event(), [], []

    def block():
        started.set()
        release.wait(10)
        finished.append(True)

    stopper = threading.Thread(target=writer.stop)
    try:
        writer.submit(block)
        writer.submit(done.append, 1)
        assert started.wait(10)
        writer.abandon()
        assert writer.failed
        with pytest.raises(RuntimeError, match="stopped"):
            writer.submit(done.append, 2)  # an abandoned thread takes no more work
        # Ended means the thread said so, not join or is_alive (an interrupted join of
        # Python 3.11 and 3.12 can report a running thread as ended).
        writer._thread.is_alive = lambda: False
        writer._thread.join = lambda *_args: pytest.fail("stop must not rely on join")
        stopper.start()
        assert not writer._ended.wait(0.2)  # stop waits for the piece running
        assert stopper.is_alive()
        release.set()
        stopper.join(10)
        assert not stopper.is_alive()
        assert writer._ended.is_set()
        assert finished == [True]  # the running piece was finished
        assert done == []  # the work after it was dropped
        writer.settle()  # nothing waits any more
    finally:
        release.set()
        if stopper.is_alive():
            stopper.join(10)


@pytest.mark.usefixtures("python_sigint")
def test_interrupts_are_deferred_to_the_end_of_the_block():
    import signal

    calls = []
    before = signal.getsignal(signal.SIGINT)

    def interrupted():
        with _writer.deferred_interrupts(lambda: calls.append(1)):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)  # a further one is only noted
            calls.append("after")  # the block goes on

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert calls == [1, "after"]  # on_interrupt once, at the first
    assert signal.getsignal(signal.SIGINT) is before
    with _writer.deferred_interrupts(lambda: calls.append(2)):
        pass  # no interrupt: nothing raised, nothing called
    assert calls == [1, "after"]


def test_interrupts_are_not_deferred_outside_the_main_thread():
    import signal

    seen = []

    def run():
        with _writer.deferred_interrupts(lambda: None):
            seen.append(signal.getsignal(signal.SIGINT))  # unchanged, no ValueError

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(10)
    assert seen == [signal.getsignal(signal.SIGINT)]
