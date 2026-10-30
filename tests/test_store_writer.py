"""The writer thread of a ShardSink: in order, bounded, errors never lost."""

import signal
import threading
from collections import deque

import pytest

from fse._store import _writer
from fse._store._writer import WriterThread
from support.waiting import arrived, blocked_in_wait


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
    submitted = threading.Event()
    reached = blocked_in_wait(writer._state, "second")

    def second():
        writer.submit(lambda: None, nbytes=60)  # 120 > 100: waits for the first
        submitted.set()

    thread = threading.Thread(target=second, name="second")
    try:
        writer.submit(gate.wait, nbytes=60)
        thread.start()
        assert arrived(reached, writer._state), "the second piece never waited for room"
        assert not submitted.wait(0.2)
        gate.set()
        assert submitted.wait(5)
        thread.join(5)
        writer.drain()
    finally:
        gate.set()
        if thread.is_alive():
            thread.join(5)
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


TIMEOUT = 10  # seconds; every wait below fails the test when it runs out


def _joined(thread: threading.Thread) -> None:
    thread.join(TIMEOUT)
    assert not thread.is_alive(), f"{thread.name} did not end within {TIMEOUT} s"


def test_abandon_finishes_the_running_piece_drops_the_rest_and_ends_the_thread():
    writer = WriterThread()
    started, release, in_stop = (threading.Event() for _ in range(3))
    finished, done = [], []

    def block():
        started.set()
        assert release.wait(TIMEOUT)
        finished.append(True)

    def stop():
        in_stop.set()
        writer.stop()

    stopper = threading.Thread(target=stop, name="stopper")
    try:
        writer.submit(block)
        writer.submit(done.append, 1)
        assert started.wait(TIMEOUT)
        writer.abandon()
        assert writer.failed
        with pytest.raises(RuntimeError, match="stopped"):
            writer.submit(done.append, 2)  # an abandoned thread takes no more work
        # Ended means the thread said so, not join or is_alive (an interrupted join of
        # Python 3.11 and 3.12 can report a running thread as ended).
        writer._thread.is_alive = lambda: False
        writer._thread.join = lambda *_args: pytest.fail("stop must not rely on join")
        stopper.start()
        assert in_stop.wait(TIMEOUT)
        assert not writer._ended.wait(0.2)  # the piece still runs ...
        assert stopper.is_alive()  # ... and stop waits for it
        release.set()
        _joined(stopper)
        assert writer._ended.is_set()
        assert finished == [True]  # the running piece was finished
        assert done == []  # the work after it was dropped
        writer.settle()  # nothing waits any more
    finally:
        release.set()
        if stopper.is_alive():
            _joined(stopper)


class _Interrupted(deque):
    """The queue of a writer, interrupted (Ctrl+C) just before or just after an append."""

    def __init__(self, after: bool):
        super().__init__()
        self.after, self.armed = after, True

    def append(self, item):
        if self.armed and item is not None:
            self.armed = False
            if self.after:
                super().append(item)
            raise KeyboardInterrupt
        super().append(item)


@pytest.mark.parametrize("after", [False, True])
def test_an_interrupt_around_the_enqueue_leaves_the_piece_queued_or_not(after):
    writer = WriterThread()
    try:
        with writer._state:
            writer._items = _Interrupted(after)  # the thread is idle: nothing queued
        done = []
        with pytest.raises(KeyboardInterrupt):
            writer.submit(done.append, 1, nbytes=7)
        writer.submit(done.append, 2, nbytes=7)
        settled = threading.Thread(target=writer.settle, name="settle")
        settled.start()
        _joined(settled)  # no wait hangs
        assert done == ([1, 2] if after else [2])  # queued once, or not at all
        assert writer._running == -1
        assert not writer._items
    finally:
        writer.stop()


def test_a_producer_waiting_for_room_is_released_when_the_thread_ends(monkeypatch):
    monkeypatch.setattr(_writer, "MAX_PENDING_BYTES", 100)
    writer = WriterThread()
    started, release, waiting = (threading.Event() for _ in range(3))
    errors = []
    reached = blocked_in_wait(writer._state, "producer")

    def block():
        started.set()
        assert release.wait(TIMEOUT)

    def produce():
        waiting.set()
        try:
            writer.submit(lambda: None, nbytes=60)  # 60 + 60 > 100: waits for room
        except RuntimeError as error:
            errors.append(error)

    producer = threading.Thread(target=produce, name="producer")
    try:
        writer.submit(block, nbytes=60)
        assert started.wait(TIMEOUT)
        producer.start()
        assert waiting.wait(TIMEOUT)
        assert arrived(reached, writer._state), "the producer never waited for room"
        producer.join(0.2)
        assert producer.is_alive()  # it waits for room
        writer.abandon()
        release.set()
        writer.stop()
        _joined(producer)
        assert [str(e) for e in errors] == ["the writer thread of the sink is stopped"]
    finally:
        release.set()
        if producer.is_alive():
            _joined(producer)


def test_at_most_max_pending_pieces_wait():
    writer = WriterThread()
    started, release = threading.Event(), threading.Event()
    reached = blocked_in_wait(writer._state, "extra")

    def block():
        started.set()
        assert release.wait(TIMEOUT)

    try:
        writer.submit(block)
        assert started.wait(TIMEOUT)
        for _ in range(_writer.MAX_PENDING):
            writer.submit(lambda: None)
        assert len(writer._items) == _writer.MAX_PENDING
        extra = threading.Thread(target=writer.submit, args=(lambda: None,), name="extra")
        extra.start()
        assert arrived(reached, writer._state), "the 17th piece never waited for room"
        extra.join(0.2)
        assert extra.is_alive()  # the 17th waits
        release.set()
        _joined(extra)
    finally:
        release.set()
        writer.stop()


@pytest.mark.usefixtures("python_sigint")
def test_interrupts_are_deferred_to_the_end_of_the_block():
    calls = []

    def interrupted():
        with _writer.deferred_interrupts(lambda: calls.append(1)) as noted:
            signal.raise_signal(signal.SIGINT)
            assert len(noted) == 1  # handled at once, and only noted
            signal.raise_signal(signal.SIGINT)  # a further one is only noted
            assert len(noted) == 2
            calls.append("after")  # the block goes on

    with pytest.raises(KeyboardInterrupt):
        interrupted()
    assert calls == [1, "after"]  # on_interrupt once, at the first
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    with _writer.deferred_interrupts(lambda: calls.append(2)):
        pass  # no interrupt: nothing raised, nothing called
    assert calls == [1, "after"]


@pytest.mark.usefixtures("python_sigint")
@pytest.mark.parametrize("interrupt", [False, True])
def test_the_handler_is_restored_when_the_block_raises(interrupt):
    def failing():
        with _writer.deferred_interrupts(lambda: None):
            if interrupt:
                signal.raise_signal(signal.SIGINT)
            raise ValueError("the block failed")

    with pytest.raises(KeyboardInterrupt if interrupt else ValueError) as raised:
        failing()
    if interrupt:
        assert isinstance(raised.value.__context__, ValueError)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_another_handler_is_left_alone_and_nothing_is_abandoned():
    seen, abandoned = [], []

    def own(signum, _frame):
        seen.append(signum)  # a handler of the program that returns

    previous = signal.signal(signal.SIGINT, own)
    try:
        with _writer.deferred_interrupts(lambda: abandoned.append(1)) as noted:
            assert signal.getsignal(signal.SIGINT) is own
            signal.raise_signal(signal.SIGINT)
            assert seen == [signal.SIGINT]  # it ran at once
        assert noted == []
        assert abandoned == []
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        with _writer.deferred_interrupts(lambda: abandoned.append(1)):
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
            signal.raise_signal(signal.SIGINT)  # ignored
        assert abandoned == []
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        with _writer.deferred_interrupts(lambda: abandoned.append(1)):
            assert signal.getsignal(signal.SIGINT) is signal.SIG_DFL  # not replaced
    finally:
        signal.signal(signal.SIGINT, previous)


def test_interrupts_are_not_deferred_outside_the_main_thread():
    seen = []

    def run():
        with _writer.deferred_interrupts(lambda: None):
            seen.append(signal.getsignal(signal.SIGINT))  # unchanged, no ValueError

    thread = threading.Thread(target=run)
    thread.start()
    _joined(thread)
    assert seen == [signal.getsignal(signal.SIGINT)]
