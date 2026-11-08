"""Lazy writer starts, failures, interrupts and competing first submissions."""

import signal
import threading

import pytest

from fse._store._writer import WriterFailure, WriterThread


def test_unused_writer_checks_drains_and_stops_without_a_thread():
    writer = WriterThread()
    assert writer._thread.ident is None
    writer.check()
    writer.drain()
    writer.stop()
    writer.stop()
    assert writer._thread.ident is None
    assert writer._ended.is_set()
    with pytest.raises(RuntimeError, match="stopped"):
        writer.submit(lambda: None)


def test_abandoned_unused_writer_never_starts():
    writer = WriterThread()
    writer.abandon()
    with pytest.raises(RuntimeError, match="stopped"):
        writer.submit(lambda: None)
    writer.stop()
    assert writer._thread.ident is None


@pytest.mark.parametrize("after_start", [False, True])
@pytest.mark.parametrize("report", [False, True])
def test_start_failure_is_shared_and_an_existing_thread_is_stopped(
    monkeypatch, after_start, report
):
    failure = WriterFailure()
    writer, other = WriterThread(failure=failure), WriterThread(failure=failure)
    root = RuntimeError("cannot start writer")
    real = writer._thread.start
    done = []

    def fail():
        assert writer._launch_state == "launching"
        if after_start:
            real()
        raise root

    monkeypatch.setattr(writer._thread, "start", fail)
    try:
        with pytest.raises(RuntimeError, match="cannot start writer") as raised:
            writer.submit(done.append, 1, report=report)
        assert raised.value is root
        assert writer.failed
        assert writer.reported is report
        with pytest.raises(RuntimeError, match="cannot start writer") as shared:
            other.check()
        assert shared.value.__cause__ is root if report else shared.value is root
        with pytest.raises(RuntimeError, match="failed earlier") as repeated:
            writer.check()
        assert repeated.value.__cause__ is root
        writer.settle()
        assert done == []
    finally:
        writer.stop()
        other.stop()
    assert writer._ended.is_set() is after_start
    if after_start:
        writer._thread.join(10)
        assert not writer._thread.is_alive()


@pytest.mark.usefixtures("python_sigint")
@pytest.mark.parametrize("when", ["before", "after"])
def test_standard_interrupt_during_start_leaves_a_thread_that_close_can_stop(monkeypatch, when):
    writer = WriterThread()
    real = writer._thread.start
    done = []

    def interrupt():
        if when == "before":
            signal.raise_signal(signal.SIGINT)
        real()
        if when == "after":
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(writer._thread, "start", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            writer.submit(done.append, 1)
        assert writer._launch_state == "started"
        assert not writer.failed
        writer.drain()
        assert done == []  # the interrupted submission did not hand work over
    finally:
        writer.stop()
    writer._thread.join(10)
    assert not writer._thread.is_alive()


@pytest.mark.parametrize("competing", ["submit", "stop"])
def test_start_serializes_with_another_first_submit_or_stop(monkeypatch, competing):
    writer = WriterThread()
    real = writer._thread.start
    entered, release = threading.Event(), threading.Event()
    calls, done, errors = [], [], []
    reached = threading.Event()

    class WatchedCondition(threading.Condition):
        def __enter__(self):
            if threading.current_thread().name == "competitor":
                reached.set()
            return super().__enter__()

    writer._state = WatchedCondition()

    def start():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        real()

    def first():
        try:
            writer.submit(done.append, 1)
        except BaseException as error:
            errors.append(error)

    def second():
        try:
            if competing == "submit":
                writer.submit(done.append, 2)
            else:
                writer.stop()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(writer._thread, "start", start)
    first_thread = threading.Thread(target=first, name="first")
    second_thread = threading.Thread(target=second, name="competitor")
    try:
        first_thread.start()
        assert entered.wait(10)
        second_thread.start()
        assert reached.wait(10)
        release.set()
        first_thread.join(10)
        second_thread.join(10)
        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert errors == []
        writer.drain()
        assert calls == [1]
        assert done == ([1, 2] if competing == "submit" else [1])
    finally:
        release.set()
        writer.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        writer.submit(lambda: None)


def test_stop_before_first_submit_wins():
    writer = WriterThread()
    writer.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        writer.submit(lambda: None)
    assert writer._thread.ident is None


@pytest.mark.parametrize("started", [False, True])
def test_stop_can_be_retried_after_an_interrupt_before_handover(monkeypatch, started):
    writer = WriterThread()
    if started:
        writer.submit(lambda: None)
        writer.drain()
    if started:
        real = writer._items

        class InterruptedQueue(type(real)):
            def append(self, _item):
                raise KeyboardInterrupt

        monkeypatch.setattr(writer, "_items", InterruptedQueue())
    else:
        real = writer._ended.set

        def interrupt():
            raise KeyboardInterrupt

        monkeypatch.setattr(writer._ended, "set", interrupt)
    with pytest.raises(KeyboardInterrupt):
        writer.stop()
    if started:
        monkeypatch.setattr(writer, "_items", real)
    else:
        monkeypatch.setattr(writer._ended, "set", real)
    stopped = threading.Thread(target=writer.stop, daemon=True)
    stopped.start()
    stopped.join(10)
    assert not stopped.is_alive()
    assert writer._ended.is_set()
    writer.stop()
