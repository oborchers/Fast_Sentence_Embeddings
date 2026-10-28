import contextlib
import random
import threading
import time

import numpy as np
import pytest

from fse._runtime import _executor
from fse._runtime._executor import run


def _record():
    committed = []
    return committed, lambda task, result: committed.append((task, result))


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_results_are_committed_in_order(workers):
    rng = random.Random(0)
    delays = [rng.random() / 500 for _ in range(200)]
    committed, commit = _record()

    def work(i):
        time.sleep(delays[i])
        return i * i

    run(range(200), work, commit, workers=workers)
    assert committed == [(i, i * i) for i in range(200)]


def test_at_most_window_tasks_are_in_flight():
    lock, state = threading.Lock(), {"now": 0, "most": 0, "taken": 0, "committed": 0}

    def tasks():
        for i in range(100):
            state["taken"] += 1
            assert state["taken"] - state["committed"] <= 3
            yield i

    def work(i):
        with lock:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(0.001)
        with lock:
            state["now"] -= 1
        return i

    def commit(_task, _result):
        state["committed"] += 1

    run(tasks(), work, commit, workers=8, window=3)
    assert state["most"] <= 3
    assert state["committed"] == 100


def test_inline_runs_in_the_calling_thread(monkeypatch):
    def no_pool(**_kwargs):
        raise AssertionError("a thread pool was created")

    monkeypatch.setattr(_executor, "ThreadPoolExecutor", no_pool)
    me = threading.get_ident()
    seen = set()
    run(
        range(5),
        lambda _i: seen.add(threading.get_ident()),
        lambda *_: None,
        workers=8,
        inline=True,
    )
    run(range(5), lambda _i: seen.add(threading.get_ident()), lambda *_: None, workers=1)
    assert seen == {me}


def test_commits_run_in_the_calling_thread():
    me = threading.get_ident()
    threads = set()
    run(range(50), lambda i: i, lambda *_: threads.add(threading.get_ident()), workers=4)
    assert threads == {me}


def test_the_first_failing_task_in_order_wins():
    """Task 30 fails first in time, task 10 later: 10's error is raised, 0 to 9 committed."""
    events = {10: threading.Event()}
    committed, commit = _record()

    def work(i):
        if i == 30:
            events[10].set()
            raise ValueError("task 30")
        if i == 10:
            assert events[10].wait(5), "task 30 never failed"
            raise ValueError("task 10")
        return i

    with pytest.raises(ValueError, match="task 10"):
        run(range(100), work, commit, workers=4, window=40)
    assert [t for t, _ in committed] == list(range(10))


def test_tasks_after_a_failure_are_skipped():
    started = []

    def work(i):
        started.append(i)
        if i == 5:
            raise ValueError("five")
        time.sleep(0.002)
        return i

    with pytest.raises(ValueError, match="five"):
        run(range(1000), work, lambda *_: None, workers=4, window=8)
    assert max(started) <= 5 + 8


def test_a_source_error_comes_after_the_tasks_before_it():
    committed, commit = _record()

    def tasks():
        yield from range(7)
        raise OSError("source broke")

    with pytest.raises(OSError, match="source broke"):
        run(tasks(), lambda i: i, commit, workers=4)
    assert [t for t, _ in committed] == list(range(7))


def test_a_worker_error_before_a_source_error_wins():
    def tasks():
        yield from range(7)
        raise OSError("source broke")

    def work(i):
        if i == 3:
            raise ValueError("three")
        return i

    with pytest.raises(ValueError, match="three"):
        run(tasks(), work, lambda *_: None, workers=4)


@pytest.mark.parametrize("error", [RuntimeError("sink full"), KeyboardInterrupt()])
def test_a_commit_error_stops_the_run(error):
    started = []

    def work(i):
        started.append(i)
        time.sleep(0.001)
        return i

    def commit(task, _result):
        if task == 3:
            raise error

    with pytest.raises(type(error)):
        run(range(10_000), work, commit, workers=4, window=8)
    assert len(started) <= 3 + 8 + 4


@pytest.mark.parametrize("fail", [False, True])
def test_no_worker_thread_survives_a_call(fail):
    def work(i):
        if fail and i == 20:
            raise ValueError("x")
        return i

    with contextlib.suppress(ValueError):
        run(range(100), work, lambda *_: None, workers=6)
    assert not [t for t in threading.enumerate() if t.name.startswith("fse-worker")]


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "2", None])
def test_workers_and_window_are_checked(value):
    with pytest.raises(ValueError, match="workers must be a positive int"):
        run([], lambda i: i, lambda *_: None, workers=value)
    if value is not None:  # None is the default window
        with pytest.raises(ValueError, match="window must be a positive int"):
            run([], lambda i: i, lambda *_: None, workers=2, window=value)


def test_numpy_integers_are_accepted():
    committed, commit = _record()
    run(range(3), lambda i: i, commit, workers=np.int64(2), window=np.int32(2))
    assert len(committed) == 3


def test_an_empty_source():
    run([], lambda i: i, lambda *_: None, workers=4)


def test_nothing_more_is_read_after_a_known_failure(monkeypatch):
    """A stream whose next item would block: after a failure the executor stops reading."""
    registered = threading.Event()
    real = _executor._Run._work

    def work_and_signal(self, sequence, task):
        try:
            return real(self, sequence, task)
        except BaseException:
            registered.set()  # failed_at is set now
            raise

    monkeypatch.setattr(_executor._Run, "_work", work_and_signal)
    reads = []

    def tasks():
        yield 0
        reads.append(1)
        assert registered.wait(5)
        yield 1
        reads.append(2)
        raise AssertionError("read after the failure was known")

    def work(i):
        if i == 0:
            raise ValueError("zero")
        return i

    with pytest.raises(ValueError, match="zero"):
        run(tasks(), work, lambda *_: None, workers=4, window=8)
    assert 2 not in reads  # at most the read already under way when task 0 failed


def test_an_error_of_reserve_stops_the_run_at_once():
    """Task 3 is reserved while task 2 still runs; its error wins over task 2's later one."""
    reserving = threading.Event()

    def reserve(task):
        if task == 3:
            reserving.set()
            raise MemoryError("no room for the rows")
        return task

    def work(i):
        if i == 2:
            assert reserving.wait(5)
            raise ValueError("two")
        return i

    committed = []
    with pytest.raises(MemoryError, match="no room"):
        run(range(10), work, lambda t, _r: committed.append(t), workers=4, reserve=reserve)
    assert 3 not in committed
    reserving.set()
    with pytest.raises(ValueError, match="two"):  # inline, task 2 fails before 3 is reserved
        run(range(10), work, lambda *_: None, workers=1, reserve=reserve)
