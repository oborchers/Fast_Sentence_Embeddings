"""fse.Pool: the threads a run of tasks executes on (docs/architecture/runtime.md, "Runtime",
Pool): the object, runs on a pool, and its life (close, collection, fork)."""

import gc
import os
import sys
import threading
import warnings

import pytest

from fse._runtime import Pool, Workers, _pool, workers_of
from fse._runtime._executor import run
from support.pool import fse_threads, make_encoder, no_fse_threads_left, settled  # noqa: F401

# --- the object -------------------------------------------------------------------------


def test_a_pool_has_workers_and_a_task_limit():
    with Pool(3) as pool:
        assert (pool.workers, pool.max_tasks, pool.closed) == (3, 6, False)
        assert "workers=3" in repr(pool)
    assert pool.closed
    assert Pool(2, max_tasks=5).max_tasks == 5
    assert Pool().workers >= 1


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_workers_of_a_pool_must_be_a_positive_int(workers):
    with pytest.raises(ValueError, match="workers must be a positive int"):
        Pool(workers)


def test_the_arguments_of_a_pool_are_checked():
    with pytest.raises(ValueError, match="max_tasks must be a positive int"):
        Pool(2, max_tasks=0)
    with pytest.raises(TypeError, match="start must be a bool"):
        Pool(2, start=1)  # type: ignore[arg-type]


def test_a_pool_starts_no_thread_before_a_task_and_all_at_once_with_start():
    with Pool(3) as lazy:
        assert fse_threads() == []
        assert lazy.max_tasks == 6
    with Pool(3, start=True):
        assert len(fse_threads()) == 3
    with Pool(1, start=True):
        assert len(fse_threads()) == 1


def test_a_closed_pool_refuses_a_call_and_close_twice_is_fine():
    pool = Pool(2)
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="the pool is closed"):
        run(range(3), lambda i: i, lambda *_: None, workers=2, pool=pool)
    with pytest.raises(RuntimeError, match="the pool is closed"):
        make_encoder().encode(["w1 w2"], pool=pool)


def test_workers_of_a_call_may_not_exceed_the_pool():
    with Pool(2) as pool:
        assert workers_of(None, pool) == Workers(2, pool)
        assert workers_of(1, pool) == Workers(1, pool)
        with pytest.raises(ValueError, match=r"workers \(3\) must not exceed pool.workers \(2\)"):
            workers_of(3, pool)
        with pytest.raises(ValueError, match="workers must be a positive int"):
            workers_of(0, pool)
        with pytest.raises(TypeError, match=r"pool must be an fse\.Pool"):
            workers_of(1, object())
    assert workers_of(5) == Workers(5, None)
    assert workers_of(Workers(4, None)) == Workers(4, None)


# --- runs on a pool ---------------------------------------------------------------------


def test_a_pool_serves_one_call_after_another_and_keeps_its_threads():
    with Pool(3) as pool:
        for _ in range(3):
            done: list = []
            run(
                range(50),
                lambda i: i * i,
                lambda t, r, d=done: d.append((t, r)),
                workers=3,
                pool=pool,
            )
            assert done == [(i, i * i) for i in range(50)]
        assert 1 <= len(fse_threads()) <= 3  # the threads of the pool outlive the calls
    assert settled(0)


def test_a_call_without_a_pool_leaves_no_thread():
    run(range(50), lambda i: i, lambda *_: None, workers=4)
    assert settled(0)


def test_a_call_runs_at_most_workers_tasks_at_once_on_a_larger_pool():
    lock, state = threading.Lock(), {"now": 0, "most": 0}

    def work(i):
        with lock:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        threading.Event().wait(0.002)
        with lock:
            state["now"] -= 1
        return i

    with Pool(6, start=True) as pool:
        done = []
        run(range(60), work, lambda _t, r: done.append(r), workers=2, pool=pool)
    assert done == list(range(60))
    assert state["most"] <= 2


def test_two_calls_at_once_share_the_pool_and_both_finish_in_order():
    results = {}

    def call(name):
        done = []
        run(range(80), lambda i: (name, i), lambda _t, r: done.append(r), workers=2, pool=pool)
        results[name] = done

    with Pool(2) as pool:
        threads = [threading.Thread(target=call, args=(k,)) for k in "abc"]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
    assert all(not t.is_alive() for t in threads)
    for name in "abc":
        assert results[name] == [(name, i) for i in range(80)]


def test_the_task_limit_of_the_pool_holds_over_calls_and_does_not_deadlock():
    lock, state = threading.Lock(), {"held": 0, "most": 0}

    def tasks(n):
        for i in range(n):
            with lock:  # the call has its place before it reads the next task
                state["held"] += 1
                state["most"] = max(state["most"], state["held"])
            yield i

    def committed(*_):
        with lock:
            state["held"] -= 1

    def call():
        run(tasks(40), lambda i: i, committed, workers=3, pool=pool)

    with Pool(3, max_tasks=3) as pool:
        threads = [threading.Thread(target=call) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert all(not t.is_alive() for t in threads), "the calls deadlocked"
        assert state["most"] <= pool.max_tasks


def test_a_failure_in_one_call_does_not_stop_the_call_beside_it():
    other_done = []
    started, release = threading.Event(), threading.Event()

    def slow(i):
        started.set()
        assert release.wait(10)
        return i

    def other():
        run(range(6), slow, lambda _t, r: other_done.append(r), workers=2, pool=pool)

    def failing(i):
        if i == 2:
            raise ValueError("boom")
        return i

    with Pool(4) as pool:
        thread = threading.Thread(target=other)
        thread.start()
        assert started.wait(10)
        with pytest.raises(ValueError, match="boom"):
            run(range(20), failing, lambda *_: None, workers=2, pool=pool)
        release.set()
        thread.join(30)
    assert other_done == list(range(6))


def test_an_error_of_commit_cancels_the_calls_tasks_and_waits_for_the_running_ones():
    seen = {"started": 0, "finished": 0}
    lock, running, release = threading.Lock(), threading.Event(), threading.Event()

    def work(i):
        with lock:
            seen["started"] += 1
        if i >= 2:
            running.set()
            assert release.wait(10)
        with lock:
            seen["finished"] += 1
        return i

    def commit(task, _result):
        if task == 1:
            assert running.wait(10)  # a task is running now
            release.set()
            raise RuntimeError("commit failed")

    with Pool(2) as pool:
        with pytest.raises(RuntimeError, match="commit failed"):
            run(range(100), work, commit, workers=2, pool=pool)
        assert seen["started"] == seen["finished"]  # nothing of the call runs on
        assert seen["started"] < 100  # the tasks that had not started were cancelled
        assert pool.try_acquire()  # every place the call held is back
        pool.release()
        done = []
        run(range(5), lambda i: i, lambda _t, r: done.append(r), workers=2, pool=pool)
        assert done == list(range(5))  # the pool still works


def test_places_are_given_back_after_a_failed_task():
    with Pool(2, max_tasks=4) as pool:

        def work(i):
            if i == 5:
                raise ValueError("no")
            return i

        for _ in range(3):
            with pytest.raises(ValueError, match="no"):
                run(
                    range(30),
                    work,
                    lambda *_: None,
                    workers=2,
                    pool=pool,
                )
        for _ in range(4):
            assert pool.try_acquire()
        assert not pool.try_acquire()
        for _ in range(4):
            pool.release()


# --- life: close, collection, streams, fork ---------------------------------------------


def test_a_pool_that_is_dropped_ends_its_threads():
    pool = Pool(3, start=True)
    assert len(fse_threads()) == 3
    del pool
    gc.collect()
    assert settled(0)


def _watch_shutdown(monkeypatch):
    """An event that is set when a thread enters ``shutdown`` of an executor of a pool, and
    a counter of those entries."""
    entered, count = threading.Event(), []
    real = _pool.Pool._new_executor

    def new(self):
        executor = real(self)
        shutdown = executor.shutdown

        def watched(*args, **kwargs):
            count.append(1)
            entered.set()
            return shutdown(*args, **kwargs)

        executor.shutdown = watched  # type: ignore[method-assign]
        return executor

    monkeypatch.setattr(_pool.Pool, "_new_executor", new)
    return entered, count


def test_every_close_waits_for_a_running_task(monkeypatch):
    entered, count = _watch_shutdown(monkeypatch)
    started, release, finished = threading.Event(), threading.Event(), []

    def work(i):
        started.set()
        assert release.wait(10)
        finished.append(i)

    pool = Pool(2)
    caller = threading.Thread(
        target=run, args=(range(1), work, lambda *_: None), kwargs={"workers": 2, "pool": pool}
    )
    caller.start()
    assert started.wait(10)
    closers = [threading.Thread(target=pool.close) for _ in range(2)]
    try:
        closers[0].start()
        assert entered.wait(10)  # the first close is in shutdown, which waits for the task
        closers[1].start()
        while len(count) < 2 and closers[1].is_alive():  # the second reached shutdown, or is
            threading.Event().wait(0.001)  # blocked on the lock of the pool for a moment
        assert all(c.is_alive() for c in closers)  # neither returned: the task still runs
    finally:
        release.set()
        for t in (*closers, caller):
            t.join(10)
    assert finished == [0]
    assert pool.closed


@pytest.mark.skipif(not hasattr(os, "fork") or sys.platform == "win32", reason="needs fork")
def test_a_child_of_fork_starts_the_threads_again():
    pool = Pool(2, start=True)
    done = []
    run(range(4), lambda i: i, lambda _t, r: done.append(r), workers=2, pool=pool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads (3.12 and later)
        pid = os.fork()
    if pid == 0:  # the child: the threads of the parent do not exist here
        code = 1
        try:
            got = []
            run(range(30), lambda i: i + 1, lambda _t, r: got.append(r), workers=2, pool=pool)
            code = 0 if got == [i + 1 for i in range(30)] else 2
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    pool.close()


def test_the_fork_hook_forgets_the_threads_and_the_places():
    pool = Pool(2, start=True)
    assert pool.try_acquire()  # a place held by a call that does not exist in the child
    pool._after_fork()
    assert pool._executor is None
    done = []
    run(range(5), lambda i: i, lambda _t, r: done.append(r), workers=2, pool=pool)
    assert done == list(range(5))
    pool._executor = None
    settled(0)


def _raised(call):
    try:
        call()
    except BaseException as error:
        return error
    return None


def test_a_call_inside_a_sink_or_a_task_of_the_same_pool_is_refused_not_left_to_hang():
    with Pool(2, max_tasks=1) as pool:

        def inner(*_):
            run(range(3), lambda i: i, lambda *_: None, workers=2, pool=pool)

        calls = (
            lambda: run(range(4), lambda i: i, inner, workers=2, pool=pool),
            lambda: run(range(4), inner, lambda *_: None, workers=2, pool=pool),
        )
        for call in calls:
            outcome: list = []
            thread = threading.Thread(target=lambda c=call, o=outcome: o.append(_raised(c)))
            thread.start()
            thread.join(30)
            assert not thread.is_alive(), "the nested call hung"
            assert isinstance(outcome[0], RuntimeError)
            assert "inside a call or a task of the same pool" in str(outcome[0])
        done = []
        run(range(3), lambda i: i, lambda _t, r: done.append(r), workers=2, pool=pool)
        assert done == [0, 1, 2]  # the pool is intact afterwards


def test_a_finished_task_is_dropped_with_its_rows_and_not_kept_until_the_call_ends(monkeypatch):
    from fse._runtime import _executor

    runs, most = [], [0]
    real = _executor._Run.execute

    def execute(self, pool):
        runs.append(self)
        return real(self, pool)

    monkeypatch.setattr(_executor._Run, "execute", execute)

    def commit(_task, _result):
        most[0] = max(most[0], len(runs[0].futures))

    with Pool(3) as own:
        run(range(500), lambda i: [i] * 10, commit, workers=3, pool=own)
    assert runs[0].futures == set()
    assert most[0] <= 2 * 3  # at most the window, never the 500 tasks of the call


def test_the_places_of_tasks_that_still_run_return_when_those_tasks_end(monkeypatch):
    from fse._runtime import _executor

    started, release = threading.Event(), threading.Event()

    def work(i):
        if i == 0:
            return i
        started.set()
        assert release.wait(10)
        return i

    def interrupted(_self):  # a second interrupt that ends the wait for the running tasks
        raise KeyboardInterrupt

    monkeypatch.setattr(_executor._Run, "_wait_for_own", interrupted)

    def commit(*_):
        assert started.wait(10)  # a task runs: it holds a place of the pool
        raise RuntimeError("stop")

    with Pool(2, max_tasks=2) as pool:
        with pytest.raises(KeyboardInterrupt):
            run(range(10), work, commit, workers=2, pool=pool)
        assert pool.try_acquire()  # the place of the committed task is free
        assert not pool.try_acquire()  # the place of the running task is not
        pool.release()
        release.set()
        pool.acquire()  # it returns when that task ends
        pool.release()
