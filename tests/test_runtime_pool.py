"""fse.Pool: the threads a run of tasks executes on, and what the public methods do with
one (docs/architecture/runtime.md, "Runtime", Pool)."""

import gc
import os
import sys
import threading
import warnings

import numpy as np
import pytest

import fse
from fse import Classifier, Encoder, Heads
from fse._runtime import Pool, Workers, _pool, workers_of
from fse._runtime._executor import run
from support.runtime import small_tasks, texts, word_vectors

F32 = np.float32


def _fse_threads():
    return [t for t in threading.enumerate() if t.name.startswith("fse-worker")]


def _settled(count=0, timeout=10):
    """Whether the number of fse worker threads is ``count`` (idle threads end a moment
    after their executor does)."""
    deadline = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if len(_fse_threads()) == count:
            return True
        deadline.wait(0.01)
    return len(_fse_threads()) == count


@pytest.fixture(autouse=True)
def _no_threads_left():
    assert _settled(0), "a test before left fse worker threads"
    yield
    gc.collect()
    assert _settled(0), "this test left fse worker threads"


def _encoder():
    return Encoder.from_word_vectors(word_vectors(dim=12), normalize=False)


def _head(encoder, index=0):
    rng = np.random.default_rng(17 + index)
    return Classifier.from_arrays(
        encoder,
        rng.standard_normal((4, 12)).astype(F32),
        rng.standard_normal(4).astype(F32),
        labels=list("abcd"),
        link="softmax",
    )


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
        assert _fse_threads() == []
        assert lazy.max_tasks == 6
    with Pool(3, start=True):
        assert len(_fse_threads()) == 3
    with Pool(1, start=True):
        assert len(_fse_threads()) == 1


def test_a_closed_pool_refuses_a_call_and_close_twice_is_fine():
    pool = Pool(2)
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="the pool is closed"):
        run(range(3), lambda i: i, lambda *_: None, workers=2, pool=pool)
    with pytest.raises(RuntimeError, match="the pool is closed"):
        _encoder().encode(["w1 w2"], pool=pool)


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
        assert 1 <= len(_fse_threads()) <= 3  # the threads of the pool outlive the calls
    assert _settled(0)


def test_a_call_without_a_pool_leaves_no_thread():
    run(range(50), lambda i: i, lambda *_: None, workers=4)
    assert _settled(0)


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
            with lock:
                state["held"] += 1
                state["most"] = max(state["most"], state["held"])
            yield i

    def call():
        run(
            tasks(40),
            lambda i: i,
            lambda *_: state.__setitem__("held", state["held"] - 1),
            workers=3,
            pool=pool,
        )

    with Pool(3, max_tasks=3) as pool:
        threads = [threading.Thread(target=call) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert all(not t.is_alive() for t in threads), "the calls deadlocked"
    assert state["most"] <= 3 + 3  # three places, and each call took one before the limit read


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
    lock = threading.Lock()

    def work(i):
        with lock:
            seen["started"] += 1
        threading.Event().wait(0.01)
        with lock:
            seen["finished"] += 1
        return i

    def commit(task, _result):
        if task == 3:
            raise RuntimeError("commit failed")

    with Pool(2) as pool:
        with pytest.raises(RuntimeError, match="commit failed"):
            run(range(100), work, commit, workers=2, pool=pool)
        assert seen["started"] == seen["finished"]  # nothing of the call runs on
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
    assert len(_fse_threads()) == 3
    del pool
    gc.collect()
    assert _settled(0)


def test_close_waits_for_a_running_task():
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
    closer = threading.Thread(target=pool.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()  # close waits
    release.set()
    closer.join(10)
    caller.join(10)
    assert finished == [0]


def _batches(n=8, size=2000):
    data = texts(n * size)
    return [data[k * size : (k + 1) * size] for k in range(n)]


def test_a_stream_runs_every_batch_on_one_pool_and_closes_it(monkeypatch):
    small_tasks(monkeypatch)
    made = []
    real = _pool.Pool._new_executor
    monkeypatch.setattr(_pool.Pool, "_new_executor", lambda self: made.append(1) or real(self))
    out = list(_encoder().encode_batches(_batches(), workers=3))
    assert len(out) == 8
    assert len(made) == 1  # one executor for the whole iteration, not one per batch
    assert _settled(0)


@pytest.mark.parametrize("how", ["break", "close", "collect", "error"])
def test_a_stream_ends_its_threads_when_it_stops_early(monkeypatch, how):
    small_tasks(monkeypatch)
    encoder = _encoder()
    items = _batches()
    if how == "error":
        items[2] = ["w1", 7]  # type: ignore[list-item]
        with pytest.raises((TypeError, ValueError)):
            list(encoder.encode_batches(items, workers=3))
    else:
        stream = encoder.encode_batches(items, workers=3)
        next(stream)
        if how == "close":
            stream.close()
        elif how == "collect":
            del stream
        else:
            stream = None
    gc.collect()
    assert _settled(0)


def test_a_stream_on_the_callers_pool_leaves_it_open():
    with Pool(2) as pool:
        out = list(_encoder().encode_batches(_batches(2, 300), pool=pool))
        assert len(out) == 2
        assert not pool.closed
        _encoder().encode(["w1 w2"], pool=pool)


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
    _settled(0)


# --- the public methods -----------------------------------------------------------------


def test_encode_on_a_pool_has_the_bits_of_encode_without_one():
    encoder, data = _encoder(), texts(3000)
    want = encoder.encode(data, workers=3)
    with Pool(3) as pool:
        assert np.array_equal(encoder.encode(data, pool=pool), want)
        assert np.array_equal(encoder.encode(data, workers=2, pool=pool), want)
        got = encoder.encode(data, pool=pool, dtype="int8")
        assert np.array_equal(got.codes, encoder.encode(data, dtype="int8").codes)


def test_encode_batches_on_a_pool_has_the_bits_of_encode_batches_without_one():
    encoder, items = _encoder(), _batches(4, 500)
    want = [b.rows for b in encoder.encode_batches(items, workers=3)]
    with Pool(3) as pool:
        got = [b.rows for b in encoder.encode_batches(items, pool=pool)]
    assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))


def test_heads_and_classifier_on_a_pool_have_the_bits_of_the_calls_without_one():
    encoder, data = _encoder(), texts(3000)
    heads = Heads(a=_head(encoder, 0), b=_head(encoder, 1))
    one = _head(encoder, 2)
    want_group = heads.predict(data, output="proba", workers=3)
    want_one = one.predict_proba(data, workers=3)
    with Pool(3) as pool:
        got = heads.predict(data, output="proba", pool=pool)
        for name in want_group:
            assert np.array_equal(got[name], want_group[name])
        assert np.array_equal(one.predict_proba(data, pool=pool), want_one)
        assert np.array_equal(one.predict(data, pool=pool), one.predict(data, workers=3))
        batches = list(one.predict_batches([data[:1500], data[1500:]], pool=pool))
        assert np.array_equal(
            np.concatenate([b.outputs for b in batches]), one.predict(data, workers=3)
        )


def test_every_text_method_takes_pool_after_workers_and_checks_it_before_reading():
    encoder = _encoder()
    head = _head(encoder)
    for call in (
        lambda p: encoder.encode(["w1"], pool=p),
        lambda p: encoder.encode_batches([["w1"]], pool=p),
        lambda p: head.predict_proba(["w1"], pool=p),
        lambda p: Heads(a=head).predict(["w1"], output="proba", pool=p),
    ):
        with pytest.raises((TypeError, ValueError), match="pool"):
            call("not a pool")
    assert fse.Pool is Pool
