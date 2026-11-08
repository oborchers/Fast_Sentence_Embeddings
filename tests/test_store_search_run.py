"""The tasks of a search, score or rescore on an ``fse.Pool`` (``_store/_search_run``,
docs/architecture/store.md, "Concurrency"): task order, at most ``workers`` tasks at
once, a place per task until it has run, no deadlock with fewer places than threads or
with other calls, and a clean end of the call that leaves a borrowed pool working."""

import threading

import pytest

from fse._runtime import Pool, Workers
from fse._runtime._executor import run
from fse._store import _search_run
from support.pool import no_fse_threads_left, settled  # noqa: F401

TIMEOUT = 60


def free_places(pool):
    return pool._permits._value  # the count of its BoundedSemaphore


def squares(n, workers):
    return list(_search_run.run(lambda i: i * i, range(n), workers, "t"))


class Running:
    """Counts the tasks that run at once and the threads they run on."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = self.most = 0
        self.threads: set[str] = set()

    def task(self, fn):
        def run(item):
            with self.lock:
                self.now += 1
                self.most = max(self.most, self.now)
                self.threads.add(threading.current_thread().name)
            try:
                return fn(item)
            finally:
                with self.lock:
                    self.now -= 1

        return run


def test_without_a_pool_the_call_makes_and_ends_its_own_threads():
    names = Running()
    got = list(_search_run.run(names.task(lambda i: -i), range(20), 3, "fse-search"))
    assert got == [-i for i in range(20)]
    assert all(name.startswith("fse-search") for name in names.threads)
    assert settled(0)


def test_on_a_pool_the_tasks_run_in_order_on_its_threads_and_leave_it_working():
    with Pool(3) as pool:
        names = Running()
        got = list(_search_run.run(names.task(lambda i: i + 1), range(50), Workers(3, pool), "t"))
        assert got == list(range(1, 51))
        assert all(name.startswith("fse-worker") for name in names.threads)
        assert free_places(pool) == pool.max_tasks
        assert squares(9, Workers(2, pool)) == [i * i for i in range(9)]  # still serves calls
        assert not pool.closed
    assert settled(0)


def test_a_call_runs_at_most_its_workers_at_once_on_a_larger_pool():
    """Tasks meet in pairs at a barrier: a third task at once would raise its count."""
    barrier = threading.Barrier(2, timeout=TIMEOUT)
    count = Running()

    def met(i):
        barrier.wait()
        return i

    with Pool(6) as pool:
        got = list(_search_run.run(count.task(met), range(12), Workers(2, pool), "t"))
    assert got == list(range(12))
    assert count.most == 2


@pytest.mark.parametrize("max_tasks", [1, 2])
def test_fewer_places_than_threads_still_finish(max_tasks):
    count = Running()
    with Pool(4, max_tasks=max_tasks) as pool:
        got = list(_search_run.run(count.task(lambda i: i), range(40), Workers(4, pool), "t"))
        assert got == list(range(40))
        assert count.most <= max_tasks
        assert free_places(pool) == max_tasks


def test_many_calls_at_once_on_one_pool_with_few_places():
    """Eight threads search on a pool of two places and an encode-like run takes places too:
    every call finishes with its own results, and every place comes back."""
    with Pool(3, max_tasks=2) as pool:
        start = threading.Barrier(9, timeout=TIMEOUT)
        outcomes, errors = {}, []

        def search(n):
            try:
                start.wait()
                outcomes[n] = squares(30 + n, Workers(3, pool))
            except BaseException as error:  # raised below in the main thread
                errors.append(error)

        def commits():
            try:
                start.wait()
                done = []
                run(range(25), lambda i: i, lambda _t, r: done.append(r), workers=3, pool=pool)
                outcomes["run"] = done
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=search, args=(n,)) for n in range(8)]
        threads.append(threading.Thread(target=commits))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(TIMEOUT)
        assert not [t for t in threads if t.is_alive()], "a call hung on the pool"
        if errors:
            raise errors[0]
        for n in range(8):
            assert outcomes[n] == [i * i for i in range(30 + n)]
        assert outcomes["run"] == list(range(25))
        assert free_places(pool) == 2


def test_an_error_cancels_the_tasks_not_started_and_gives_every_place_back():
    started = []

    def fail_at_five(i):
        started.append(i)
        if i == 5:
            raise ValueError("task 5")
        return i

    with Pool(2) as pool:
        with pytest.raises(ValueError, match="task 5"):
            list(_search_run.run(fail_at_five, range(1000), Workers(2, pool), "t"))
        assert len(started) < 20  # the call stopped reading and submitting
        assert free_places(pool) == pool.max_tasks
        assert squares(5, Workers(2, pool)) == [0, 1, 4, 9, 16]


def test_an_error_of_the_tasks_gives_the_place_back():
    def tasks():
        yield from range(6)
        raise OSError("source")

    with Pool(2) as pool:
        with pytest.raises(OSError, match="source"):
            list(_search_run.run(lambda i: i, tasks(), Workers(2, pool), "t"))
        assert free_places(pool) == pool.max_tasks


@pytest.mark.parametrize("how", ["close", "collect"])
def test_a_stream_ended_early_waits_for_its_tasks_and_frees_the_pool(how):
    started = []

    def slow(i):
        started.append(i)
        return i

    with Pool(2) as pool:
        stream = _search_run.run(slow, range(1000), Workers(2, pool), "t")
        assert next(stream) == 0
        if how == "close":
            stream.close()
        else:
            del stream
        count = len(started)
        assert count < 20
        assert free_places(pool) == pool.max_tasks  # no task of the stream holds a place
        assert squares(4, Workers(2, pool)) == [0, 1, 4, 9]
        assert len(started) == count  # nothing of the stream ran after it ended


def test_a_call_between_two_blocks_of_a_stream_on_the_same_pool_finds_places():
    """A consumer that searches on the pool of its stream before it takes the next block:
    the blocks the stream holds take no place, so the inner call does not wait forever."""
    with Pool(2, max_tasks=2) as pool:
        stream = _search_run.run(lambda i: i, range(30), Workers(2, pool), "t")
        got = []
        for value in stream:
            got.append(value)
            assert squares(6, Workers(2, pool)) == [i * i for i in range(6)]
        assert got == list(range(30))
        assert free_places(pool) == 2


def test_a_call_on_the_pool_from_within_one_of_its_tasks_is_refused():
    with Pool(2) as pool:

        def inner(_i):
            return squares(4, Workers(2, pool))

        def one(_i):  # workers=1: the inner call runs in the task, on no place
            return squares(4, Workers(1, pool))

        with pytest.raises(RuntimeError, match="inside a call or a task of the same pool"):
            list(_search_run.run(inner, range(4), Workers(2, pool), "t"))
        assert list(_search_run.run(one, range(4), Workers(2, pool), "t")) == [[0, 1, 4, 9]] * 4
        assert free_places(pool) == pool.max_tasks


def test_one_task_runs_in_the_calling_thread_without_a_place_but_inside_the_pool():
    me = threading.current_thread().name
    seen = []

    def where(i):
        seen.append(threading.current_thread().name)
        return i

    with Pool(2, max_tasks=1) as pool:

        def nested(_i):
            return squares(1, Workers(2, pool))

        assert list(_search_run.run(where, [7], Workers(2, pool), "t")) == [7]
        assert seen == [me]
        assert free_places(pool) == 1
        with pytest.raises(RuntimeError, match="inside a call or a task of the same pool"):
            list(_search_run.run(nested, range(4), Workers(2, pool), "t"))


def test_a_closed_pool_is_refused_before_any_task():
    pool = Pool(2)
    pool.close()
    ran = []
    with pytest.raises(RuntimeError, match="the pool is closed"):
        list(_search_run.run(ran.append, range(10), Workers(2, pool), "t"))
    with pytest.raises(RuntimeError, match="the pool is closed"):
        list(_search_run.run(ran.append, [1], Workers(2, pool), "t"))
    assert ran == []


def test_a_call_waits_for_a_place_held_by_another_and_then_runs():
    """With the only place held by a running task of another call, a new call waits for a
    place (it has nothing of its own to take) and runs as soon as that task ends."""
    entered, release = threading.Event(), threading.Event()

    def held(i):
        entered.set()
        assert release.wait(TIMEOUT)
        return i

    with Pool(2, max_tasks=1) as pool:
        outcome = {}

        def first():
            outcome["a"] = list(_search_run.run(held, range(3), Workers(2, pool), "t"))

        def second():
            outcome["b"] = squares(4, Workers(2, pool))

        holder = threading.Thread(target=first)
        holder.start()
        assert entered.wait(TIMEOUT)
        assert free_places(pool) == 0  # the running task holds the only place
        waiter = threading.Thread(target=second)
        waiter.start()
        release.set()
        holder.join(TIMEOUT)
        waiter.join(TIMEOUT)
        assert outcome == {"a": [0, 1, 2], "b": [0, 1, 4, 9]}
        assert free_places(pool) == 1


def test_the_count_of_workers_in_a_log_is_an_int():
    with Pool(3) as pool:
        assert _search_run.count(Workers(2, pool)) == 2
    assert _search_run.count(5) == 5
