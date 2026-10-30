"""fse._backends._mlx._device: the lease of a job's arrays. The device thread lends the
launched arrays while the caller copies from them (``use``) and drops them itself, after
the caller returned them, also when the caller fails or is interrupted, and without
waiting for a slow caller before a group already queued. A fake MLX stands in for the GPU."""

import threading

import numpy as np
import pytest

from fse._backends._mlx import _device, _kernels
from fse._backends._mlx._device import Device
from support.fake_mlx import FakeMx
from support.waiting import until


class Tracked:
    """An array that records the thread that frees it."""

    def __init__(self, freed):
        self.freed = freed

    def __del__(self):
        self.freed.append(threading.current_thread().name)


@pytest.fixture
def device():
    mx = FakeMx(keep=False)
    return Device(lambda: mx, group_tasks=1)


def _threads():
    return [t for t in threading.enumerate() if t.name == "fse-mlx"]


def test_the_arrays_live_while_lent_and_die_in_the_device_thread(device):
    freed, seen = [], []

    def use(result):
        seen.append(list(freed))
        return result + 1

    got = device.run(lambda _mx, _s: (Tracked(freed),), lambda _arrays: 6, 0, use)
    assert got == 7
    assert seen == [[]]  # not freed while the caller used them
    assert freed == ["fse-mlx"]
    assert _threads() == []


@pytest.mark.parametrize("error", [ValueError("copy"), KeyboardInterrupt()])
def test_a_caller_that_fails_in_use_returns_the_lease(device, error):
    freed = []

    def use(_result):
        raise error

    with pytest.raises(type(error)):
        device.run(lambda _mx, _s: (Tracked(freed),), lambda _arrays: 1, 0, use)
    assert freed == ["fse-mlx"]
    assert _threads() == []


def test_a_slow_caller_does_not_hold_the_next_group(device):
    """Caller A uses its rows only after the job of caller B, the next group (queued while
    A's launch runs), is finished: the device thread runs B without waiting for A's
    lease, and drops A's arrays when A returns it. (A caller's use must not wait for
    another call to return: the last group's callers wait until every lease is back.)"""
    freed, b_done, results = [], threading.Event(), {}
    launching, b_queued = threading.Event(), threading.Event()

    def launch_a(_mx, _s):
        launching.set()
        b_queued.wait(10)
        return (Tracked(freed),)

    def slow(result):
        assert b_done.wait(10), "the device thread waited for the lease"
        return result

    def call_a():
        results["a"] = device.run(launch_a, lambda _a: "a", 0, slow)

    def finish_b(_arrays):
        b_done.set()  # in the device thread, while A still holds its lease
        return "b"

    def call_b():
        results["b"] = device.run(lambda _mx, _s: (Tracked(freed),), finish_b, 0, None)

    a = threading.Thread(target=call_a)
    a.start()
    launching.wait(10)
    b = threading.Thread(target=call_b)
    b.start()
    while len(device._queue) < 1:
        pass
    b_queued.set()
    b.join(10)
    a.join(10)
    assert results == {"a": "a", "b": "b"}
    assert freed == ["fse-mlx", "fse-mlx"]
    assert _threads() == []


def test_a_job_without_use_keeps_no_arrays(device):
    freed = []
    assert device.run(lambda _mx, _s: (Tracked(freed),), lambda _a: 3) == 3
    assert freed == ["fse-mlx"]


def test_a_failed_finish_lends_nothing(device):
    freed = []

    def finish(_arrays):
        raise RuntimeError("read")

    with pytest.raises(RuntimeError, match="read"):
        device.run(lambda _mx, _s: (Tracked(freed),), finish, 0, lambda r: r)
    assert freed == ["fse-mlx"]


class View:
    """What a finish gives: an object that keeps a launched array alive (as a numpy view
    keeps its MLX array)."""

    def __init__(self, array):
        self.array = array


@pytest.mark.parametrize("error", [ValueError("copy"), KeyboardInterrupt()])
def test_a_failing_use_drops_its_views_before_the_lease(device, error):
    """The traceback of the error keeps no view: the device thread frees the array."""
    freed = []

    def use(view):
        assert isinstance(view, View)  # a frame of use holding the view
        raise error

    with pytest.raises(type(error)) as caught:
        device.run(lambda _mx, _s: (Tracked(freed),), lambda arrays: View(arrays[0]), 0, use)
    assert freed == ["fse-mlx"]
    assert caught.value is error  # the error itself is kept
    assert _threads() == []


class Ended:
    """A device thread whose join is interrupted ``times`` times."""

    def __init__(self, times):
        self.times, self.joined = times, 0

    def join(self):
        if self.times:
            self.times -= 1
            raise KeyboardInterrupt
        self.joined += 1


@pytest.mark.parametrize("times", [1, 3])
def test_the_lease_comes_back_and_the_thread_is_joined_through_interrupts(times):
    job = _device.Job(lambda *_: (), lambda _: None, 0, True)
    job.ended = Ended(times)
    with pytest.raises(KeyboardInterrupt):
        _device._give_back(job)
    assert job.returned.is_set()
    assert job.ended.joined == 1


class Owned(bytearray):
    """A buffer that records the thread that frees it, as an MLX array owns its data."""

    def __init__(self, size, freed):
        super().__init__(size)
        self.freed = freed

    def __del__(self):
        self.freed.append(threading.current_thread().name)


@pytest.mark.parametrize("words", [1, 3])
def test_the_flags_of_a_view_keep_no_array_alive(device, words):
    """R5 of review 3: with one flag word per sentence the combined flags were a view of
    the lent flags; they are a copy, so the device thread frees the buffer."""
    freed = []

    def launch(_mx, _s):
        rows = np.frombuffer(Owned(4 * 2, freed), np.float32).reshape(2, 1)
        flags = np.frombuffer(Owned(4 * 2 * words, freed), np.uint32).reshape(2, words)
        bounds = np.frombuffer(Owned(4 * 4, freed), np.float32).reshape(2, 2)
        return rows, flags, bounds

    def use(viewed):
        _, flags, bounds = viewed
        return flags, bounds  # what the backend keeps after the lease

    flags, bounds = device.run(launch, _kernels.view, 0, use)
    assert freed == ["fse-mlx"] * 3
    assert flags.tolist() == [0, 0]
    assert bounds.shape == (2, 2)


class InterruptingLock:
    """The device's lock, interrupted ``times`` times when the abort takes it."""

    def __init__(self, lock, times):
        self.lock, self.times = lock, times
        self.armed = False

    def __enter__(self):
        if self.armed and self.times:
            self.times -= 1
            raise KeyboardInterrupt
        return self.lock.__enter__()

    def __exit__(self, *exc):
        return self.lock.__exit__(*exc)


@pytest.mark.parametrize("times", [1, 3])
def test_an_abort_interrupted_again_still_settles_the_job(device, monkeypatch, times):
    """R6 of review 3: through run(): the wait for the job is interrupted, then the abort
    is interrupted taking the lock; it still settles the job and returns its lease."""
    freed, started = [], threading.Event()
    lock = InterruptingLock(device._lock, times)
    monkeypatch.setattr(device, "_lock", lock)
    real_wait = threading.Event.wait

    class Interrupted(_device.Job):
        def __init__(self, *args):
            super().__init__(*args)
            self.done = Done()

    class Done(threading.Event):
        def wait(self, timeout=None):
            if not lock.armed:
                started.wait(10)
                lock.armed = True
                raise KeyboardInterrupt
            return real_wait(self, timeout)

    def launch(_mx, _s):
        started.set()
        return (Tracked(freed),)

    monkeypatch.setattr(_device, "Job", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        device.run(launch, lambda arrays: View(arrays[0]), 0, lambda v: v)
    assert lock.times == 0
    assert freed == ["fse-mlx"]
    assert _threads() == []


def _in_thread(function, timeout=10):
    """``function()`` in a thread; its result or error, and whether it ended in time."""
    out = {}

    def call():
        try:
            out["result"] = function()
        except BaseException as err:  # an interrupt raised in this thread
            out["error"] = err

    thread = threading.Thread(target=call)
    thread.start()
    thread.join(timeout)
    return out, not thread.is_alive()


def test_an_interrupt_before_the_first_enqueue_cancels_the_job(device, monkeypatch):
    """F3 of review 4: the job was never queued; the abort must not wait for it."""
    jobs = []

    def interrupted(job):
        jobs.append(job)
        raise KeyboardInterrupt  # as if taking the lock in _enqueue were interrupted

    monkeypatch.setattr(device, "_enqueue", interrupted)
    out, ended = _in_thread(lambda: device.run(lambda _m, _s: (), lambda _a: 1))
    assert ended, "the abort waited for a job that was never queued"
    assert isinstance(out["error"], KeyboardInterrupt)
    assert jobs[0].state == "cancelled"
    monkeypatch.undo()
    assert device.run(lambda _m, _s: (), lambda _a: 2) == 2  # the device goes on


def test_a_new_call_does_not_wait_under_the_lock_for_the_thread_before(device, monkeypatch):
    """F4 of review 4: caller A's job is done, and the device thread waits for A's lease;
    A is interrupted after its wait and must take the lock to settle, while caller B
    queues a job and starts a new thread. B must not hold the lock while it joins the
    thread before (which waits for A): both calls end."""
    b_queued = threading.Event()
    b_in_enqueue = []
    real_wait = threading.Event.wait

    class Done(threading.Event):
        interrupts = 1

        def wait(self, timeout=None):
            real_wait(self, timeout)
            if self.interrupts:  # once, after the job is done
                self.interrupts -= 1
                b_queued.wait(10)
                # B is in its enqueue: it has started the thread behind the one that waits
                # for A (``_thread`` was None until then).
                b_in_enqueue.append(until(lambda: device._thread is not None))
                raise KeyboardInterrupt
            return True

    class Interrupted(_device.Job):
        def __init__(self, *args):
            super().__init__(*args)
            self.done = Done()

    freed, results, plain = [], {}, _device.Job
    monkeypatch.setattr(_device, "Job", Interrupted)

    def call_a():
        try:
            device.run(
                lambda _m, _s: (Tracked(freed),), lambda arrays: View(arrays[0]), 0, lambda v: v
            )
        except KeyboardInterrupt:
            results["a"] = "interrupted"

    thread_a = threading.Thread(target=call_a)
    thread_b = threading.Thread(target=lambda: None)

    def call_b():
        b_queued.set()
        results["b"] = device.run(lambda _m, _s: (), lambda _a: "b")

    try:
        thread_a.start()
        assert until(_threads), "the device thread never started"
        # the thread found no more jobs: it waits for A
        assert until(lambda: device._thread is None), "the device thread never went idle"
        monkeypatch.setattr(_device, "Job", plain)
        thread_b = threading.Thread(target=call_b)
        thread_b.start()
    finally:
        b_queued.set()  # whatever happened, A's wait must end
        thread_a.join(10)
        thread_b.join(10)
    assert not thread_a.is_alive(), "the calls wait for each other"
    assert not thread_b.is_alive()
    assert b_in_enqueue == [True], "B was not in its enqueue when A was interrupted"
    assert results == {"a": "interrupted", "b": "b"}
    assert freed == ["fse-mlx"]
    assert _threads() == []
