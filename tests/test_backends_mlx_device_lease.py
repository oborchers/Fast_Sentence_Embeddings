"""fse._backends._mlx._device: the lease of a job's arrays. The device thread lends the
launched arrays while the caller copies from them (``use``) and drops them itself, after
the caller returned them, also when the caller fails or is interrupted, and without
waiting for a slow caller before a group already queued. A fake MLX stands in for the GPU."""

import threading

import pytest

from fse._backends._mlx import _device
from fse._backends._mlx._device import Device
from support.fake_mlx import FakeMx


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
