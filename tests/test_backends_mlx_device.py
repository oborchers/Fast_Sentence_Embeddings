"""fse._backends._mlx._device: the device thread groups waiting calls, keeps errors to
their jobs and leaves no thread behind.

A fake MLX stands in for the GPU, so these run on every machine; the last tests drive the
real backend from many threads where it can run.
"""

import threading
import time

import numpy as np
import pytest

from fse._backends import CpuBackend, PoolInput
from fse._backends._mlx import _kernels
from fse._contract import Batch
from support.conformance import mlx_module, require_mlx

_device = mlx_module("fse._backends._mlx._device")
MlxBackend = mlx_module("fse._backends._mlx._backend").MlxBackend
Device, Job = _device.Device, _device.Job
F32 = np.float32


class FakeMx:
    gpu = "gpu"

    def __init__(self):
        self.evaluated = []
        self.fail = None

    def new_thread_unsafe_stream(self, device):
        return f"stream on {device}"

    def eval(self, *arrays):
        if self.fail is not None:
            raise self.fail
        self.evaluated.append(arrays)


@pytest.fixture
def mx(monkeypatch):
    fake = FakeMx()
    monkeypatch.setattr(_device._load, "core", lambda: fake)
    return fake


def _threads():
    return [t for t in threading.enumerate() if t.name == "fse-mlx"]


def _job(device, value, hold=None):
    """A job that returns ``10 * value``; with ``hold`` (entered, release), its launch
    signals that it runs and waits to be released, so others queue behind it."""

    def launch(_mx, stream):
        assert stream == "stream on gpu"
        if hold is not None:
            hold[0].set()
            hold[1].wait()
        return (value,)

    return device.run(launch, lambda arrays: arrays[0] * 10)


def test_a_job_runs_and_no_thread_remains(mx):
    device = Device()
    assert _job(device, 4) == 40
    assert mx.evaluated == [(4,)]
    assert _threads() == []
    assert _job(device, 5) == 50  # a new thread after the first ended
    assert _threads() == []


def _concurrently(device, values, call=_job):
    """``call(device, v)`` for every value, each from its own thread; the others queue
    while the first holds the device thread. The results and errors by value."""
    results, errors = {}, {}
    hold = (threading.Event(), threading.Event())

    def run(v):
        try:
            results[v] = call(device, v, hold if v == values[0] else None)
        except Exception as err:
            errors[v] = err

    threads = [threading.Thread(target=run, args=(v,)) for v in values]
    threads[0].start()
    hold[0].wait()
    for t in threads[1:]:
        t.start()
    while len(device._queue) < len(values) - 1:
        time.sleep(0.001)
    hold[1].set()
    for t in threads:
        t.join()
    return results, errors


def test_waiting_calls_share_one_evaluation(mx):
    device = Device(group_tasks=4)
    results, errors = _concurrently(device, list(range(1, 10)))
    assert errors == {}
    assert results == {v: 10 * v for v in range(1, 10)}
    # The first job alone, then the eight that waited in groups of at most four.
    assert [len(a) for a in mx.evaluated] == [1, 4, 4]
    assert (device.evaluations, device.launches) == (3, 9)
    assert _threads() == []


@pytest.mark.usefixtures("mx")
def test_a_group_holds_at_most_its_tokens():
    device = Device(group_tasks=8, group_tokens=10)
    jobs = [
        Job(lambda _m, _s, v=v: (v,), lambda a: a, tokens) for v, tokens in enumerate([6, 4, 2, 9])
    ]
    device._queue.extend(jobs)
    assert device._take() == jobs[:2]
    assert device._take() == jobs[2:3]  # 2 + 9 is over 10
    assert device._take() == jobs[3:]  # the first job of a group always goes
    assert device._take() == []


def test_an_error_in_one_launch_or_finish_fails_only_that_job(mx):
    def call(device, value, hold):
        if value == 3:
            return device.run(lambda _m, _s: 1 / 0, lambda a: a)
        if value == 4:
            return device.run(lambda _m, _s: (4,), lambda _a: [][1])
        return _job(device, value, hold)

    device = Device()
    results, errors = _concurrently(device, [1, 2, 3, 4, 5], call)
    assert results == {1: 10, 2: 20, 5: 50}
    assert isinstance(errors[3], ZeroDivisionError)
    assert isinstance(errors[4], IndexError)
    assert [len(a) for a in mx.evaluated] == [1, 3]  # the group of 2 to 5 without 3


def test_an_error_of_the_evaluation_fails_its_group(mx):
    device = Device()
    mx.fail = RuntimeError("[metal] command buffer failed")
    with pytest.raises(RuntimeError, match="command buffer failed"):
        _job(device, 1)
    mx.fail = None
    assert _job(device, 2) == 20  # the next group runs again
    assert _threads() == []


def test_a_failure_of_the_thread_reaches_every_waiter(monkeypatch):
    def broken():
        raise MemoryError("no device")

    monkeypatch.setattr(_device._load, "core", broken)
    with pytest.raises(MemoryError, match="no device"):
        _job(Device(), 1)
    assert _threads() == []


class _Interrupted:
    """A done event whose first wait is interrupted (KeyboardInterrupt in the caller)."""

    def __init__(self, job):
        self.job, self.waits = job, 0

    def wait(self):
        self.waits += 1
        if self.waits == 1:
            raise KeyboardInterrupt


@pytest.mark.usefixtures("mx")
def test_an_interrupted_waiter_takes_its_queued_job_out():
    device = Device()
    job = Job(lambda _m, _s: (1,), lambda a: a, 0)
    job.done = _Interrupted(job)
    device._queue.append(job)
    with pytest.raises(KeyboardInterrupt):
        device._wait(job)
    assert list(device._queue) == []
    assert job.done.waits == 1


@pytest.mark.usefixtures("mx")
def test_an_interrupted_waiter_waits_for_its_running_job():
    device = Device()
    job = Job(lambda _m, _s: (1,), lambda a: a, 0)
    job.state = "running"
    job.done = _Interrupted(job)
    with pytest.raises(KeyboardInterrupt):
        device._wait(job)
    assert job.done.waits == 2  # waited until the device thread was done with it


@pytest.mark.usefixtures("mx")
def test_the_canary_runs_once_and_its_error_stays(monkeypatch):
    calls = []
    monkeypatch.setattr(_device._kernels, "canary", lambda _m, _s: calls.append(1))
    device = Device()
    device.ready()
    device.ready()
    assert calls == [1]

    def broken(_m, _s):
        raise ValueError("the canary failed")

    monkeypatch.setattr(_device._kernels, "canary", broken)
    other = Device()
    for _ in range(2):
        with pytest.raises(ValueError, match="the canary failed"):
            other.ready()


# --- The real backend from many threads -------------------------------------------------------


@pytest.fixture(scope="module")
def backend():
    require_mlx()
    return MlxBackend(CpuBackend())


def test_many_threads_get_their_own_rows(backend):
    rng = np.random.default_rng(0)
    pool = PoolInput(rng.standard_normal((50, 33)).astype(F32), rng.random(50).astype(F32))
    prepared, host = backend.prepare(pool, ()), CpuBackend()
    host_prepared = host.prepare(pool, ())
    batches = []
    for k in range(64):
        lengths = rng.integers(0, 9, int(rng.integers(1, 20)))
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        batches.append(Batch(k * 100, offsets, rng.integers(0, 50, offsets[-1]).astype(np.uint32)))
    got = [np.empty((b.n_sentences, 33), F32) for b in batches]
    before = _device.DEVICE.evaluations
    threads = [
        threading.Thread(
            target=lambda i=i: [
                backend.rows(batches[j], prepared, got[j]) for j in range(i, 64, 16)
            ]
        )
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for b, rows in zip(batches, got, strict=True):
        want = np.empty_like(rows)
        host.rows(b, host_prepared, want)
        assert rows.tobytes() == want.tobytes()
    assert _device.DEVICE.evaluations - before <= 64
    assert _threads() == []


def test_rows_of_one_task_launch_one_kernel(backend):
    pool = PoolInput(np.ones((3, 4), F32), np.ones(3, F32))
    prepared = backend.prepare(pool, ())
    before = (_device.DEVICE.evaluations, _device.DEVICE.launches)
    backend.rows(
        Batch(0, np.array([0, 2], np.int64), np.array([0, 1], np.uint32)),
        prepared,
        np.empty((1, 4), F32),
    )
    assert (_device.DEVICE.evaluations, _device.DEVICE.launches) == (before[0] + 1, before[1] + 1)
    assert isinstance(prepared.state.table, _kernels.Table)
