"""fse._backends._mlx._device: the device thread groups waiting calls, keeps errors to
their jobs, survives interrupts and failures to start, and leaves no thread behind.

A fake MLX stands in for the GPU, so these run on every machine, Linux too; the last
tests drive the real backend from many threads and calls where it can run.
"""

import threading
import time

import numpy as np
import pytest

from fse._backends import CpuBackend, PoolInput
from fse._backends._mlx import _device, _kernels
from fse._backends._mlx._device import Device, Job
from fse._contract import Batch
from fse._runtime import EncodeSpec, MemorySink, Pipeline, _tasks
from fse._text import StringSource
from support.conformance import mlx_module, require_mlx
from support.fake_mlx import FakeMx
from support.runtime import expected, texts, word_vectors

F32 = np.float32


@pytest.fixture
def mx():
    return FakeMx()


@pytest.fixture
def make(mx):
    """A device thread on the fake MLX."""
    return lambda **options: Device(lambda: mx, **options)


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


def test_a_job_runs_and_no_thread_remains(mx, make):
    device = make()
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


def test_waiting_calls_share_one_evaluation(mx, make):
    device = make(group_tasks=4)
    results, errors = _concurrently(device, list(range(1, 10)))
    assert errors == {}
    assert results == {v: 10 * v for v in range(1, 10)}
    # The first job alone, then the eight that waited in groups of at most four.
    assert [len(a) for a in mx.evaluated] == [1, 4, 4]
    assert (device.evaluations, device.launches) == (3, 9)
    assert _threads() == []


def test_a_group_holds_at_most_its_tokens(make):
    device = make(group_tasks=8, group_tokens=10)
    jobs = [
        Job(lambda _m, _s, v=v: (v,), lambda a: a, tokens) for v, tokens in enumerate([6, 4, 2, 9])
    ]
    device._queue.extend(jobs)
    assert device._take() == jobs[:2]
    assert device._take() == jobs[2:3]  # 2 + 9 is over 10
    assert device._take() == jobs[3:]  # the first job of a group always goes
    assert device._take() == []


def test_an_error_in_one_launch_or_finish_fails_only_that_job(mx, make):
    def call(device, value, hold):
        if value == 3:
            return device.run(lambda _m, _s: 1 / 0, lambda a: a)
        if value == 4:
            return device.run(lambda _m, _s: (4,), lambda _a: [][1])
        return _job(device, value, hold)

    device = make()
    results, errors = _concurrently(device, [1, 2, 3, 4, 5], call)
    assert results == {1: 10, 2: 20, 5: 50}
    assert isinstance(errors[3], ZeroDivisionError)
    assert isinstance(errors[4], IndexError)
    assert [len(a) for a in mx.evaluated] == [1, 3]  # the group of 2 to 5 without 3


def test_an_error_of_the_evaluation_fails_its_group(mx, make):
    device = make()
    mx.fail = RuntimeError("[metal] command buffer failed")
    with pytest.raises(RuntimeError, match="command buffer failed"):
        _job(device, 1)
    mx.fail = None
    assert _job(device, 2) == 20  # the next group runs again
    assert _threads() == []


def test_a_failure_of_the_thread_reaches_every_waiter():
    def broken():
        raise MemoryError("no device")

    with pytest.raises(MemoryError, match="no device"):
        _job(Device(broken), 1)
    assert _threads() == []


class _Interrupting(threading.Event):
    """The done event of a job whose first ``times`` waits are interrupted (as a
    KeyboardInterrupt reaches the calling thread), after ``when`` is set."""

    times = 1
    when: threading.Event | None = None

    def __init__(self):
        super().__init__()
        self.interrupts = 0

    def wait(self, timeout=None):
        if self.interrupts < self.times:
            if self.when is not None:
                self.when.wait()
            self.interrupts += 1
            raise KeyboardInterrupt
        return super().wait(timeout)


def _interrupting(monkeypatch, times=1, when=None):
    """Jobs made from here on have a done event whose first ``times`` waits are
    interrupted, after ``when`` is set."""

    class Interrupting(_Interrupting):
        pass

    Interrupting.times, Interrupting.when = times, when

    class InterruptedJob(Job):
        __slots__ = ()

        def __init__(self, *args):
            super().__init__(*args)
            self.done = Interrupting()

    monkeypatch.setattr(_device, "Job", InterruptedJob)


def test_an_interrupted_caller_takes_its_queued_job_out(make, monkeypatch):
    device = make()
    hold = (threading.Event(), threading.Event())
    first = threading.Thread(target=_job, args=(device, 1, hold))
    first.start()
    hold[0].wait()  # the first job holds the device thread
    queued, enqueue = [], device._enqueue
    monkeypatch.setattr(device, "_enqueue", lambda job: queued.append(job) or enqueue(job))
    _interrupting(monkeypatch)
    launched = []
    with pytest.raises(KeyboardInterrupt):
        device.run(lambda _m, _s: launched.append(1) or (2,), lambda a: a, 0)
    assert list(device._queue) == []
    assert queued[0].state == "cancelled"
    hold[1].set()
    first.join()
    assert launched == []  # the cancelled job never ran
    assert _threads() == []


@pytest.mark.parametrize("times", [1, 3], ids=["once", "repeatedly"])
def test_an_interrupted_caller_waits_until_its_rows_are_written(make, monkeypatch, times):
    """The device thread writes the rows in finish; the call returns only after it."""
    running, release = threading.Event(), threading.Event()
    _interrupting(monkeypatch, times, running)
    target = np.zeros(3, F32)

    def finish(_arrays):
        release.wait()
        target[...] = 7  # what a read into the rows does
        return None

    def launch(_m, _s):
        running.set()
        return (1,)

    timer = threading.Timer(0.05, release.set)
    timer.start()
    with pytest.raises(KeyboardInterrupt):
        make().run(launch, finish, 0)
    assert target.tolist() == [7, 7, 7]
    timer.join()
    assert _threads() == []


@pytest.mark.parametrize("error", [RuntimeError("can't start new thread"), KeyboardInterrupt()])
def test_a_device_thread_that_does_not_start_leaves_the_state_as_before(make, monkeypatch, error):
    device = make()

    def refuse(_thread):
        raise error

    with monkeypatch.context() as m:
        m.setattr(_device.threading.Thread, "start", refuse)
        with pytest.raises(type(error)):
            _job(device, 1)
    assert list(device._queue) == []
    assert device._thread is None
    assert _job(device, 2) == 20  # the next call starts one
    assert _threads() == []


def test_the_canary_runs_once_and_its_error_stays(make, monkeypatch):
    calls = []
    monkeypatch.setattr(_device._kernels, "canary", lambda _m, _s: calls.append(1))
    device = make()
    device.ready()
    device.ready()
    assert calls == [1]

    def broken(_m, _s):
        raise ValueError("the canary failed")

    monkeypatch.setattr(_device._kernels, "canary", broken)
    other = make()
    for _ in range(2):
        with pytest.raises(ValueError, match="the canary failed"):
            other.ready()


# --- The real backend from many threads and calls ------------------------------------------


@pytest.fixture(scope="module")
def mlx():
    require_mlx()
    return mlx_module("fse._backends._mlx._backend")


def _batches(rng, n=64):
    out = []
    for k in range(n):
        lengths = rng.integers(0, 9, int(rng.integers(1, 20)))
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        out.append(Batch(k * 100, offsets, rng.integers(0, 50, offsets[-1]).astype(np.uint32)))
    return out


def test_many_threads_get_their_own_rows(mlx):
    rng = np.random.default_rng(0)
    backend, host = mlx.MlxBackend(CpuBackend()), CpuBackend()
    pool = PoolInput(rng.standard_normal((50, 33)).astype(F32), rng.random(50).astype(F32))
    prepared, host_prepared = backend.prepare(pool, ()), host.prepare(pool, ())
    batches = _batches(rng)
    got = [np.empty((b.n_sentences, 33), F32) for b in batches]
    before = mlx.DEVICE.evaluations

    def rows(i):
        for j in range(i, 64, 16):
            backend.rows(batches[j], prepared, got[j])

    threads = [threading.Thread(target=rows, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for b, rows_ in zip(batches, got, strict=True):
        want = np.empty_like(rows_)
        host.rows(b, host_prepared, want)
        assert rows_.tobytes() == want.tobytes()
    assert mlx.DEVICE.evaluations - before <= 64
    assert _threads() == []


def test_rows_of_one_task_launch_one_kernel(mlx):
    backend = mlx.MlxBackend(CpuBackend())
    prepared = backend.prepare(PoolInput(np.ones((3, 4), F32), np.ones(3, F32)), ())
    before = (mlx.DEVICE.evaluations, mlx.DEVICE.launches)
    batch = Batch(0, np.array([0, 2], np.int64), np.array([0, 1], np.uint32))
    backend.rows(batch, prepared, np.empty((1, 4), F32))
    assert (mlx.DEVICE.evaluations, mlx.DEVICE.launches) == (before[0] + 1, before[1] + 1)
    assert isinstance(prepared.state.table, _kernels.Table)


class _FailingSink(MemorySink):
    """Commits fail from row ``at`` on, as a full disk would."""

    def __init__(self, dim, at):
        super().__init__(dim)
        self.at = at

    def commit(self, first_row, rows):
        if first_row >= self.at:
            raise OSError("no space left")
        super().commit(first_row, rows)


@pytest.mark.usefixtures("mlx")
def test_concurrent_infer_calls_succeed_or_fail_on_their_own(monkeypatch):
    """Several infer calls on one pipeline at once, one of them failing in its sink:
    the others give the reference bits, and no device thread remains."""
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    spec = EncodeSpec(
        word_vectors("int8", True), np.ones(200, F32), (), oov_weight=0.5, max_ngrams=12
    )
    pipeline = Pipeline(spec, "mlx")
    sentences = texts(1500)
    want = expected(spec, sentences)
    results, errors = {}, {}

    def call(k):
        try:
            sink = _FailingSink(12, 700) if k == 2 else None
            results[k] = pipeline.infer(
                StringSource(sentences), workers=3, chunk_bytes=1024, sink=sink
            )
        except Exception as err:
            errors[k] = err

    threads = [threading.Thread(target=call, args=(k,)) for k in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert list(errors) == [2]
    assert isinstance(errors[2], OSError)
    for k in (0, 1, 3, 4):
        assert results[k].tobytes() == want.tobytes()
    assert _threads() == []
