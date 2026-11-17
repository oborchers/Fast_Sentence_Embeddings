"""Native task owners survive a second interrupt until the running task actually ends."""

import gc
import threading
import weakref

import numpy as np
import pytest

from fse._runtime import EncodeSpec, Pipeline, Pool, _executor, _pipeline, _pool, _tasks
from fse._runtime._pool import Workers
from fse._runtime._sinks import ArrayRows
from fse._text import StringSource
from support.runtime import word_vectors
from support.waiting import arrived, blocked_in_wait, until


class _Ownership:
    """Block the second native task and retain only weak references to its owners."""

    def __init__(self):
        self.condition = threading.Condition()
        self.release = threading.Event()
        self.owners, self.marks, self.ready = [], [], []
        self.capturing = False
        self.array_fn = StringSource._array
        self.factory_fn = Pipeline._task_work
        self.view_fn = ArrayRows._view
        self.first_fn = _pipeline.first_rows
        self.submit_fn = _pool.Pool._submit

    def install(self, monkeypatch):
        monkeypatch.setattr(StringSource, "_array", lambda source: self.array(source))
        monkeypatch.setattr(
            Pipeline, "_task_work", lambda pipe, src, *a: self.factory(pipe, src, *a)
        )
        monkeypatch.setattr(ArrayRows, "_view", lambda sink, first, n: self.target(sink, first, n))
        monkeypatch.setattr(_pipeline, "first_rows", lambda target, n: self.first_rows(target, n))
        monkeypatch.setattr(
            _pool.Pool, "_submit", lambda pool, work, *a: self.submit(pool, work, *a)
        )
        monkeypatch.setattr(_executor._Run, "_commit_oldest", lambda *_: self.interrupt())
        monkeypatch.setattr(_executor._Run, "_wait_for_own", _second_interrupt)
        monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)

    def array(self, source):
        view = self.array_fn(source)
        if self.capturing:
            self.owners.append(weakref.ref(view))
        return view

    def factory(self, pipeline, source, *args):
        self.capturing = True
        try:
            worker = self.factory_fn(pipeline, source, *args)
        finally:
            self.capturing = False
        if worker is not None:
            self.owners.extend((weakref.ref(worker), weakref.ref(source._offsets)))
        return worker

    def target(self, sink, first, n):
        result = self.view_fn(sink, first, n)
        if first == 0:
            self.owners.append(weakref.ref(sink._out))
        result.fill(-1 if first == 0 else -2)
        return result

    def first_rows(self, target, n):
        if target[0][0, 0] == -2:
            name = threading.current_thread().name
            self.ready.append(blocked_in_wait(self.condition, name))
            with self.condition:
                while not self.release.is_set():
                    self.condition.wait(10)
        return self.first_fn(target, n)

    def submit(self, pool, work, *args):
        def tracked(*items):
            try:
                return work(*items)
            finally:
                self.marks.append(id(pool) not in _pool._ACTIVE.__dict__.get("pools", set()))

        return self.submit_fn(pool, tracked, *args)

    def interrupt(self):
        assert until(lambda: bool(self.ready))
        assert arrived(self.ready[0], self.condition)
        raise KeyboardInterrupt("first")

    def unblock(self):
        self.release.set()
        with self.condition:
            self.condition.notify_all()

    def check_held(self, pool):
        gc.collect()
        assert len(self.owners) == 4
        assert all(owner() is not None for owner in self.owners)
        assert pool._try_acquire()
        assert not pool._try_acquire()
        pool._release()

    def released(self):
        gc.collect()
        return all(owner() is None for owner in self.owners)

    def check_finished(self, pool):
        assert until(lambda: len(self.marks) == 2)
        assert all(self.marks)
        assert until(self.released)
        assert pool._try_acquire()
        assert pool._try_acquire()
        assert not pool._try_acquire()
        pool._release()
        pool._release()


def _second_interrupt(_run):
    raise KeyboardInterrupt("second")


def _interrupted(pool):
    wv = word_vectors()
    pipeline = Pipeline(EncodeSpec(wv, np.ones(len(wv), np.float32), ()))
    with pytest.raises(KeyboardInterrupt, match=r"^second$"):
        pipeline.infer(StringSource(["w1", "w2"]), workers=Workers(2, pool), chunk_bytes=1)


def test_running_native_task_keeps_owners_after_two_interrupts(monkeypatch):
    probe = _Ownership()
    probe.install(monkeypatch)
    with Pool(2, max_tasks=2) as pool:
        try:
            _interrupted(pool)
            probe.check_held(pool)
        finally:
            probe.unblock()
        probe.check_finished(pool)
