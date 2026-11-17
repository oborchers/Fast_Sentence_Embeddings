"""Native task owners survive a second interrupt until the running task actually ends."""

import gc
import threading
import weakref

import numpy as np

from fse._runtime import EncodeSpec, Pipeline, Pool, _pipeline, _pool, _tasks
from fse._runtime import _executor
from fse._runtime._sinks import ArrayRows
from fse._runtime._pool import Workers
from fse._text import StringSource
from support.runtime import word_vectors
from support.waiting import arrived, blocked_in_wait, until


def test_running_native_task_keeps_owners_after_two_interrupts(monkeypatch):
    condition = threading.Condition()
    release = threading.Event()
    owners, marks = [], []
    ready = []
    original_factory = Pipeline._task_work
    original_view = ArrayRows._view
    original_first = _pipeline.first_rows
    original_submit = _pool.Pool._submit

    def factory(self, source, *args):
        worker = original_factory(self, source, *args)
        if worker is not None:
            owners.extend((weakref.ref(worker), weakref.ref(source._offsets)))
        return worker

    def target(self, first, n):
        result = original_view(self, first, n)
        if first == 0:
            owners.append(weakref.ref(self._out))
        result.fill(-1 if first == 0 else -2)
        return result

    def first_rows(target, n):
        if target[0][0, 0] == -2:
            name = threading.current_thread().name
            ready.append(blocked_in_wait(condition, name))
            with condition:
                while not release.is_set():
                    condition.wait(10)
        return original_first(target, n)

    def submit(self, work, *args):
        def tracked(*items):
            try:
                return work(*items)
            finally:
                marks.append(id(self) not in _pool._ACTIVE.__dict__.get("pools", set()))

        return original_submit(self, tracked, *args)

    def first_interrupt(_self, _pool):
        assert until(lambda: bool(ready))
        assert arrived(ready[0], condition)
        raise KeyboardInterrupt("first")

    def second_interrupt(_self):
        raise KeyboardInterrupt("second")

    monkeypatch.setattr(Pipeline, "_task_work", factory)
    monkeypatch.setattr(ArrayRows, "_view", target)
    monkeypatch.setattr(_pipeline, "first_rows", first_rows)
    monkeypatch.setattr(_pool.Pool, "_submit", submit)
    monkeypatch.setattr(_executor._Run, "_commit_oldest", first_interrupt)
    monkeypatch.setattr(_executor._Run, "_wait_for_own", second_interrupt)
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)

    def interrupted(pool):
        wv = word_vectors()
        pipeline = Pipeline(EncodeSpec(wv, np.ones(len(wv), np.float32), ()))
        try:
            pipeline.infer(StringSource(["w1", "w2"]), workers=Workers(2, pool), chunk_bytes=1)
        except KeyboardInterrupt as error:
            assert str(error) == "second"
        else:
            raise AssertionError("the two interrupts did not escape")

    with Pool(2, max_tasks=2) as pool:
        try:
            interrupted(pool)
            gc.collect()
            assert len(owners) == 3
            assert all(owner() is not None for owner in owners)
            assert pool._try_acquire()
            assert not pool._try_acquire()
            pool._release()
        finally:
            release.set()
            with condition:
                condition.notify_all()
        pool._acquire()
        pool._release()
        assert until(lambda: len(marks) == 2)
        assert all(marks)
        gc.collect()
        assert all(owner() is None for owner in owners)
