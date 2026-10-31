#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Records the events of the runtime executor, in this process only (nothing in ``src``).

``tracing`` replaces, for the duration of a ``with`` block, ``run`` of the executor (in the
modules that imported it), the pool the executor creates, ``_Run._commit_oldest`` and
``Pipeline.infer`` with versions that read ``time.perf_counter_ns`` around the same calls and
call the originals. What a run did is then in ``Recorder.runs`` (runtime_timeline_attr.py
splits its idle time). The wrapped callables are the ones the executor was given, so an
inline run and a threaded run are recorded the same way; a task is the call of ``work``.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from runtime_timeline_attr import Run, Span, Task, Wait

now = time.perf_counter_ns


class Recorder:
    def __init__(self) -> None:
        self.runs: list[Run] = []
        self.infers: list[tuple[int, int]] = []
        self.current: Run | None = None


class Tagged:
    """An item of the executor with its sequence number (the order of ``reserve``)."""

    __slots__ = ("item", "seq")

    def __init__(self, seq: int, item: Any):
        self.seq, self.item = seq, item


def _rows_of(item: Any) -> int:
    task = item[0] if isinstance(item, tuple) and item else item
    rows = getattr(task, "n_rows", 1)
    return rows if isinstance(rows, int) else 1


def _timed_tasks(run: Run, tasks: Any) -> Iterator[Any]:
    iterator = iter(tasks)
    while True:
        start = now()
        try:
            task = next(iterator)
        except StopIteration:
            return
        finally:
            run.spans.append(Span("next", start, now()))
        yield task


def _wrap(run: Run, work: Callable, commit: Callable, reserve: Callable | None):
    """The three callables of the executor, recording ``run``."""
    counter = itertools.count()

    def reserved(task: Any) -> Tagged:
        start = now()
        try:
            item = task if reserve is None else reserve(task)
        finally:
            run.spans.append(Span("reserve", start, now()))
        return Tagged(next(counter), item)

    def worked(tagged: Tagged) -> Any:
        start = now()
        try:
            return work(tagged.item)
        finally:
            run.tasks.append(
                Task(tagged.seq, threading.get_ident(), start, now(), _rows_of(tagged.item))
            )

    def committed(tagged: Tagged, result: Any) -> None:
        start = now()
        try:
            commit(tagged.item, result)
        finally:
            run.spans.append(Span("commit", start, now(), tagged.seq))

    return worked, committed, reserved


def pool_class(rec: Recorder, reuse: Reuse | None = None) -> type:
    """The pool the executor creates: it records every submit and wraps ``reuse``'s pool,
    which is not shut down, or a thread pool of its own."""

    class Pool:
        def __init__(self, max_workers: int, thread_name_prefix: str = ""):
            self.real = (
                ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
                if reuse is None
                else reuse.get(max_workers)
            )

        def submit(self, fn: Callable, *args: Any) -> Any:
            assert rec.current is not None
            rec.current.submits.append(now())
            return self.real.submit(fn, *args)

        def __enter__(self) -> Pool:
            return self

        def __exit__(self, *exc: object) -> bool:
            if reuse is None:
                self.real.shutdown(wait=True)
            return False

    return Pool


class Reuse:
    """Thread pools with all threads started, kept across runs (ablation A3: what a pool
    that outlives the call would save; it breaks "no thread outlives a call")."""

    def __init__(self) -> None:
        self.pools: dict[int, ThreadPoolExecutor] = {}

    def get(self, workers: int) -> ThreadPoolExecutor:
        if workers not in self.pools:
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fse-reused")
            barrier = threading.Barrier(workers)
            for future in [pool.submit(barrier.wait) for _ in range(workers)]:
                future.result()
            self.pools[workers] = pool
        return self.pools[workers]

    def close(self) -> None:
        for pool in self.pools.values():
            pool.shutdown(wait=True)
        self.pools.clear()


def _patched_run(rec: Recorder, original: Callable) -> Callable:
    def traced(tasks, work, commit, *, workers, inline=False, window=None, reserve=None, stop=None):
        valid = isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1
        wide = 2 * workers if window is None and valid else window
        run = Run(workers if valid else 1, wide if isinstance(wide, int) else 1, False)
        run.inline = bool(inline or workers == 1)
        rec.current = run
        worked, committed, reserved = _wrap(run, work, commit, reserve)
        run.start = now()
        try:
            original(
                _timed_tasks(run, tasks),
                worked,
                committed,
                workers=workers,
                inline=inline,
                window=window,
                reserve=reserved,
                stop=stop,
            )
        finally:
            run.end = now()
            rec.runs.append(run)
            rec.current = None

    return traced


def _patched_commit_oldest(rec: Recorder, original: Callable) -> Callable:
    def traced(self) -> None:
        run = rec.current
        assert run is not None
        start, pending, seen = now(), len(self.pending), len(run.spans)
        try:
            original(self)
        finally:
            end = now()
            commits = [s for s in run.spans[seen:] if s.kind == "commit"]
            run.waits.append(Wait(start, commits[0].start if commits else end, pending))

    return traced


def _patched_infer(rec: Recorder, original: Callable) -> Callable:
    def traced(self, *args, **kwargs):
        start = now()
        try:
            return original(self, *args, **kwargs)
        finally:
            rec.infers.append((start, now()))

    return traced


@contextmanager
def tracing(rec: Recorder, reuse: Reuse | None = None):
    """Record the runs of the executor into ``rec`` inside the block."""
    from fse._runtime import _executor, _head, _pipeline

    run, commit_oldest = _executor.run, _executor._Run._commit_oldest
    infer, pool = _pipeline.Pipeline.infer, _executor.ThreadPoolExecutor
    traced = _patched_run(rec, run)
    _executor.run = _pipeline.run = _head.run = traced
    _executor._Run._commit_oldest = _patched_commit_oldest(rec, commit_oldest)
    _executor.ThreadPoolExecutor = pool_class(rec, reuse)
    _pipeline.Pipeline.infer = _patched_infer(rec, infer)
    try:
        yield rec
    finally:
        _executor.run = _pipeline.run = _head.run = run
        _executor._Run._commit_oldest = commit_oldest
        _executor.ThreadPoolExecutor = pool
        _pipeline.Pipeline.infer = infer
