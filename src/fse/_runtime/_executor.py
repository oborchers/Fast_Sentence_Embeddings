#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The executor of encoding (docs/architecture/runtime.md, "Runtime").

``run`` takes tasks from an iterator in the calling thread, runs ``work`` on each in a
worker thread (or inline) and commits the results in the calling thread in task order,
so a sink needs no locks. At most ``window`` tasks are in flight or waiting to be
committed, so a slow commit slows the source and memory stays bounded.

Errors are deterministic for a given sequence of tasks: the error raised is the one of
the first failing task in order, after every task before it was committed, as with one
worker. Tasks after a failing one are skipped where they have not started. An error of
the task iterator ends the run after the tasks taken before it were committed. An error
of ``commit`` (or an interrupt) stops the run at once: tasks that have not started are
cancelled, running ones finish, then it is raised. No thread outlives a call that made
its own pool; a pool the caller passes is the caller's to close.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, wait
from typing import Any, TypeVar

import numpy as np

from ._pool import Pool

T = TypeVar("T")
R = TypeVar("R")

_SKIPPED = object()
# Waits wake up this often, so an interrupt reaches the calling thread (also on Windows).
_POLL_SECONDS = 0.25


def run(
    tasks: Iterable[T],
    work: Callable[[Any], R],
    commit: Callable[[Any, R], None],
    *,
    workers: int,
    inline: bool = False,
    window: int | None = None,
    reserve: Callable[[T], Any] | None = None,
    stop: threading.Event | None = None,
    pool: Pool | None = None,
) -> None:
    """Run ``work`` on every task and ``commit`` each result in task order.

    ``reserve``, if given, turns each task into what ``work`` and ``commit`` receive, in
    the calling thread before the task is submitted (a sink reserving its rows); an error
    of it stops the run like an error of ``commit``. ``stop``, if given, is set as soon as
    a task fails, so a source can stop reading (a stream between two sentences).

    ``pool`` is the threads to run on, which the caller owns and which may run other calls
    at once: at most ``workers`` tasks of this call run at once, and the pool's own limit
    on tasks is shared. Without a pool the call makes one and closes it before it returns.
    """
    workers = _positive(workers, "workers")
    window = 2 * workers if window is None else _positive(window, "window")
    reserve = reserve or _same
    if inline or workers == 1:
        for task in tasks:
            item = reserve(task)
            commit(item, work(item))
        return
    if pool is not None:
        _Run(iter(tasks), work, commit, window, reserve, stop, workers).execute(pool)
        return
    with Pool(workers, max_tasks=window) as own:  # all the window this call needs
        _Run(iter(tasks), work, commit, window, reserve, stop, workers).execute(own)


def _same(task: Any) -> Any:
    return task


class _Run:
    def __init__(
        self,
        tasks: Iterator[Any],
        work: Callable,
        commit: Callable,
        window: int,
        reserve: Callable,
        stop: threading.Event | None,
        workers: int,
    ):
        self.tasks, self.work, self.commit, self.window = tasks, work, commit, window
        self.reserve, self.stop, self.workers = reserve, stop, workers
        self.lock = threading.Lock()
        self.failed_at: float = float("inf")  # sequence of the first failed task so far
        self.pending: deque[tuple[int, Any, Future]] = deque()
        self.futures: set[Future] = (
            set()
        )  # the tasks this call submitted and that have not finished
        self.unfinished = 0  # submitted and not yet finished
        self.limit = window  # of them at most: ``workers`` on a pool of more threads
        self.changed = threading.Event()  # a task finished
        self.held = 0  # places of the pool this call holds
        self.source_error: BaseException | None = None
        self.exhausted = False
        self.sequence = 0  # of the next task
        self.pool_of_call: Pool

    def execute(self, pool: Pool):
        pool._check_open()
        self.pool_of_call = pool
        if self.workers < pool.workers:  # this call may use no more threads than it asked for
            self.limit = self.workers
        with pool._inside():
            try:
                self._loop(pool)
            except BaseException:
                self._stop()
                for future in self._running():
                    future.cancel()
                self._wait_for_own()
                raise
            finally:
                self._give_back_all(pool)

    def _running(self) -> list[Future]:
        with self.lock:
            return list(self.futures)

    def _wait_for_own(self):
        """Running tasks finish, as when an executor is left: nothing of this call outlives
        it, and the tasks of other calls on the pool are not waited for. A second interrupt
        stops the wait, but not the accounting: see ``_give_back_all``."""
        while True:
            running = [f for f in self._running() if not f.done()]
            if not running:
                return
            wait(running, timeout=_POLL_SECONDS)

    def _give_back_all(self, pool: Pool):
        """The places of this call: those of tasks that ended now, those of tasks that still
        run when their task ends, so the pool never holds more tasks than its limit."""
        # By ``done()``, not by the set: the callback that drops a future runs a moment after it
        # is done, and a task that is done and committed gave its place back already.
        running = [f for f in self._running() if not f.done()]
        for future in running:
            future.add_done_callback(lambda _f: pool._release())
        for _ in range(self.held - len(running)):
            pool._release()
        self.held = 0

    def _loop(self, pool: Pool):
        self.sequence = 0
        while True:
            self.changed.clear()
            self._fill(pool)
            if not self.pending:
                if self.source_error is not None:
                    raise self.source_error
                if self.exhausted:
                    return
                continue  # a place was refused with nothing to commit: try again
            if self.pending[0][2].done():
                self._commit_oldest(pool)
            else:
                # Not the oldest yet: a finished task of this call may free a slot for the
                # next one, so do not sit on the oldest alone.
                self.changed.wait(timeout=_POLL_SECONDS)

    def _fill(self, pool: Pool):
        """Submit tasks until the window, the slots of this call or the places of the pool are
        used. After a worker failed, nothing more is read: only the tasks before it remain."""
        while (
            not self.exhausted
            and len(self.pending) < self.window
            and not self._failed()
            and self._unfinished() < self.limit
        ):
            if not self._place(pool):
                return
            task = self._next()
            if task is _SKIPPED:
                self._give_back(pool)
                return
            item = self.reserve(task)  # an error stops the run at once
            future = pool._submit(self._work, self.sequence, item)
            with self.lock:
                self.unfinished += 1
                self.futures.add(future)  # before the callback: it may run at once
            future.add_done_callback(self._finished)
            self.pending.append((self.sequence, item, future))
            self.sequence += 1

    def _place(self, pool: Pool) -> bool:
        """A place of the pool for one more task. If none is free and this call has tasks to
        commit, False: committing releases a place. A call with no task waits for one, so the
        calls that hold places are the ones that release them."""
        if pool._try_acquire():
            self.held += 1
            return True
        if self.pending:
            return False
        pool._acquire()
        self.held += 1
        return True

    def _give_back(self, pool: Pool):
        self.held -= 1
        pool._release()

    def _unfinished(self) -> int:
        with self.lock:
            return self.unfinished

    def _finished(self, future: Future):
        with self.lock:
            self.unfinished -= 1
            self.futures.discard(future)  # a finished future holds its rows: drop it
        self.changed.set()

    def _next(self) -> Any:
        try:
            return next(self.tasks)
        except StopIteration:
            pass
        except Exception as error:  # committed in order before it is raised
            self.source_error = error
        self.exhausted = True
        return _SKIPPED

    def _work(self, sequence: int, task: Any) -> Any:
        if sequence > self.failed_at:
            return _SKIPPED
        try:
            with self.pool_of_call._inside():
                return self.work(task)
        except BaseException:
            with self.lock:
                self.failed_at = min(self.failed_at, sequence)
            if self.stop is not None:
                self.stop.set()
            raise

    def _commit_oldest(self, pool: Pool):
        _, task, future = self.pending.popleft()
        result = future.result()  # raises the error of the first failing task in order
        assert result is not _SKIPPED  # only tasks after a failed one are skipped
        self.commit(task, result)
        self._give_back(pool)  # a task that failed or was never committed is given back by execute

    def _failed(self) -> bool:
        with self.lock:
            return self.failed_at != float("inf")

    def _stop(self):
        with self.lock:
            self.failed_at = -1.0
        if self.stop is not None:
            self.stop.set()


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)


def prepare(
    tasks: Iterable[T],
    work: Callable[[T], None],
    *,
    workers: int,
    pool: Pool | None = None,
    inline: bool = False,
) -> None:
    """Run independent preparation blocks, helping instead of waiting for pool places.

    Preparation may hold a model's publication lock. A pool task or sink can be waiting
    for that lock, so neither a free place nor a queued helper may be waited for. Up to
    ``workers - 1`` helpers claim free places without waiting; the caller drains the same
    iterator. It then cancels queued helpers and joins only helpers already running.
    Thus the caller plus helpers use at most ``workers`` threads, even on a larger pool.
    Blocks write disjoint output slices; errors are raised in input order after joining.
    """
    if pool is None or inline or workers == 1:
        run(tasks, work, lambda _task, _done: None, workers=workers, pool=pool, inline=inline)
        return
    pool._check_open()
    helping = _Helping(tasks, work)
    futures: list[Future] = []
    try:
        for _ in range(workers - 1):
            if not pool._try_acquire():
                break
            futures.append(helping.submit(pool))
        helping.drain()
    finally:
        for future in futures:
            if future.cancel():
                pool._release()
        for future in futures:
            if not future.cancelled():
                future.result()
    if helping.error is not None:
        raise helping.error[1]


class _Helping:
    """Independent preparation blocks shared by the caller and optional pool helpers."""

    def __init__(self, tasks: Iterable, work: Callable):
        self.tasks, self.work = enumerate(tasks), work
        self.lock = threading.Lock()
        self.error: tuple[int, BaseException] | None = None

    def submit(self, pool: Pool) -> Future:
        try:
            return pool._submit(self.worker, pool)
        except BaseException:
            pool._release()
            raise

    def worker(self, pool: Pool) -> None:
        try:
            with pool._inside():
                self.drain()
        finally:
            pool._release()

    def drain(self) -> None:
        while True:
            with self.lock:
                if self.error is not None:
                    return
                item = next(self.tasks, None)
                if item is None:
                    return
            sequence, task = item
            try:
                self.work(task)
            except BaseException as error:
                with self.lock:
                    if self.error is None or sequence < self.error[0]:
                        self.error = (sequence, error)
                return
