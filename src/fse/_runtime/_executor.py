#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The executor of train and infer (docs/architecture/runtime.md, "Runtime").

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
        self.futures: list[Future] = []  # every task this call submitted (to wait for them)
        self.unfinished = 0  # submitted and not yet finished
        self.limit = window  # of them at most: ``workers`` on a pool of more threads
        self.changed = threading.Event()  # a task finished
        self.held = 0  # places of the pool this call holds
        self.source_error: BaseException | None = None
        self.exhausted = False

    def execute(self, pool: Pool):
        pool.check_open()
        if self.workers < pool.workers:  # this call may use no more threads than it asked for
            self.limit = self.workers
        try:
            self._loop(pool)
        except BaseException:
            self._stop()
            for future in self.futures:
                future.cancel()
            self._wait_for_own()
            raise
        finally:
            for _ in range(self.held):
                pool.release()
            self.held = 0

    def _wait_for_own(self):
        """Running tasks finish, as when an executor is left: nothing of this call outlives
        it, and the tasks of other calls on the pool are not waited for."""
        while True:
            running = [f for f in self.futures if not f.done()]
            if not running:
                return
            wait(running, timeout=_POLL_SECONDS)

    def _loop(self, pool: Pool):
        sequence = 0
        while True:
            self.changed.clear()
            # After a worker failed, nothing more is read: only the tasks before it remain.
            while (
                not self.exhausted
                and len(self.pending) < self.window
                and not self._failed()
                and self._unfinished() < self.limit
            ):
                if not self._place(pool):
                    break
                task = self._next()
                if task is _SKIPPED:
                    self._give_back(pool)
                    break
                item = self.reserve(task)  # an error stops the run at once
                future = pool.submit(self._work, sequence, item)
                with self.lock:
                    self.unfinished += 1
                future.add_done_callback(self._finished)
                self.futures.append(future)
                self.pending.append((sequence, item, future))
                sequence += 1
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

    def _place(self, pool: Pool) -> bool:
        """A place of the pool for one more task. If none is free and this call has tasks to
        commit, False: committing releases a place. A call with no task waits for one, so the
        calls that hold places are the ones that release them."""
        if pool.try_acquire():
            self.held += 1
            return True
        if self.pending:
            return False
        pool.acquire()
        self.held += 1
        return True

    def _give_back(self, pool: Pool):
        self.held -= 1
        pool.release()

    def _unfinished(self) -> int:
        with self.lock:
            return self.unfinished

    def _finished(self, _future: Future):
        with self.lock:
            self.unfinished -= 1
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
