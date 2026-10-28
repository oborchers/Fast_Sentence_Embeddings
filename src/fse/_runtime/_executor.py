#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The executor of train and infer (ARCHITECTURE.md, "Runtime").

``run`` takes tasks from an iterator in the calling thread, runs ``work`` on each in a
worker thread (or inline) and commits the results in the calling thread in task order,
so a sink needs no locks. At most ``window`` tasks are in flight or waiting to be
committed, so a slow commit slows the source and memory stays bounded.

Errors are deterministic for a given sequence of tasks: the error raised is the one of
the first failing task in order, after every task before it was committed, as with one
worker. Tasks after a failing one are skipped where they have not started. An error of
the task iterator ends the run after the tasks taken before it were committed. An error
of ``commit`` (or an interrupt) stops the run at once: tasks that have not started are
cancelled, running ones finish, then it is raised. No thread outlives a call.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")

_SKIPPED = object()
# Waits wake up this often, so an interrupt reaches the calling thread (also on Windows).
_POLL_SECONDS = 0.25


def run(
    tasks: Iterable[T],
    work: Callable[[T], R],
    commit: Callable[[T, R], None],
    *,
    workers: int,
    inline: bool = False,
    window: int | None = None,
) -> None:
    """Run ``work`` on every task and ``commit`` each result in task order."""
    workers = _positive(workers, "workers")
    window = 2 * workers if window is None else _positive(window, "window")
    if inline or workers == 1:
        for task in tasks:
            commit(task, work(task))
        return
    _Run(iter(tasks), work, commit, window).threads(workers)


class _Run:
    def __init__(self, tasks: Iterator[Any], work: Callable, commit: Callable, window: int):
        self.tasks, self.work, self.commit, self.window = tasks, work, commit, window
        self.lock = threading.Lock()
        self.failed_at: float = float("inf")  # sequence of the first failed task so far
        self.pending: deque[tuple[int, Any, Future]] = deque()
        self.source_error: BaseException | None = None
        self.exhausted = False

    def threads(self, workers: int):
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fse-worker") as pool:
            try:
                self._loop(pool)
            except BaseException:
                self._stop()
                for _, _, future in self.pending:
                    future.cancel()
                raise

    def _loop(self, pool: ThreadPoolExecutor):
        sequence = 0
        while True:
            while not self.exhausted and len(self.pending) < self.window:
                task = self._next()
                if task is _SKIPPED:
                    break
                self.pending.append((sequence, task, pool.submit(self._work, sequence, task)))
                sequence += 1
            if not self.pending:
                if self.source_error is not None:
                    raise self.source_error
                return
            self._commit_oldest()

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
            raise

    def _commit_oldest(self):
        _, task, future = self.pending.popleft()
        while not future.done():
            wait([future], timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED)
        result = future.result()  # raises the error of the first failing task in order
        assert result is not _SKIPPED  # only tasks after a failed one are skipped
        self.commit(task, result)

    def _stop(self):
        with self.lock:
            self.failed_at = -1.0


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)
