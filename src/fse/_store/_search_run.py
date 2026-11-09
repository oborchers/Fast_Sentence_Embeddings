#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The tasks of a search, score, rescore or sign-bit backfill, in task order, inline or on worker
threads (docs/architecture/store.md, "Concurrency").

Without a pool a call that has two tasks or more makes its own threads and ends them when
it ends, as before. With an ``fse.Pool`` (``pool=``) its tasks run on the threads of the
pool instead, at most ``workers`` of them at once, and the call takes one of the pool's
``max_tasks`` places for every task from before it is submitted until it has run. A place
is given back when its task ends, not when its result is taken: the results a call has not
taken yet are bounded by its own window (two per worker), so a stream of blocks
(``predict_batches`` of a store) whose consumer pauses holds no place, and a call made
between two of its blocks finds the places of the stream free again as soon as its tasks
end. A call that finds no place free takes a finished result of its own first, and waits
for a place only when it has none.

A call of one task, or with ``workers=1``, runs in the calling thread and takes no place;
with a pool and more than one worker it still runs every task inside the pool, so a call
on the same pool from within one is refused as from a task on the pool's threads.

When the call ends early (an error, a prefilter that gives up, a stream closed or
collected before its last block), its tasks that have not started are cancelled and its
running ones finish before it returns; their places come back as they end. The tasks of
other calls on the pool are not waited for, and the pool stays open.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any

from fse._runtime import Pool, Workers, workers_of

# Waits wake up this often, so an interrupt reaches the calling thread (also on Windows).
_POLL_SECONDS = 0.25
_END = object()


def run(fn: Callable[[Any], Any], tasks: Iterable[Any], workers: Any, name: str) -> Iterator[Any]:
    """The results of ``fn`` on every task, in task order: in this thread, or on worker
    threads with at most two tasks per worker in flight, so finished results stay bounded
    and a task starts with the floors merged so far. ``workers`` is an int or the
    ``Workers`` of the call (with its pool)."""
    threads = workers_of(workers)
    tasks = iter(tasks)
    head = list(itertools.islice(tasks, 2))
    pool = threads.pool
    if threads.workers == 1 or len(head) < 2:
        if pool is None or threads.workers == 1:
            yield from map(fn, itertools.chain(head, tasks))
        else:
            yield from _inline(fn, itertools.chain(head, tasks), pool)
        return
    if pool is None:
        yield from _own_threads(fn, itertools.chain(head, tasks), threads.workers, name)
    else:
        yield from _on_pool(fn, itertools.chain(head, tasks), threads)


def count(workers: Any) -> int:
    """The workers of a call (an int or its ``Workers``), as its log records them."""
    return workers_of(workers).workers


def _inline(fn: Callable[[Any], Any], tasks: Iterator[Any], pool: Pool) -> Iterator[Any]:
    """The tasks in this thread, each inside the pool (not across a ``yield``: what the
    consumer does between two results is not a task of the pool)."""
    pool._check_open()
    for task in tasks:
        with pool._inside():
            result = fn(task)
        yield result


def _own_threads(fn: Callable[[Any], Any], tasks: Iterator[Any], workers: int, name: str):
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name) as executor:
        pending: deque = deque()
        for task in tasks:
            pending.append(executor.submit(fn, task))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


class _Own:
    """The tasks one call submitted to a pool and that have not ended. A task gives its place
    back in its worker when it ends (run or failed), or here when it is cancelled before it
    started: exactly once, also when an interrupt hits the calling thread meanwhile."""

    def __init__(self, pool: Pool):
        self.pool = pool
        self.ended = threading.Condition()
        self.unended = 0  # submitted, neither ended nor cancelled
        self.futures: set[Future] = set()  # to cancel; a finished one may linger, harmlessly
        self.changed = threading.Event()  # a task finished: wakes the call (or its poll does)

    def submit(self, fn: Callable[[Any], Any], task: Any) -> Future:
        """``fn(task)`` on the pool, on a place the caller took."""
        with self.ended:
            self.unended += 1
        try:
            future = self.pool._submit(self._work, fn, task)
        except BaseException:
            self._end()
            raise
        with self.ended:
            self.futures.add(future)
        future.add_done_callback(self._finished)
        return future

    def _work(self, fn: Callable[[Any], Any], task: Any) -> Any:
        try:
            with self.pool._inside():
                return fn(task)
        finally:
            self._end()

    def _end(self) -> None:
        self.pool._release()  # first: once no task is left, every place is back
        with self.ended:
            self.unended -= 1
            self.ended.notify_all()

    def _finished(self, future: Future) -> None:
        with self.ended:
            self.futures.discard(future)
        self.changed.set()

    def unfinished(self) -> int:
        with self.ended:
            return self.unended

    def settle(self) -> None:
        """Cancel every task of this call that has not started and wait for the running ones:
        nothing of this call outlives it, and the tasks of other calls are not waited for.
        A second interrupt stops the wait, not the places: each comes back when its task
        ends."""
        with self.ended:
            mine = list(self.futures)
        for future in mine:  # outside the lock: a cancel runs the callbacks at once
            if future.cancel():
                self._end()
        with self.ended:
            while self.unended:
                self.ended.wait(_POLL_SECONDS)


def _taken(future: Future) -> None:
    """Wait for ``future`` in steps, so an interrupt reaches the calling thread."""
    while not future.done():
        wait([future], timeout=_POLL_SECONDS)


def _on_pool(fn: Callable[[Any], Any], tasks: Iterator[Any], threads: Workers) -> Iterator[Any]:
    pool, limit = threads.pool, threads.workers
    assert pool is not None
    pool._check_open()
    with pool._inside():  # refused inside a call or a task of the same pool
        pass
    own = _Own(pool)
    pending: deque[Future] = deque()
    exhausted = False
    try:
        while True:
            own.changed.clear()
            exhausted = exhausted or _fill(own, fn, tasks, pending, 2 * limit, limit)
            if not pending:
                return
            oldest = pending[0]
            if oldest.done() or exhausted or len(pending) >= 2 * limit:
                _taken(oldest)
                pending.popleft()  # only once it is done: until then settle cancels it
                yield oldest.result()
            else:  # a finished task of this call may free a slot for the next one
                own.changed.wait(timeout=_POLL_SECONDS)
    finally:
        own.settle()


def _fill(own: _Own, fn, tasks: Iterator[Any], pending: deque, window: int, limit: int) -> bool:
    """Submit tasks until the window, the slots of this call or the places of the pool are
    used; True once the tasks are exhausted. With no place free, a call that has results
    of its own takes those first; one that has none waits for a place."""
    while len(pending) < window and own.unfinished() < limit:
        if not own.pool._try_acquire():
            if pending:
                return False
            own.pool._acquire()
        try:
            task = next(tasks, _END)
        except BaseException:
            own.pool._release()
            raise
        if task is _END:
            own.pool._release()
            return True
        pending.append(own.submit(fn, task))
    return False
