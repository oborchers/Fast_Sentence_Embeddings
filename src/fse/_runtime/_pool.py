#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The threads a run of tasks executes on (docs/architecture/runtime.md, "Runtime", Pool).

A ``Pool`` is a set of worker threads that runs the tasks of any number of calls, one after
the other or at once. A call without a pool makes one for itself and closes it at its end, so
no thread outlives a call; a stream (``encode_batches``, ``predict_batches``) makes one for
its iteration. ``fse.Pool`` hands the same object to the caller, who then owns its life.
"""

from __future__ import annotations

import functools
import inspect
import os
import threading
import weakref
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import Any, NamedTuple, ParamSpec, TypeVar

import numpy as np

from fse._cpus import usable_cpus

# Waits wake up this often, so an interrupt reaches the calling thread (also on Windows).
_POLL_SECONDS = 0.25
_POOLS: weakref.WeakSet[Pool] = weakref.WeakSet()
_ACTIVE = threading.local()  # the pools whose call or task the thread is in


class Pool:
    """Worker threads that run the tasks of encoding and prediction, reused across calls.

    ``Pool(workers=None)`` has ``workers`` threads (None: the usable CPUs). Pass it to a
    method as ``pool=``; the method then runs its tasks on these threads instead of making
    threads of its own, and a call that gives ``workers=`` runs at most that many tasks at
    once (it may not exceed ``pool.workers``). Threads are started by the first task, or at
    once with ``start=True``, so the first call of a server does not wait for them. A pool
    holds at most ``max_tasks`` tasks (running, waiting or finished and not yet committed)
    over all calls at once (default ``2 * workers``); a call that finds none free waits.

    ``close()`` (or leaving a ``with`` block) waits for running tasks and ends the threads;
    a closed pool refuses further calls. A pool that is dropped without ``close()`` ends its
    threads when it is collected. A child of ``os.fork`` starts the threads again at its
    first call. ``max_tasks`` bounds the tasks, not their memory: every call keeps its own
    window of rows.
    """

    def __init__(
        self, workers: int | None = None, *, start: bool = False, max_tasks: int | None = None
    ):
        self._workers = usable_cpus() if workers is None else _positive(workers, "workers")
        self._max_tasks = (
            2 * self._workers if max_tasks is None else _positive(max_tasks, "max_tasks")
        )
        if not isinstance(start, bool):
            raise TypeError(f"start must be a bool, not {type(start).__name__}")
        self._lock = threading.Lock()
        self._permits = threading.BoundedSemaphore(self._max_tasks)
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False
        _POOLS.add(self)
        if start:
            self._start()

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def max_tasks(self) -> int:
        return self._max_tasks

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Wait for the running tasks and end the threads; later calls of the pool raise.
        Every caller of ``close`` waits until the threads have ended."""
        with self._lock:
            self._closed = True
            executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True)  # idempotent: each caller joins the threads
            with self._lock:
                if self._executor is executor:
                    self._executor = None

    def __enter__(self) -> Pool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:  # nothing waits in a finalizer; idle threads end by themselves
            executor.shutdown(wait=False)

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"Pool(workers={self._workers}, max_tasks={self._max_tasks}, {state})"

    # --- what the executor of a run uses ------------------------------------------------

    def submit(self, work: Callable[..., Any], *args: Any) -> Future:
        """``work(*args)`` on a thread of the pool."""
        with self._lock:
            if self._closed:
                raise RuntimeError("the pool is closed")
            if self._executor is None:
                self._executor = self._new_executor()
            executor = self._executor
        return executor.submit(work, *args)

    def try_acquire(self) -> bool:
        """Take one of the ``max_tasks`` places if one is free."""
        return self._permits.acquire(blocking=False)

    def acquire(self) -> None:
        """Wait for a place; the wait ends early only by an interrupt."""
        while not self._permits.acquire(timeout=_POLL_SECONDS):
            pass

    def release(self) -> None:
        self._permits.release()

    def check_open(self) -> None:
        if self._closed:
            raise RuntimeError("the pool is closed")

    @contextmanager
    def inside(self) -> Iterator[None]:
        """Mark the calling thread as inside a call or a task of this pool. A call of the pool
        from there (a sink that encodes on the same pool, a task that runs a call) would wait
        for places that its own call holds, so it is refused instead of left to hang."""
        active = _ACTIVE.__dict__.setdefault("pools", set())
        if id(self) in active:
            raise RuntimeError(
                "a call on this pool was made inside a call or a task of the same pool; "
                "give the inner call its own pool or run it with workers=1"
            )
        active.add(id(self))
        try:
            yield
        finally:
            active.discard(id(self))

    # --- threads -------------------------------------------------------------------------

    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="fse-worker")

    def _start(self) -> None:
        """Start every thread now: as many tasks that wait for each other need as many
        threads, so the barrier opens only when all of them run."""
        if self._workers == 1:
            self.submit(_nothing).result()
            return
        barrier = threading.Barrier(self._workers)
        futures = [self.submit(_arrive, barrier) for _ in range(self._workers)]
        wait(futures)
        for future in futures:
            future.result()  # a timeout of the barrier is an error, not a silent half start

    def _after_fork(self) -> None:
        """In the child of a fork the threads do not exist: forget them (the next call makes
        new ones), and take a lock and places that no thread of the parent holds."""
        self._lock = threading.Lock()
        self._permits = threading.BoundedSemaphore(self._max_tasks)
        self._executor = None


class Workers(NamedTuple):
    """How many tasks a call runs at once and on which pool (None: it makes its own).

    The public methods make one from ``workers=`` and ``pool=`` before they read anything;
    the layers below pass it on where they passed ``workers``."""

    workers: int
    pool: Pool | None = None


def workers_of(workers: object, pool: object = None) -> Workers:
    """The ``Workers`` of a call, checked before the source is read: ``workers`` None is the
    usable CPUs, or the workers of ``pool``; with a pool it may not exceed them, and a
    closed pool is refused."""
    if isinstance(workers, Workers):
        return workers
    if workers is not None:
        workers = _positive(workers, "workers")
    if pool is None:
        return Workers(usable_cpus() if workers is None else int(workers))
    if not isinstance(pool, Pool):
        raise TypeError(f"pool must be an fse.Pool, not {type(pool).__name__}")
    pool.check_open()
    if workers is None:
        return Workers(pool.workers, pool)
    if workers > pool.workers:
        raise ValueError(f"workers ({workers}) must not exceed pool.workers ({pool.workers})")
    return Workers(int(workers), pool)


def resolve_workers(workers: object) -> tuple[int, Pool | None]:
    """The count and the pool of ``workers``: an int, None or ``Workers``."""
    made = workers_of(workers)
    return made.workers, made.pool


@contextmanager
def iteration(workers: Workers) -> Iterator[Workers]:
    """The ``Workers`` for the whole iteration of a stream: a pool of the caller stays the
    caller's; otherwise one pool serves every batch of the stream and is closed when the
    iteration ends (exhausted, failed, closed or collected), so no thread outlives it.
    Threads start with the first batch that needs them."""
    if workers.pool is not None or workers.workers == 1:
        yield workers
        return
    own = Pool(workers.workers)
    try:
        yield Workers(workers.workers, own)
    finally:
        own.close()


P = ParamSpec("P")
T = TypeVar("T")


def streaming(
    function: Callable[P, Generator[T, None, None]],
) -> Callable[P, Generator[T, None, None]]:
    """A generator function with a ``workers`` argument whose iteration runs on one pool:
    the ``Workers`` it is given become those of ``iteration``, so its batches share threads
    and the iteration closes them (exhausted, failed, closed or collected)."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Generator[T, None, None]:
        bound = signature.bind(*args, **kwargs)
        with iteration(workers_of(bound.arguments["workers"])) as workers:
            bound.arguments["workers"] = workers
            yield from function(*bound.args, **bound.kwargs)

    return wrapper


def _arrive(barrier: threading.Barrier) -> None:
    barrier.wait(timeout=30)


def _nothing() -> None:
    return None


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)


def _reset_pools_in_child() -> None:
    for pool in list(_POOLS):
        pool._after_fork()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pools_in_child)
