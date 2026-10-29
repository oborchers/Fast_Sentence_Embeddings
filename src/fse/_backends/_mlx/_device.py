#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The device thread of the mlx backend: the one owner of all MLX work in the process.

``rows`` blocks in the worker that calls it. Every call becomes a job: the worker queues
it and waits, the device thread takes all jobs that wait at that moment (up to
``GROUP_TASKS`` tasks or ``GROUP_TOKENS`` tokens), queues the kernel of each and
evaluates them together, so the workers of a run share one GPU round trip instead of
one each. The thread starts with the first job and ends when no job is left, so no
thread outlives the calls that use it; a new thread first joins the one before, so
MLX work never overlaps. All threads use one stream made for this hand-over
(``mx.new_thread_unsafe_stream``): a thread's default stream costs about 1 ms to make.

A job's ``launch`` (queue the kernel) and ``finish`` (its results as host arrays, such as
a view of an MLX array) run in the device thread; its ``use`` (copy the view into the
caller's arrays) runs in the caller after the job is done. The device thread lends the
launched MLX arrays until the caller returns the lease, after ``use`` or its error or
interrupt, and drops them itself (their last reference) afterwards, before it ends; it
runs the groups already queued meanwhile. A ``use`` must not wait for another call: the
callers of the last group wait until every lease is back. An error in one job's launch or finish fails that job only;
an error of the evaluation fails every job of the group. A caller that is interrupted
(KeyboardInterrupt in the calling thread) anywhere from queueing to waiting takes its job
out of the queue, or waits until the device thread is done with it (further interrupts
wait too), so nothing writes into its arrays after the call. A device thread that
cannot start leaves the queue and the state as they were before the call.

The device thread gets MLX from ``core`` (``_load.core`` for the process's ``DEVICE`` in
``_backend``), so this module and its tests need no MLX.
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from collections.abc import Callable
from types import ModuleType
from typing import Any

from . import _canary, _kernels

GROUP_TASKS = 8
GROUP_TOKENS = 1 << 24

Launch = Callable[[Any, Any], tuple]
Finish = Callable[[tuple], Any]
Use = Callable[[Any], Any]


class Job:
    """One call: its launch and finish, its token count, and its outcome."""

    __slots__ = (
        "arrays",
        "done",
        "ended",
        "error",
        "finish",
        "launch",
        "lease",
        "result",
        "returned",
        "state",
        "tokens",
    )

    def __init__(self, launch: Launch, finish: Finish, tokens: int, lease: bool = False):
        self.launch, self.finish, self.tokens, self.lease = launch, finish, tokens, lease
        self.arrays: tuple | None = None  # the launched arrays, while lent to the caller
        self.returned = threading.Event()  # the caller is done with the lent arrays
        self.state = "queued"  # then "running"; a cancelled job leaves the queue
        self.done = threading.Event()
        self.ended: threading.Thread | None = None  # the device thread, if it ends after it
        self.result: Any = None
        self.error: BaseException | None = None


class Device:
    """The queue and the device thread; one per process (``DEVICE``), for every encoder."""

    def __init__(
        self,
        core: Callable[[], ModuleType],
        group_tasks: int = GROUP_TASKS,
        group_tokens: int = GROUP_TOKENS,
    ):
        if not 0 < group_tokens <= _kernels.LIMIT:
            raise ValueError(f"group_tokens must be from 1 to {_kernels.LIMIT}, not {group_tokens}")
        self.core, self.group_tasks, self.group_tokens = core, group_tasks, group_tokens
        self._lock = threading.Lock()
        self._queue: deque[Job] = deque()
        self._thread: threading.Thread | None = None
        self._previous: threading.Thread | None = None
        self._stream: Any = None
        self._canary: BaseException | bool = False
        self.evaluations = 0  # groups evaluated, and jobs launched (for the tests)
        self.launches = 0

    def run(self, launch: Launch, finish: Finish, tokens: int = 0, use: Use | None = None) -> Any:
        """Run one job in the device thread and give its result, or ``use(result)`` in
        the caller while the device thread lends the job's arrays, or raise its error."""
        job = Job(launch, finish, tokens, use is not None)
        try:
            self._enqueue(job)
            job.done.wait()
        except BaseException:
            self._abort(job)
            raise
        try:
            result = _used(job, use)
        finally:
            _give_back(job)
        if job.error is not None:
            raise job.error
        return result

    def call(self, function: Callable[[Any, Any], Any]) -> Any:
        """``function(mx, stream)`` in the device thread, which evaluates what it needs."""
        box: list = []

        def launch(mx: Any, stream: Any) -> tuple:
            box.append(function(mx, stream))
            return ()

        return self.run(launch, lambda _arrays: box[0])

    def ready(self) -> None:
        """Run the canary once per process; raise its error on every call after it."""
        with self._lock:
            known = self._canary
        if known is False:
            try:
                self.call(_canary.canary)
            except Exception as err:
                known = err
            else:
                known = True
            with self._lock:
                self._canary = known
        if isinstance(known, BaseException):
            raise known

    def _enqueue(self, job: Job) -> None:
        """Queue ``job`` and start the device thread if none runs (after the one before
        has ended); a thread that does not start takes the job out again."""
        with self._lock:
            self._queue.append(job)
            if self._thread is not None:
                return
            if self._previous is not None:
                self._previous.join()
                self._previous = None
            thread = threading.Thread(target=self._serve, name="fse-mlx", daemon=True)
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                if thread.ident is None:  # it never ran: as before the call
                    self._thread = None
                    self._queue.remove(job)
                    job.state = "cancelled"
                raise

    def _abort(self, job: Job) -> None:
        """After an interrupt or error of the caller: take ``job`` out of the queue, or
        wait until the device thread has settled it, through further interrupts."""
        with self._lock:
            if job.state == "queued" and job in self._queue:
                self._queue.remove(job)
                job.state = "cancelled"
            if job.state == "cancelled":
                return
        while True:
            try:
                job.done.wait()
            except BaseException:  # the device thread may still write into its arrays
                continue
            job.result = None
            _give_back(job)
            return

    def _take(self, end: bool = True) -> list[Job]:
        """The next group, in queue order (with the lock held); with ``end``, an empty
        queue ends the thread."""
        with self._lock:
            group: list[Job] = []
            tokens = 0
            while self._queue and len(group) < self.group_tasks:
                if group and tokens + self._queue[0].tokens > self.group_tokens:
                    break
                job = self._queue.popleft()
                job.state = "running"
                tokens += job.tokens
                group.append(job)
            if not group and end:
                self._previous, self._thread = self._thread, None
            return group

    def _serve(self) -> None:
        """Launch a group while the one before computes on the GPU, then complete that one:
        at most two groups in flight, so the finish and the copies of one overlap the
        kernels of the next. With nothing queued, the group in flight completes at once."""
        group: list[Job] = []
        flight: list[tuple[Job, tuple]] = []
        lent: list[Job] = []
        try:
            mx = self.core()
            if self._stream is None:
                self._stream = mx.new_thread_unsafe_stream(mx.gpu)
            group = self._take()
            while group or flight:
                launched, failed = self._launch(mx, group)
                done = [*failed, *self._complete(mx, flight)]
                flight, group = launched, []
                group = self._take(end=not launched)
                lent = self._settle(done, lent, last=not group and not launched)
        except BaseException as err:  # a failure here must not leave a waiter hanging
            self._abandon([*group, *(job for job, _ in flight)], err)
            _collect(lent, wait=True)

    def _abandon(self, group: list[Job], err: BaseException) -> None:
        with self._lock:
            jobs = [*group, *self._queue]
            self._queue.clear()
            self._previous, self._thread = self._thread, None
        for job in jobs:
            job.error = job.error or err
            job.done.set()

    def _launch(self, mx: Any, group: list[Job]) -> tuple[list[tuple[Job, tuple]], list[Job]]:
        """Queue the kernel of every job of ``group`` and start their evaluation: the
        launched jobs with their arrays, and the jobs that failed (with their error)."""
        launched, failed = [], []
        for job in group:
            try:
                launched.append((job, job.launch(mx, self._stream)))
            except Exception as err:
                job.error = err
                failed.append(job)
        if not launched:
            return [], failed
        self.evaluations += 1
        self.launches += len(launched)
        try:
            mx.async_eval(*[array for _, arrays in launched for array in arrays])
        except Exception as err:
            for job, _ in launched:
                job.error = err
            return [], [*failed, *(job for job, _ in launched)]
        return launched, failed

    @staticmethod
    def _complete(mx: Any, launched: list[tuple[Job, tuple]]) -> list[Job]:
        """Wait for the evaluation of ``launched`` and finish each job (not yet done)."""
        if not launched:
            return []
        try:
            mx.eval(*[array for _, arrays in launched for array in arrays])
        except Exception as err:
            for job, _ in launched:
                job.error = err
            return [job for job, _ in launched]
        for job, arrays in launched:
            try:
                job.result = job.finish(arrays)
            except Exception as err:
                job.error = err
            else:
                job.arrays = arrays if job.lease else None
        return [job for job, _ in launched]

    @staticmethod
    def _settle(done: list[Job], lent: list[Job], last: bool) -> list[Job]:
        """Set ``done`` (the thread ends after them if ``last``) and take back the leases
        returned so far, all of them if ``last``; the leases still out."""
        for job in done:
            if job.error is not None:  # its frames would keep the arrays alive
                traceback.clear_frames(job.error.__traceback__)
            job.ended = threading.current_thread() if last else None
            if job.arrays is not None:
                lent.append(job)
            job.done.set()
        return _collect(lent, wait=last)


def _used(job: Job, use: Use | None) -> Any:
    """``use`` of the job's result in the caller. If it fails, its frames are cleared
    first, so that no view of a lent array outlives the lease in a traceback."""
    result, job.result = job.result, None
    if job.error is not None or use is None:
        return result
    try:
        return use(result)
    except BaseException as err:
        del result
        traceback.clear_frames(err.__traceback__)
        raise


def _give_back(job: Job) -> None:
    """Return the lease and join a device thread that ends after the job, through any
    interrupt (the first one is raised again afterwards), so that no thread outlives
    the call."""
    interrupted: BaseException | None = None
    while True:
        try:
            job.returned.set()
            if job.ended is not None:
                job.ended.join()  # the thread ends right after its last group and lease
            break
        except BaseException as err:
            interrupted = interrupted or err
    if interrupted is not None:
        raise interrupted


def _collect(lent: list[Job], wait: bool) -> list[Job]:
    """Drop the arrays of the jobs whose callers returned their lease (in the device
    thread, so MLX frees them here); with ``wait``, of all of them. The others remain."""
    remaining = []
    for job in lent:
        if wait:
            job.returned.wait()
        if job.returned.is_set():
            job.arrays = None
        else:
            remaining.append(job)
    return remaining
