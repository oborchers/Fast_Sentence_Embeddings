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

A job's ``launch`` (queue the kernel) and ``finish`` (read the results into host arrays)
run in the device thread. An error in one job's launch or finish fails that job only;
an error of the evaluation fails every job of the group. A waiter that is interrupted
(KeyboardInterrupt in the calling thread) takes its job out of the queue, or waits until
the device thread is done with it, so nothing writes into its arrays after the call.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from . import _kernels, _load

GROUP_TASKS = 8
GROUP_TOKENS = 1 << 24

Launch = Callable[[Any, Any], tuple]
Finish = Callable[[tuple], Any]


class Job:
    """One call: its launch and finish, its token count, and its outcome."""

    __slots__ = ("done", "ended", "error", "finish", "launch", "result", "state", "tokens")

    def __init__(self, launch: Launch, finish: Finish, tokens: int):
        self.launch, self.finish, self.tokens = launch, finish, tokens
        self.state = "queued"  # then "running"; a cancelled job leaves the queue
        self.done = threading.Event()
        self.ended: threading.Thread | None = None  # the device thread, if it ends after it
        self.result: Any = None
        self.error: BaseException | None = None


class Device:
    """The queue and the device thread; one per process (``DEVICE``), for every encoder."""

    def __init__(self, group_tasks: int = GROUP_TASKS, group_tokens: int = GROUP_TOKENS):
        self.group_tasks, self.group_tokens = group_tasks, group_tokens
        self._lock = threading.Lock()
        self._queue: deque[Job] = deque()
        self._thread: threading.Thread | None = None
        self._previous: threading.Thread | None = None
        self._stream: Any = None
        self._canary: BaseException | bool = False
        self.evaluations = 0  # groups evaluated, and jobs launched (for the tests)
        self.launches = 0

    def run(self, launch: Launch, finish: Finish, tokens: int = 0) -> Any:
        """Run one job in the device thread and give its result, or raise its error."""
        job = Job(launch, finish, tokens)
        with self._lock:
            self._queue.append(job)
            if self._thread is None:
                self._start()
        self._wait(job)
        if job.ended is not None:
            job.ended.join()  # the thread ends right after it settled its last group
        if job.error is not None:
            raise job.error
        return job.result

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
                self.call(_kernels.canary)
            except Exception as err:
                known = err
            else:
                known = True
            with self._lock:
                self._canary = known
        if isinstance(known, BaseException):
            raise known

    def _start(self) -> None:
        """Start the device thread (with the lock held), after the one before has ended."""
        if self._previous is not None:
            self._previous.join()
        self._thread = threading.Thread(target=self._serve, name="fse-mlx", daemon=True)
        self._thread.start()

    def _wait(self, job: Job) -> None:
        try:
            job.done.wait()
        except BaseException:
            with self._lock:
                queued = job.state == "queued"
                if queued:
                    self._queue.remove(job)
            if not queued:
                job.done.wait()
            raise

    def _take(self) -> list[Job]:
        """The next group, in queue order; none ends the thread (with the lock held)."""
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
            if not group:
                self._previous, self._thread = self._thread, None
            return group

    def _serve(self) -> None:
        group: list[Job] = []
        try:
            mx = _load.core()
            if self._stream is None:
                self._stream = mx.new_thread_unsafe_stream(mx.gpu)
            group = self._take()
            while group:
                self._run(mx, group)
                following = self._take()
                for job in group:
                    job.ended = None if following else threading.current_thread()
                    job.done.set()
                group = following
        except BaseException as err:  # a failure here must not leave a waiter hanging
            self._abandon(group, err)

    def _abandon(self, group: list[Job], err: BaseException) -> None:
        with self._lock:
            jobs = [*group, *self._queue]
            self._queue.clear()
            self._previous, self._thread = self._thread, None
        for job in jobs:
            job.error = job.error or err
            job.done.set()

    def _run(self, mx: Any, group: list[Job]) -> None:
        """Launch every job of ``group``, evaluate them together, then finish each."""
        launched = []
        for job in group:
            try:
                launched.append((job, job.launch(mx, self._stream)))
            except Exception as err:
                job.error = err
        self.evaluations += 1
        self.launches += len(launched)
        try:
            mx.eval(*[array for _, arrays in launched for array in arrays])
        except Exception as err:
            for job, _ in launched:
                job.error = err
            return
        for job, arrays in launched:
            try:
                job.result = job.finish(arrays)
            except Exception as err:
                job.error = err


DEVICE = Device()
