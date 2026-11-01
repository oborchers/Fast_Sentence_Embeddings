#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The file work of a ShardSink in one thread of its own, in the order it was handed over.

The calling thread of a run reads the source, reserves rows and commits them; writing the
committed rows into their shard (``pwrite``), hashing them (the checksum, which must see the
rows in order) and publishing finished shards (syncs and a rename) would make it the
bound of the run. A ShardSink therefore hands this work to one writer thread through a
bounded queue: the order of the work stays the order of the commits, and a full queue
makes the calling thread wait, so the rows waiting for their write stay bounded. Both
``pwrite`` and hashing release the GIL. An error of the writer thread is kept and raised
at the next call of the sink; work handed over after it is dropped.

A thread that is abandoned (an interrupt while the sink closes) finishes the piece it
runs, drops the rest and ends. Whether it ended is an event the thread sets itself as its
last step: an interrupted ``Thread.join`` of Python 3.11 and 3.12 can report a thread
that still runs as ended (``_wait_for_tstate_lock`` stops it in its ``except``).
"""

from __future__ import annotations

import signal
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

# Work waiting for the writer thread (not counting the piece it runs): at most this many
# pieces and, with the piece running, this many bytes of rows, unless the thread is idle
# (one larger piece then passes alone).
MAX_PENDING = 16
MAX_PENDING_BYTES = 64 * 2**20
_POLL = 0.05  # seconds between checks while waiting, so a signal handler runs meanwhile

_Item = tuple[Callable[..., Any], tuple, int] | None


class WriterFailure:
    """The first failure of cooperating threads, reported once to their caller."""

    def __init__(self):
        self.error: BaseException | None = None
        self.reported = False
        self._lock = threading.Lock()

    def record(self, error: BaseException) -> None:
        with self._lock:
            if self.error is None:
                self.error = error

    def check(self, report: bool) -> None:
        if self.error is None:
            return
        with self._lock:
            error = self.error
            reported = self.reported
            self.reported |= report
        if not reported:
            raise error
        raise RuntimeError(f"the writer of the sink failed earlier: {error!r}") from error


class WriterThread:
    """Runs callables in one thread, in the order they were submitted.

    A daemon thread, so a sink that is never closed cannot keep the interpreter from
    exiting (its shard in progress then stays a temporary directory, which readers
    ignore). An error of the thread is raised at the next ``submit``, ``check`` or
    ``drain``, itself once and then as a RuntimeError naming it; work after it is
    dropped. Cooperating threads share a ``WriterFailure`` so a secondary failure never
    replaces the first one. Each thread still finishes its queued work unless it itself
    fails or is abandoned, so a failure elsewhere cannot close a file it still writes.

    The waiting pieces and the piece running are the whole state of the queue, kept
    under one condition: no counter is changed after a piece is queued, so an interrupt
    (Ctrl+C) of the caller at any point leaves a piece either queued or not.
    """

    def __init__(self, name: str = "fse-shard-writer", *, failure: WriterFailure | None = None):
        self._items: deque[_Item] = deque()
        self._state = threading.Condition()
        self._failure = failure if failure is not None else WriterFailure()
        self._error: BaseException | None = None  # only this thread's work stops
        self._abandoned = False
        self._accepting = True  # until the thread ends
        self._stopping = False
        self._ended = threading.Event()  # set by the thread itself, as its last step
        self._running = -1  # the bytes of the piece running, -1 for none
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(
        self, work: Callable[..., Any], *args: Any, nbytes: int = 0, report: bool = True
    ) -> None:
        """Hand ``work(*args)`` to the writer thread, waiting while MAX_PENDING pieces
        wait for it or, unless it is idle, this piece (``nbytes``) would bring the bytes
        waiting and running above MAX_PENDING_BYTES. Another writer thread that hands
        work on passes ``report=False``: an error it meets is not reported to the caller
        yet."""
        self.check(report)
        with self._state:
            while self._accepting and self._full(nbytes):
                self.check(report)
                self._state.wait(_POLL)
            self.check(report)
            if not self._accepting or self._abandoned:
                raise RuntimeError("the writer thread of the sink is stopped")
            # The thread wakes only once this releases the lock, so the piece is queued by
            # then; an interrupt between the two leaves nothing queued and nothing changed.
            self._state.notify_all()
            self._items.append((work, args, nbytes))

    def drain(self) -> None:
        """Wait until all work handed over so far is done, then raise its error, if any."""
        self.settle()
        self.check()

    def check(self, report: bool = True) -> None:
        """Raise the error of the thread: itself until it is reported to the caller (a
        call with ``report``), then a RuntimeError that names it, so no work is ever
        dropped without an error."""
        self._failure.check(report)

    @property
    def reported(self) -> bool:
        """Whether a call has raised the error of the thread."""
        return self._failure.reported

    def settle(self) -> None:
        """Wait until all work handed over so far is done or dropped, without raising."""
        with self._state:
            while self._items or self._running >= 0:
                self._state.wait(_POLL)

    @property
    def failed(self) -> bool:
        """Whether the work after now is dropped: the thread failed or was abandoned."""
        return self._error is not None or self._abandoned

    def abandon(self) -> None:
        """Drop the work still waiting: the piece running finishes, nothing after it runs,
        and the thread ends once it reaches its next piece (or the end ``stop`` hands it).
        It only sets a flag, so a signal handler may call it at any point."""
        self._abandoned = True

    def stop(self) -> None:
        """Finish the work handed over (unless an error or ``abandon`` drops it) and end
        the thread; returns once the thread has ended its last piece, which has no time
        bound while a piece blocks (a write to a hanging drive)."""
        if not self._stopping:
            with self._state:
                self._state.notify_all()
                self._items.append(None)
            self._stopping = True
        while not self._ended.wait(_POLL):
            pass

    def _full(self, nbytes: int) -> bool:
        busy = sum(item[2] for item in self._items if item is not None) + max(self._running, 0)
        idle = not self._items and self._running < 0
        return len(self._items) >= MAX_PENDING or (not idle and busy + nbytes > MAX_PENDING_BYTES)

    def _run(self) -> None:
        try:
            while self._step(self._take()):
                pass
        finally:
            with self._state:  # take no more work and drop what still waits
                self._accepting = False
                self._items.clear()
                self._running = -1
                self._state.notify_all()
            self._ended.set()

    def _take(self) -> _Item:
        """The next piece, marked as running; the one before is done."""
        with self._state:
            self._running = -1
            self._state.notify_all()  # a producer or settle waits for that
            while not self._items:
                self._state.wait()
            item = self._items.popleft()
            if item is not None:
                self._running = item[2]
            return item

    def _step(self, item: _Item) -> bool:
        """Run one piece, unless an error or ``abandon`` drops it; whether to go on."""
        if item is None or self._abandoned:
            return False
        work, args, _ = item
        try:
            if self._error is None:
                work(*args)
        except BaseException as error:  # kept for the calling thread
            self._failure.record(error)
            self._error = self._error or error
        return True


@contextmanager
def deferred_interrupts(on_interrupt: Callable[[], None]) -> Iterator[list]:
    """Run the block with SIGINT (Ctrl+C) only noted, if it has Python's default handler
    (``signal.default_int_handler``) and this is the main thread: the first one calls
    ``on_interrupt`` at once (from the signal handler, so it must only set flags), and one
    KeyboardInterrupt is raised after the block. Yields the list of the interrupts noted.

    With any other handler (a handler of the program, ignored, the default action, or a
    handler of C code) or in another thread nothing is deferred, and ``on_interrupt`` is
    never called.
    """
    noted: list[tuple[int, Any]] = []
    previous = signal.getsignal(signal.SIGINT)
    main = threading.current_thread() is threading.main_thread()
    if not main or previous is not signal.default_int_handler:
        yield noted
        return

    def note(signum: int, frame: Any) -> None:
        if not noted:
            on_interrupt()
        noted.append((signum, frame))

    signal.signal(signal.SIGINT, note)
    try:
        yield noted
    finally:
        signal.signal(signal.SIGINT, previous)
        if noted:
            raise KeyboardInterrupt
