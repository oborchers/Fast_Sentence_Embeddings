#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

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

import queue
import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

# Work waiting for the writer thread: at most this many pieces and, unless a single piece
# is larger, this many bytes of rows.
MAX_PENDING = 16
MAX_PENDING_BYTES = 64 * 2**20
_POLL = 0.05  # seconds between checks while waiting, so a signal handler runs meanwhile


class WriterThread:
    """Runs callables in one thread, in the order they were submitted.

    A daemon thread, so a sink that is never closed cannot keep the interpreter from
    exiting (its shard in progress then stays a temporary directory, which readers
    ignore). An error of the thread is raised at the next ``submit``, ``check`` or
    ``drain``, itself once and then as a RuntimeError naming it; work after it is
    dropped.
    """

    def __init__(self, name: str = "fse-shard-writer"):
        self._items: queue.SimpleQueue[tuple[Callable[..., Any], tuple, int] | None] = (
            queue.SimpleQueue()
        )
        self._error: BaseException | None = None
        self._reported = False
        self._abandoned = False
        # Pieces handed over and not yet done or dropped, their bytes, and whether the
        # thread takes more; all under the condition.
        self._state = threading.Condition()
        self._pending = self._pending_bytes = 0
        self._accepting = True
        self._stopping = False
        self._ended = threading.Event()  # set by the thread itself, as its last step
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(
        self, work: Callable[..., Any], *args: Any, nbytes: int = 0, report: bool = True
    ) -> None:
        """Hand ``work(*args)`` to the writer thread, waiting while more than MAX_PENDING
        pieces or, unless it is the only one, MAX_PENDING_BYTES of work wait for it
        (``nbytes`` is the size of this piece). Another writer thread that hands work on
        passes ``report=False``: an error it meets is not reported to the caller yet."""
        self.check(report)
        with self._state:
            while self._accepting and self._pending and self._full(nbytes):
                self._state.wait(_POLL)
            if not self._accepting or self._abandoned:
                raise RuntimeError("the writer thread of the sink is stopped")
            self._pending += 1
            self._pending_bytes += nbytes
            try:
                self._items.put((work, args, nbytes))
            except BaseException:  # interrupted (Ctrl+C): the piece never waits
                self._pending -= 1
                self._pending_bytes -= nbytes
                raise

    def drain(self) -> None:
        """Wait until all work handed over so far is done, then raise its error, if any."""
        self.settle()
        self.check()

    def check(self, report: bool = True) -> None:
        """Raise the error of the thread: itself until it is reported to the caller (a
        call with ``report``), then a RuntimeError that names it, so no work is ever
        dropped without an error."""
        if self._error is None:
            return
        if not self._reported:
            self._reported = report
            raise self._error
        raise RuntimeError(
            f"the writer of the sink failed earlier: {self._error!r}"
        ) from self._error

    @property
    def reported(self) -> bool:
        """Whether a call has raised the error of the thread."""
        return self._reported

    def settle(self) -> None:
        """Wait until all work handed over so far is done or dropped, without raising."""
        with self._state:
            while self._pending:
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
            self._items.put(None)
            self._stopping = True
        while not self._ended.wait(_POLL):
            pass

    def _full(self, nbytes: int) -> bool:
        return self._pending > MAX_PENDING or self._pending_bytes + nbytes > MAX_PENDING_BYTES

    def _run(self) -> None:
        try:
            while self._step(self._items.get()):
                pass
        finally:
            self._drop_waiting()
            self._ended.set()

    def _step(self, item: tuple[Callable[..., Any], tuple, int] | None) -> bool:
        """Run one piece, unless an error or ``abandon`` drops it; whether to go on."""
        if item is None:
            return False
        work, args, nbytes = item
        try:
            if self._abandoned:
                return False
            if self._error is None:
                work(*args)
        except BaseException as error:  # kept for the calling thread
            self._error = self._error or error
        finally:
            self._finished(nbytes)
        return True

    def _finished(self, nbytes: int) -> None:
        with self._state:  # also after an error, so no producer waits forever
            self._pending -= 1
            self._pending_bytes -= nbytes
            self._state.notify_all()

    def _drop_waiting(self) -> None:
        """At the end of the thread: take no more work and drop what still waits."""
        with self._state:
            self._accepting = False
            while True:
                try:
                    item = self._items.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    self._pending -= 1
                    self._pending_bytes -= item[2]
            self._state.notify_all()


@contextmanager
def deferred_interrupts(on_interrupt: Callable[[], None]) -> Iterator[None]:
    """Run the block with SIGINT (Ctrl+C) only noted: the first one calls ``on_interrupt``
    at once (from the signal handler, so it must only set flags), and the handler that
    was installed runs once after the block, which raises KeyboardInterrupt by default.

    Only Python raises KeyboardInterrupt, and only in the main thread, so the block runs
    as is in another thread or when SIGINT has no Python handler (ignored, the default
    action, or a handler of C code).
    """
    previous = signal.getsignal(signal.SIGINT)
    if threading.current_thread() is not threading.main_thread() or not callable(previous):
        yield
        return
    noted: list[tuple[int, Any]] = []

    def note(signum: int, frame: Any) -> None:
        if not noted:
            on_interrupt()
        noted.append((signum, frame))

    signal.signal(signal.SIGINT, note)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if noted:
            previous(*noted[0])
