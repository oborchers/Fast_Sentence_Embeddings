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
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

# Work waiting for the writer thread: at most this many pieces and, unless a single piece
# is larger, this many bytes of rows.
MAX_PENDING = 16
MAX_PENDING_BYTES = 64 * 2**20


class WriterThread:
    """Runs callables in one thread, in the order they were submitted.

    A daemon thread, so a sink that is never closed cannot keep the interpreter from
    exiting (its shard in progress then stays a temporary directory, which readers
    ignore). An error of the thread is raised at the next ``submit``, ``check`` or
    ``drain``, itself once and then as a RuntimeError naming it; work after it is
    dropped.
    """

    def __init__(self, name: str = "fse-shard-writer"):
        self._queue: queue.Queue[tuple[Callable[..., Any], tuple, int] | None] = queue.Queue(
            MAX_PENDING
        )
        self._error: BaseException | None = None
        self._reported = False
        self._budget = threading.Condition()
        self._pending_bytes = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, work: Callable[..., Any], *args: Any, nbytes: int = 0) -> None:
        """Hand ``work(*args)`` to the writer thread, waiting while the work waiting for it
        holds more than MAX_PENDING_BYTES (``nbytes`` is the size of this piece)."""
        self.check()
        if not self._thread.is_alive():
            raise RuntimeError("the writer thread of the sink is stopped")
        with self._budget:
            while self._pending_bytes and self._pending_bytes + nbytes > MAX_PENDING_BYTES:
                self._budget.wait()
            self._pending_bytes += nbytes
        self._queue.put((work, args, nbytes))

    def drain(self) -> None:
        """Wait until all work handed over so far is done, then raise its error, if any."""
        self._queue.join()
        self.check()

    def check(self) -> None:
        """Raise the error of the thread: itself the first time, then a RuntimeError that
        names it, so no work is ever dropped without an error."""
        if self._error is None:
            return
        if not self._reported:
            self._reported = True
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
        self._queue.join()

    @property
    def failed(self) -> bool:
        return self._error is not None

    def stop(self) -> None:
        """Finish the work handed over (unless an error stopped it) and end the thread."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                work, args, nbytes = item
                try:
                    if self._error is None:
                        work(*args)
                finally:
                    with self._budget:  # also after an error, so no producer waits forever
                        self._pending_bytes -= nbytes
                        self._budget.notify_all()
            except BaseException as error:  # kept for the calling thread
                self._error = self._error or error
            finally:
                self._queue.task_done()
