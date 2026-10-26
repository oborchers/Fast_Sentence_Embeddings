#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Training into a ring of float32 windows, for sentence vectors stored as float16 or int8.

With index i for the i-th sentence, every job writes its own contiguous rows and a row is
final once its job is done. Instead of a float32 matrix for all rows, the kernel then
writes into a ring of ``n_windows`` windows of ``window_rows`` rows (``KernelContext.
use_ring``). Once every row of the next window is done, the worker that finished it hands
the window to ``finalize(first_row, rows)`` (which sums its Gram blocks, or removes the
principal components and stores it in the target format) and frees its slot. Windows
are finalized in row order by one thread at a time, outside the lock, so the stored rows
and Gram matrix are bit-identical to those of a float32 training. The producer waits
before it hands out a job whose rows would land in a slot that is not free yet.
"""

import math
import threading
from collections.abc import Callable
from typing import Any

from numpy import empty, ndarray
from numpy import float32 as REAL

from fse.models import utils

# Bytes of one window; the ring holds at least two.
WINDOW_BYTES = 256 * 1024**2


def window_rows(dim: int, rows: int) -> int:
    """Rows of a window: a multiple of REMOVAL_BLOCK_ROWS of about WINDOW_BYTES, at most rows."""
    block_rows = utils.REMOVAL_BLOCK_ROWS
    blocks = max(1, WINDOW_BYTES // max(dim * REAL(0).itemsize * block_rows, 1))
    return max(1, min(int(blocks) * block_rows, rows))


class WindowedWriter:
    """Finished rows of a ring of windows, finalized in order; lives only during train."""

    def __init__(
        self,
        n_rows: int,
        dim: int,
        finalize: Callable[[int, ndarray], Any],
        max_job_rows: int,
        rows_per_window: int | None = None,
    ):
        self.n_rows = n_rows
        self.window_rows = rows_per_window or window_rows(dim, n_rows)
        # A job of r rows touches at most ceil(r / W) + 1 windows; all of them must fit
        # into the ring together, or its producer would wait for a slot forever.
        self.num_windows = -(-n_rows // self.window_rows)
        self.n_windows = max(2, math.ceil(max(max_job_rows, 1) / self.window_rows) + 1)
        # A ring never needs more slots than there are windows; small inputs fit one.
        self.n_windows = max(1, min(self.n_windows, self.num_windows))
        self.ring = empty((self.window_rows * self.n_windows, dim), dtype=REAL)
        self._finalize = finalize
        self._written = [0] * self.num_windows
        self._next = 0
        self._finalizing = False
        self._failed = False
        self._cond = threading.Condition()

    def _window_size(self, k: int) -> int:
        return min(self.window_rows, self.n_rows - k * self.window_rows)

    def wait_for(self, end_row: int, stop: threading.Event | None = None):
        """Wait until rows up to end_row (exclusive) have a free slot in the ring.

        Returns early once ``stop`` is set or a finalizer failed; the training then ends
        with that error, so no job is handed out.
        """
        limit = lambda: (self._next + self.n_windows) * self.window_rows  # noqa: E731
        with self._cond:
            while end_row > limit():
                if self._failed or (stop is not None and stop.is_set()):
                    return
                self._cond.wait(0.1)

    def rows_done(self, first: int, last: int):
        """Report that rows first to last (exclusive) are final; finalize complete windows."""
        if not 0 <= first <= last <= self.n_rows:
            raise ValueError(f"rows [{first}, {last}) are outside the {self.n_rows} rows")
        with self._cond:
            k = first // self.window_rows
            while first < last:
                end = min(last, (k + 1) * self.window_rows)
                self._written[k] += end - first
                if self._written[k] > self._window_size(k):
                    raise RuntimeError(f"more rows of window {k} were reported than it has")
                first, k = end, k + 1
        self._finalize_in_order()

    def _finalize_in_order(self):
        """Finalize complete windows from _next on, by one thread at a time."""
        with self._cond:
            if self._finalizing:
                return
            self._finalizing = True
        try:
            while True:
                with self._cond:
                    k = self._next
                    if k == self.num_windows or self._written[k] != self._window_size(k):
                        self._finalizing = False
                        return
                slot = (k % self.n_windows) * self.window_rows
                self._finalize(k * self.window_rows, self.ring[slot : slot + self._window_size(k)])
                with self._cond:
                    self._next += 1
                    self._cond.notify_all()
        except BaseException:
            with self._cond:
                self._finalizing = False
                self._failed = True
                self._cond.notify_all()
            raise

    def finish(self):
        """Raise unless every window was finalized."""
        if self._next != self.num_windows:
            raise RuntimeError(
                f"only {self._next} of {self.num_windows} windows of sentence vectors were "
                "finalized"
            )
