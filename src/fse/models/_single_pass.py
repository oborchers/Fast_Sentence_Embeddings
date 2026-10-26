#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""One pass over the sentence vectors during training.

With index i for the i-th sentence, every job covers its own contiguous rows, and a row is
final once its job is done. Training then lets the kernel set the rows of each job to EPS
right before averaging, and SIF and uSIF sum the Gram matrix of the finished rows while
they are still cached, instead of separate passes over all rows before and after.
"""

import threading
from typing import Any

from numpy import asarray, float64, ndarray, zeros

from fse.models import utils

# Block products held while their predecessors are missing, in bytes. Beyond it, a
# finished block is only recorded and its product computed again when it is summed.
PENDING_BUDGET_BYTES = 256 * 1024**2


class OrderedGram:
    """The Gram matrix of rows written during training, identical to gram_matrix.

    Workers report finished row ranges with add_rows. Once all rows of a block of
    ``REMOVAL_BLOCK_ROWS`` rows are reported, the worker that finished it computes the
    block product while the rows are cached, and the products are summed in block order.
    The result is therefore bit-identical to gram_matrix on the finished vectors, however
    the jobs were distributed over the workers. At most PENDING_BUDGET_BYTES of products
    wait for a missing predecessor, besides the product of the block summed next; further
    blocks are recomputed from their rows when they are summed. One thread at a time sums,
    outside the lock.
    """

    def __init__(self, target: ndarray, first_row: int = 0):
        self.target = target
        self.first_row = first_row
        self.num_rows = target.shape[0] - first_row
        self.block_dtype = utils.gram_block_dtype(target.dtype)
        dim = target.shape[1]
        self.gram = zeros((dim, dim), dtype=float64)
        self._block_rows = utils.REMOVAL_BLOCK_ROWS
        self.num_blocks = -(-self.num_rows // self._block_rows)
        product_bytes = dim * dim * asarray(0, dtype=self.block_dtype).itemsize
        # Possibly 0: then every block that waits is recomputed.
        self.max_pending = PENDING_BUDGET_BYTES // max(product_bytes, 1)
        self._written = [0] * self.num_blocks
        # Block number to its product, or to None for a finished block to recompute.
        self._pending: dict[int, ndarray | None] = {}
        # Blocks whose products are held in _pending or being computed for it, reserved
        # before computing; never more than max_pending.
        self._reserved: set[int] = set()
        self._next = 0
        # Whether a thread is summing; only that thread touches gram and _next.
        self._summing = False
        self._lock = threading.Lock()

    def _block(self, k: int) -> ndarray:
        begin = self.first_row + k * self._block_rows
        size = min(self._block_rows, self.num_rows - k * self._block_rows)
        return asarray(self.target[begin : begin + size], dtype=self.block_dtype)

    def _block_size(self, k: int) -> int:
        return min(self._block_rows, self.num_rows - k * self._block_rows)

    def add_rows(self, first: int, last: int):
        """Report that rows first to last (exclusive, rows of target) are final."""
        first -= self.first_row
        last -= self.first_row
        if not 0 <= first <= last <= self.num_rows:
            raise ValueError(f"rows [{first}, {last}) are outside the {self.num_rows} rows")
        done = []
        with self._lock:
            k = first // self._block_rows
            while first < last:
                end = min(last, (k + 1) * self._block_rows)
                self._written[k] += end - first
                if self._written[k] > self._block_size(k):
                    raise RuntimeError(f"more rows of block {k} were reported than it has")
                if self._written[k] == self._block_size(k):
                    done.append(k)
                first, k = end, k + 1
        for k in done:
            with self._lock:
                # The block that is summed next is never held; others only within budget,
                # reserved before the product is computed.
                keep = k != self._next and len(self._reserved) < self.max_pending
                if keep:
                    self._reserved.add(k)
            if keep or k == self._next:
                try:
                    product = utils.gram_block_product(self._block(k))
                except BaseException:
                    with self._lock:
                        self._reserved.discard(k)
                    raise
            else:
                product = None
            with self._lock:
                self._pending[k] = product
                # Only _pending holds it now, so the summed product is freed with it.
                del product
        self._sum_in_order()

    def _sum_in_order(self):
        """Sum the products from block _next on while they are there, by one thread at a time.

        Recomputing a block that was not held happens outside the lock, so other workers can
        report their rows meanwhile; a block reported then is summed by the running thread.
        """
        with self._lock:
            if self._summing:
                return
            self._summing = True
        try:
            while True:
                with self._lock:
                    if self._next not in self._pending:
                        self._summing = False
                        return
                    held = self._pending.pop(self._next)
                    self._reserved.discard(self._next)
                if held is None:
                    held = utils.gram_block_product(self._block(self._next))
                utils.add_gram_block(self.gram, held)
                del held
                with self._lock:
                    self._next += 1
        except BaseException:
            with self._lock:
                self._summing = False
            raise

    @property
    def complete(self) -> bool:
        return self._next == self.num_blocks

    def result(self) -> ndarray:
        """The Gram matrix; raises unless every row was reported."""
        if not self.complete:
            raise RuntimeError(
                f"only {self._next} of {self.num_blocks} blocks of the Gram matrix are complete"
            )
        return self.gram


class SinglePass:
    """State of one training call whose jobs prefill their rows; lives only during train.

    The producer checks that the jobs cover the new rows in order, without gaps or
    overlaps, before a worker writes them; workers report finished rows to the Gram
    matrix, if one is summed.
    """

    def __init__(self, target: ndarray, first_row: int, sum_gram: bool):
        self.first_row = first_row
        self.end_row = target.shape[0]
        self._expected = first_row
        self.gram = OrderedGram(target, first_row) if sum_gram else None

    def check_job(self, job: Any):
        """Raise unless the job starts at the next row and its rows are contiguous."""
        if not getattr(job, "contiguous", True) or job.first_row != self._expected:
            raise RuntimeError(
                f"the input yields other indices than the scan found: a job starts at row "
                f"{job.first_row} instead of {self._expected}"
            )
        self._expected += job.n_items
        if self._expected > self.end_row:
            raise RuntimeError("the input yields more sentences than the scan found")

    def rows_done(self, first: int, last: int):
        if self.gram is not None:
            self.gram.add_rows(first, last)

    def finish(self) -> ndarray | None:
        """Raise unless the jobs covered every new row; return the Gram matrix, if summed."""
        if self._expected != self.end_row:
            raise RuntimeError(
                f"training wrote {self._expected - self.first_row} of "
                f"{self.end_row - self.first_row} rows; the input yielded fewer sentences "
                "than the scan found"
            )
        return None if self.gram is None else self.gram.result()
