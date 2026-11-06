#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The partition rule of a source that is cut into parts (docs/architecture/store.md,
"Resume stores").

``shard=(i, n)`` of ``Encoder.encode`` takes part ``i`` of ``n`` of the rows of a source.
The parts move in blocks of 16,384 rows, so the rule is cheap to compute, never splits a
default shard, and gives the same cut to every process that applies it. It freezes with
fse 2.1: stores record the cut they were written with.
"""

from __future__ import annotations

BLOCK_ROWS = 16_384


def partition(rows: int, index: int, count: int) -> tuple[int, int]:
    """The source rows ``[first, last)`` of part ``index`` of ``count`` parts of a source of
    ``rows`` rows: whole blocks of 16,384 rows, contiguous, disjoint and complete. A part
    can be empty; the last one ends at ``rows``."""
    blocks = -(-rows // BLOCK_ROWS)
    first = BLOCK_ROWS * (index * blocks // count)
    last = min(rows, BLOCK_ROWS * ((index + 1) * blocks // count))
    return min(first, rows), last
