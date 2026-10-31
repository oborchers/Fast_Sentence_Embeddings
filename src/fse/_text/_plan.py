#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The cuts of a source whose sentences have offsets: chunks of about equal bytes.

The k pieces, ``max(ceil(total / chunk_bytes), ceil(n / chunk_rows))`` of them, start at
the first sentence that starts at or after ``j * total // k`` (a sentence is never
split), and a piece that still has more than ``chunk_rows`` sentences is split into
parts of about equal rows. So no chunk holds
more than ``chunk_bytes`` plus its last sentence, and the work is bounded by the cuts it
makes, not by the bytes: the targets inside one sentence are skipped at once
(docs/architecture/runtime.md, "Runtime", Tasks).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from numpy import ndarray


def offset_cuts(offsets: ndarray, chunk_bytes: int, chunk_rows: int) -> ndarray:
    """The positions 0, the cuts and n of the n sentences of ``offsets`` (int64, n + 1
    entries, not decreasing): each chunk from one position to the next."""
    n = len(offsets) - 1
    if n <= 0:
        return np.zeros(max(n + 1, 1), dtype=np.int64)[: n + 1]
    total = int(offsets[-1]) - int(offsets[0])
    # As many pieces as the larger of both bounds asks for, so where the rows bind the
    # pieces are cut by bytes at once rather than each split by rows afterwards.
    pieces = max(1, -(-total // chunk_bytes), -(-n // chunk_rows))
    cuts = [0, *_byte_cuts(offsets, pieces), n]
    return np.array(_row_cuts(cuts, chunk_rows), dtype=np.int64)


def _byte_cuts(offsets: ndarray, pieces: int) -> list[int]:
    """The cuts into ``pieces`` of about equal bytes, one loop step per cut, in Python
    ints (no overflow)."""
    n = len(offsets) - 1
    base = int(offsets[0])
    total = int(offsets[-1]) - base
    if total == 0:
        return []
    cuts: list[int] = []
    j = 1
    while j < pieces:
        target = base + j * total // pieces
        cut = int(np.searchsorted(offsets, target, side="left"))
        if cut >= n:
            break
        if cut > (cuts[-1] if cuts else 0):
            cuts.append(cut)
        # The next target behind the start of the sentence cut: (j * total) // pieces
        # > offsets[cut] - base, that is j >= ceil((offsets[cut] - base + 1) * pieces / total).
        j = max(j + 1, -(-(int(offsets[cut]) - base + 1) * pieces // total))
    return cuts


def _row_cuts(cuts: list[int], chunk_rows: int) -> list[int]:
    """``cuts`` with every piece of more than ``chunk_rows`` sentences split into parts
    whose rows differ by at most one."""
    out = [cuts[0]]
    for start, end in pairwise(cuts):
        parts = -(-(end - start) // chunk_rows)
        out.extend(start + (end - start) * p // parts for p in range(1, parts + 1))
    return out
