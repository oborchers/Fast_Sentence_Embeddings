#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Whether sentence indices repeat, in about 1 byte per sentence instead of a set.

The scan needs to know whether every index occurs at most once. For indices that are not
the positions of their sentences, a set of Python ints costs 60 to 70 bytes per sentence.
``IndexTracker`` marks indices in a bitmap instead, one bit per index value, which is at
most ``max(BITMAP_MIN_BYTES, 1 byte per sentence)`` large. An index beyond the bitmap
waits as int64 in ``pending`` until the number of sentences allows a bitmap that covers
it. Valid indices are below the number of sentences, so a permutation of n sentences
needs at most about n bytes on the way and ends with a bitmap of n / 8 bytes; indices
beyond any affordable bitmap, which the scan rejects afterwards, cost 8 bytes each and
are sorted once at the end.

``scan_indexed_sentences`` in ``average_inner.pyx`` has a compiled twin with the same
rules that calls ``fold`` and ``sorted_unique`` below; tests compare both.
"""

from array import array

import numpy

# Below this size the bitmap may always grow, whatever the number of sentences.
BITMAP_MIN_BYTES = 1 << 20

# Indices are tracked as int64 and the output has max_index + 1 rows.
MAX_INDEX = 2**63 - 2


def check_index_range(index: int, position: int):
    """Raise for an index whose row count would not fit into int64."""
    if index > MAX_INDEX:
        raise ValueError(
            f"At {position}: Passed index {index}, larger than the largest supported index "
            f"{MAX_INDEX}"
        )


def bitmap_limit(sentences: int) -> int:
    """The largest bitmap in bytes for this number of sentences."""
    return max(BITMAP_MIN_BYTES, sentences)


def new_bitmap(seen: int) -> bytearray:
    """A bitmap with the indices 0 to seen - 1 marked."""
    bits = bytearray(max(seen // 8 + 1, 64))
    bits[: seen // 8] = b"\xff" * (seen // 8)
    bits[seen // 8] = (1 << (seen % 8)) - 1
    return bits


def fold(bits: bytearray, pending: array, pending_max: int, sentences: int) -> bool:
    """Grow the bitmap to cover the pending indices and mark them; False on a repeat.

    The caller has checked that ``pending_max // 8 < bitmap_limit(sentences)``. The bitmap
    at least doubles, up to the limit, so growing it costs amortized constant time.
    """
    size = max(pending_max // 8 + 1, min(2 * len(bits), bitmap_limit(sentences)))
    bits.extend(bytes(size - len(bits)))
    values = numpy.frombuffer(pending, dtype=numpy.int64)
    view = numpy.frombuffer(bits, dtype=numpy.uint8)
    try:
        positions = values >> 3
        masks = (1 << (values & 7)).astype(numpy.uint8)
        if (view[positions] & masks).any() or not _unique(values):
            return False
        numpy.bitwise_or.at(view, positions, masks)
    finally:
        del values, view  # both arrays can only change size without exported buffers
    del pending[:]
    return True


def sorted_unique(pending: array) -> bool:
    """Whether no pending index occurs twice; sorts them in place."""
    values = numpy.frombuffer(pending, dtype=numpy.int64)
    try:
        values.sort()
        return not bool((values[1:] == values[:-1]).any())
    finally:
        del values


def _unique(values: numpy.ndarray) -> bool:
    ordered = numpy.sort(values)
    return not bool((ordered[1:] == ordered[:-1]).any())


class IndexTracker:
    """Tracks indices after the first ``seen`` sentences had the indices 0 to seen - 1."""

    def __init__(self, seen: int):
        self.bits = new_bitmap(seen)
        self.pending = array("q")
        self.pending_max = -1

    def add(self, index: int, sentences: int) -> bool:
        """Record ``index`` of the ``sentences``-th sentence; False if it occurred before."""
        byte = index >> 3
        if byte < len(self.bits):
            mask = 1 << (index & 7)
            if self.bits[byte] & mask:
                return False
            self.bits[byte] |= mask
            return True
        self.pending.append(index)
        self.pending_max = max(self.pending_max, index)
        if self.pending_max // 8 < bitmap_limit(sentences):
            ok = fold(self.bits, self.pending, self.pending_max, sentences)
            self.pending_max = -1
            return ok
        return True

    def unique(self, sentences: int) -> bool:
        """Whether no index occurred twice; call once after the last ``add``."""
        if not self.pending:
            return True
        if self.pending_max // 8 < bitmap_limit(sentences):
            return fold(self.bits, self.pending, self.pending_max, sentences)
        # The pending indices all lie beyond the bitmap, so they can only repeat each other.
        return sorted_unique(self.pending)
