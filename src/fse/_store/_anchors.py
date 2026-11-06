#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The anchors a sink gives its shards in the mode ``source_position@1`` (docs/architecture/
store.md, "Stream positions").

A run reads batches; each non-empty batch ``b`` has the position ``P_b`` of its first record,
and its first record is the store row ``R_b``. The sink keeps the anchors of the current batch
and of the one before it. A shard that ends at row ``E`` (its last row is ``E - 1``) takes the
anchor of the current batch if that batch starts at or before ``E - 1``, else that of the
batch before. Inside one ``encode_batches`` every batch is committed before the next is read,
so the last row of a shard is always in the current batch or in the one before it, and the
rule gives each shard an anchor with ``R < E``: the failing batch of a closed sink, which
committed nothing into the shard, is not the shard's anchor.
"""

from __future__ import annotations

from collections.abc import Mapping

from ._position_format import SourcePosition, check_position


class Anchors:
    """The anchor of the current batch and of the batch before it, and the last anchor given.

    No anchor lies before the last one given (the rows of the chain do not decrease): when two
    stores resume together, the longer one meets batches that start before its own last
    anchor, and it keeps that anchor, which stays true (the record of its row is still at its
    position, and the row is before the end of the new shard)."""

    def __init__(self, last: SourcePosition | None = None) -> None:
        self.previous: SourcePosition | None = None
        self.current: SourcePosition | None = None
        self.last = last  # the anchor of the last shard of the store, then of the last given

    def begin(self, position: str, row: int) -> None:
        """A non-empty batch whose first record is the store row ``row`` and whose position
        is ``position`` starts."""
        self.previous, self.current = self.current, SourcePosition(check_position(position), row)

    def take(self, end: int, shared: Mapping[int, SourcePosition] | None = None) -> SourcePosition:
        """The anchor of a shard that ends at the store row ``end`` (exclusive). ``shared`` are
        the anchors of the shards another store holds at the ends of its shards: a shard that
        ends at one of them takes it, so two stores that resume together keep one history."""
        found = shared[end] if shared is not None and end in shared else self._by_batch(end)
        if self.last is not None and found.row < self.last.row:
            found = self.last
        self.last = found
        return found

    def _by_batch(self, end: int) -> SourcePosition:
        current = self.current
        if current is not None and end - 1 >= current.row:
            return current
        if self.previous is not None:
            return self.previous
        if self.last is None:
            raise RuntimeError("a shard of a position store ends before the first batch")
        return self.last
