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

from ._position_format import SourcePosition, check_position


class Anchors:
    """The anchor of the current batch and of the batch before it."""

    def __init__(self) -> None:
        self.previous: SourcePosition | None = None
        self.current: SourcePosition | None = None

    def begin(self, position: str, row: int) -> None:
        """A non-empty batch whose first record is the store row ``row`` and whose position
        is ``position`` starts."""
        self.previous, self.current = self.current, SourcePosition(check_position(position), row)

    def take(self, end: int) -> SourcePosition:
        """The anchor of a shard that ends at the store row ``end`` (exclusive)."""
        current = self.current
        if current is not None and end - 1 >= current.row:
            return current
        if self.previous is None:
            raise RuntimeError("a shard of a position store ends before the first batch")
        return self.previous
