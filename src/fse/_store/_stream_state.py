#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""What a sink of the mode ``source_position@1`` knows of its stream (docs/architecture/
store.md, "Stream positions"): the anchors of the batch it is writing, the anchor of its last
published shard, the anchors of the shards the store held and where this run starts.

The sinks of vectors and of scores share it (``ShardSinkBase`` holds one when the store is of
that mode), so that the anchor rule, ``source_position`` and the comparison of two stores
are written once.
"""

from __future__ import annotations

from ._anchors import Anchors
from ._position_format import SourcePosition, skip_of


class StreamState:
    """The stream state of one sink."""

    def __init__(
        self,
        root: SourcePosition,
        anchor: SourcePosition,
        rows: int,
        chain: tuple[tuple[int, SourcePosition], ...],
    ):
        self.anchors = Anchors()
        self.root = root  # the anchor of the start of the store
        self.tail: tuple[SourcePosition, int] = (anchor, rows)  # last published, rows then
        self.chain = chain  # (end row, anchor) of every shard the store held
        self.start = (anchor.row, rows)  # (R, rows) the run starts from, set when it attaches

    def position(self) -> tuple[str, int]:
        """``(P, skip)`` of the last published shard (else of the root)."""
        anchor, rows = self.tail
        return skip_of(anchor, rows)

    def published(self, anchor: SourcePosition, end: int) -> None:
        """A shard with the anchor ``anchor`` ending at the store row ``end`` is published."""
        self.tail = (anchor, end)

    @property
    def skip(self) -> int:
        """The records the run drops after the position of its start: the rows the store held
        when it attached, less the row of its anchor."""
        row, rows = self.start
        return rows - row
