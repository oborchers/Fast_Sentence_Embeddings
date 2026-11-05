#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The blocks a score store is read in, and ``fse.ScoreBatch`` (docs/architecture/scores.md,
"Reading")."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from numpy import ndarray


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ScoreBatch:
    """One block of a score store: its rows ``store_row`` to ``store_row + n_rows - 1`` of
    the store, the ``outputs`` of every head by name and the ``doc_ids`` of the rows (None
    if the store has none).

    ``ScoreStore.iter_batches`` gives the raw stored values as read-only views of the mapped
    shards (a head of several columns is not contiguous); ``predict_batches`` gives labels or
    values as ``Heads`` does. Read the fields by name: a batch is not iterable and does not
    unpack, so a field added in a later minor release (with a default) breaks no caller."""

    store_row: int
    n_rows: int
    outputs: Mapping[str, ndarray]
    doc_ids: ndarray | None = None


def blocks(
    shards: list[tuple[Any, dict[str, ndarray]]], start: int, stop: int, block_rows: int
) -> Iterator[tuple[Any, dict[str, ndarray], int, int]]:
    """The blocks of the rows ``start`` to ``stop - 1``: (shard, tensors, first, end) with
    rows of the shard from ``first`` to ``end - 1``, at most ``block_rows`` each and never
    over the end of a shard."""
    for shard, tensors in shards:
        low = max(start, shard.first_row) - shard.first_row
        high = min(stop, shard.first_row + shard.rows) - shard.first_row
        for first in range(low, high, block_rows):
            yield shard, tensors, first, min(first + block_rows, high)
