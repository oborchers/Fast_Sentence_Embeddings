#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The blocks a vector store is read in, and ``fse.VectorBatch`` (docs/architecture/store.md,
"Reading rows")."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from numpy import ndarray

from fse._runtime import QuantizedRows


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VectorBatch:
    """One block of a vector store: its rows ``store_row`` to ``store_row + n_rows - 1``
    (positions in the store), their ``vectors`` and the ``doc_ids`` of the rows (None if the
    store has none).

    ``VectorStore.iter_batches`` gives the stored values as read-only views of the mapped
    shards: float32 or float16 rows as an array, int8 rows as ``QuantizedRows(codes,
    scales)``, whose float32 rows are ``codes.astype(np.float32) * scales[:, None]`` (the
    rows ``get_vectors`` returns). Read the fields by name: a batch is not iterable and does
    not unpack, so a field added in a later minor release (with a default) breaks no caller."""

    store_row: int
    n_rows: int
    vectors: ndarray | QuantizedRows
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


def make(tensors: dict[str, ndarray], store_row: int, first: int, end: int) -> VectorBatch:
    """The batch of the rows ``first`` to ``end - 1`` of one shard, as read-only views."""
    rows = _frozen(tensors["vectors"][first:end])
    vectors = (
        QuantizedRows(rows, _frozen(tensors["scales"][first:end])) if "scales" in tensors else rows
    )
    ids = _frozen(tensors["doc_ids"][first:end]) if "doc_ids" in tensors else None
    return VectorBatch(
        store_row=store_row + first, n_rows=end - first, vectors=vectors, doc_ids=ids
    )


def _frozen(a: ndarray) -> ndarray:
    view = a.view()
    view.flags.writeable = False
    return view
