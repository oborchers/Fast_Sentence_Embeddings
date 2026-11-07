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

import numpy as np
from numpy import ndarray

from fse._runtime import QuantizedRows


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class VectorBatch:
    """One block of a vector store: its rows ``store_row`` to ``store_row + n_rows - 1``
    (positions in the store), their ``vectors``, the ``doc_ids`` of the rows (None if the
    store has none) and their ``inv_norms`` (None for a batch built without them).

    ``VectorStore.iter_batches`` gives the stored values as read-only views of the mapped
    shards: float32 or float16 rows as an array, int8 rows as ``QuantizedRows(codes,
    scales)``, whose float32 rows are ``codes.astype(np.float32) * scales[:, None]`` (the
    rows ``get_vectors`` returns); ``inv_norms`` is the read-only float32 view of the stored
    inverse norms of the rows, which stays readable after ``close``. A batch is a new type of
    2.1 and its pickles are not promised across versions. Read the fields by name: a batch is not iterable and does
    not unpack, so a field added in a later minor release (with a default) breaks no caller."""

    store_row: int
    n_rows: int
    vectors: ndarray | QuantizedRows
    doc_ids: ndarray | None = None
    inv_norms: ndarray | None = None

    def normalized(self) -> ndarray:
        """The rows as a new, writable, C-contiguous float32 array ``(n_rows, dim)`` with
        every element ``float32(stored) * inv_norm``, for an index that takes float32 rows
        (FAISS, hnswlib). An int8 row uses its codes without the scale, which cancels out of
        the inverse norm. In the default IEEE environment (round to nearest even, subnormals
        kept) the bits do not depend on the block; under another rounding mode or flush to
        zero the result is only approximate. The rows are approximately unit length, because
        the stored inverse norm is already rounded to float32; a zero row stays zero. This is
        not the ``normalize@1`` of the numerical contract. Raises ``ValueError`` or
        ``TypeError`` for a batch that has no ``inv_norms`` or whose fields do not fit."""
        rows = _stored_rows(self.vectors, self.n_rows)
        inverse = _checked_inv_norms(self.inv_norms, self.n_rows)
        out = np.empty(rows.shape, dtype=np.float32)
        np.multiply(rows, inverse[:, None], out=out, dtype=np.float32)
        return out


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
        store_row=store_row + first,
        n_rows=end - first,
        vectors=vectors,
        doc_ids=ids,
        inv_norms=_frozen(tensors["inv_norms"][first:end]),
    )


def _stored_rows(vectors: Any, n_rows: int) -> ndarray:
    """The 2-D array ``normalized`` multiplies: the rows, or the int8 codes."""
    if isinstance(vectors, QuantizedRows):
        rows, allowed = vectors.codes, (np.dtype(np.int8),)
    else:
        rows, allowed = vectors, (np.dtype(np.float32), np.dtype(np.float16))
    if not isinstance(rows, ndarray) or rows.dtype not in allowed:
        raise TypeError(
            "vectors must be a float32 or float16 array or QuantizedRows of int8 codes, "
            f"not {getattr(rows, 'dtype', type(rows).__name__)}"
        )
    if rows.ndim != 2 or rows.shape[0] != n_rows:
        raise ValueError(f"vectors must have the shape ({n_rows}, dim), not {rows.shape}")
    return rows


def _checked_inv_norms(inv_norms: Any, n_rows: int) -> ndarray:
    if inv_norms is None:
        raise ValueError("the batch has no inv_norms; normalized() needs them")
    if not isinstance(inv_norms, ndarray) or inv_norms.dtype != np.float32:
        raise TypeError(
            "inv_norms must be a float32 array, "
            f"not {getattr(inv_norms, 'dtype', type(inv_norms).__name__)}"
        )
    if inv_norms.shape != (n_rows,):
        raise ValueError(f"inv_norms must have the shape ({n_rows},), not {inv_norms.shape}")
    if not (np.isfinite(inv_norms).all() and (inv_norms >= 0).all()):
        raise ValueError("inv_norms must be finite and not negative")
    return inv_norms


def _frozen(a: ndarray) -> ndarray:
    view = a.view()
    view.flags.writeable = False
    return view
