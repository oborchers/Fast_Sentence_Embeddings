#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The rows of a store at a list of positions, in the order asked (docs/architecture/store.md,
"Reading rows").

The positions are checked as a whole before a row is read. The reads then run once per shard
and block on the positions sorted, so every shard is visited in file order, and the rows are
put back where they were asked. Duplicates stay, and the bits of a row do not depend on the
other rows asked with it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._ids import checked_ids

# The bytes one block of rows may take in memory besides the output.
_BLOCK_BYTES = 16 << 20
_ITEMSIZE = {"float32": 4, "float16": 2, "int8": 1}
_HINT = "rows must be a list of integers or an int64 array of one or two dimensions"


def positions(value: Any) -> ndarray:
    """The int64 positions of ``value``, in its shape: an array of one or two dimensions is
    used as it is (no copy), a list is converted. Anything else raises."""
    if isinstance(value, np.ndarray):
        if (
            isinstance(value, np.ma.MaskedArray)
            or value.dtype != np.dtype("int64")
            or value.ndim not in (1, 2)
        ):
            kind = "a masked array" if isinstance(value, np.ma.MaskedArray) else "an array"
            raise TypeError(
                f"{_HINT}, not {kind} of dtype {value.dtype} and shape {value.shape}; "
                "use np.asarray(rows, dtype=np.int64) and reshape it to one or two dimensions"
            )
        return np.asarray(value)  # a subclass such as np.matrix would break ravel
    if not isinstance(value, list):
        raise TypeError(f"{_HINT}, not {type(value).__name__}")
    checked_ids(value, "rows")
    return np.array(value, dtype=np.int64)


def vector_row_bytes(dim: int, dtype: str) -> int:
    """The bytes the blocks hold per row while vectors are decoded: the index, the stored
    row, the scale of an int8 row and the float32 copy."""
    return 8 + dim * _ITEMSIZE[dtype] + (4 if dtype == "int8" else 0) + 4 * dim


ID_ROW_BYTES = 16


def read_vectors(tensors: dict[str, ndarray], local: ndarray) -> ndarray:
    """The float32 rows at the positions ``local`` of one shard, decoded as ``get_vectors``."""
    scales = tensors["scales"][local] if "scales" in tensors else None
    return _codes.dequantize(tensors["vectors"][local], scales)


def read_doc_ids(tensors: dict[str, ndarray], local: ndarray) -> ndarray:
    """The document IDs at the positions ``local`` of one shard."""
    return tensors["doc_ids"][local]


def gather(
    shards: list[tuple[Any, dict[str, ndarray]]],
    wanted: ndarray,
    total: int,
    read: Any,
    *,
    tail: tuple[int, ...],
    dtype: Any,
    row_bytes: int,
) -> ndarray:
    """``read(tensors, local)`` for the positions ``wanted`` (of a store of ``total`` rows),
    as a new array of shape ``wanted.shape + tail``, in the order of ``wanted``.

    Raises ValueError for a position outside the store, before a row is read.
    """
    flat = wanted.ravel(order="C")
    if flat.size == 0:
        return np.empty(wanted.shape + tail, dtype)
    if total == 0:
        raise ValueError("rows must be positions of the store, but the store has no rows")
    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    if ordered[0] < 0 or ordered[-1] >= total:
        raise ValueError(_outside(flat, wanted.shape, total))
    firsts = [shard.first_row for shard, _ in shards]
    bounds = np.searchsorted(ordered, [*firsts, total])
    out = np.empty((flat.size, *tail), dtype)
    block = max(1, _BLOCK_BYTES // row_bytes)
    for (shard, tensors), begin, end in zip(shards, bounds[:-1], bounds[1:], strict=True):
        for low in range(int(begin), int(end), block):
            high = min(low + block, int(end))
            out[order[low:high]] = read(tensors, ordered[low:high] - shard.first_row)
    return out.reshape(wanted.shape + tail)


def gather_sorted(
    shards: list[tuple[Any, dict[str, ndarray]]],
    wanted: ndarray,
    total: int,
    name: str,
) -> ndarray:
    """Read column ``name`` at ascending unique 1-D positions into one owned array.

    The caller owns and orders the positions. Validate their range before accessing any
    tensor; unlike general ``gather``, neither sorting nor an output permutation is needed.
    """
    if wanted.size:
        if total == 0:
            raise ValueError("rows must be positions of the store, but the store has no rows")
        if wanted[0] < 0 or wanted[-1] >= total:
            raise ValueError(_outside(wanted, wanted.shape, total))
    column = shards[0][1][name]
    out = np.empty((len(wanted), *column.shape[1:]), column.dtype)
    firsts = [shard.first_row for shard, _ in shards]
    bounds = np.searchsorted(wanted, [*firsts, total])
    row_bytes = max(1, column.dtype.itemsize * int(np.prod(column.shape[1:]))) + 8
    block = max(1, _BLOCK_BYTES // row_bytes)
    for (shard, tensors), begin, end in zip(shards, bounds[:-1], bounds[1:], strict=True):
        for low in range(int(begin), int(end), block):
            high = min(low + block, int(end))
            source = tensors[name]
            assert source.dtype == out.dtype  # a dtype conversion could silently buffer
            # The whole range was checked above. No index clips; raise would buffer out.
            np.take(
                source,
                wanted[low:high] - shard.first_row,
                axis=0,
                out=out[low:high],
                mode="clip",
            )
    return out


def _outside(flat: ndarray, shape: tuple[int, ...], total: int) -> str:
    """The message for the first position of ``flat`` outside the store."""
    at = int(np.argmax((flat < 0) | (flat >= total)))
    index = ", ".join(str(i) for i in np.unravel_index(at, shape))
    return f"rows[{index}] is {int(flat[at])}; rows must be positions from 0 to {total - 1}"
