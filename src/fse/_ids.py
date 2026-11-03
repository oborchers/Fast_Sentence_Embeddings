"""Strict signed-int64 input, shared by document writers and search filters."""

from __future__ import annotations

from typing import Any

import numpy as np


def checked_ids(value: Any, name: str = "doc_ids") -> Any:
    """Keep arrays and lists in place; never coerce values or copy an entire column."""
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype("int64") or value.ndim != 1:
            raise TypeError(f"{name} must be a one-dimensional int64 array")
        return value
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of integers or a one-dimensional int64 array")
    for row, item in enumerate(value):
        if type(item) not in (int, np.int64):
            raise TypeError(f"{name} row {row} must be an int64 integer, not {type(item).__name__}")
        if not -(2**63) <= item < 2**63:
            raise ValueError(f"{name} row {row} is outside the signed int64 range")
    return value


def id_block(value: Any, begin: int, end: int) -> np.ndarray:
    """Own the bytes handed to asynchronous writing and hashing threads."""
    if not 0 <= begin <= end <= len(value):
        raise ValueError(f"doc_ids has {len(value)} rows, but rows {begin} to {end} are required")
    block = np.array(value[begin:end], dtype="<i8", order="C", copy=True)
    block.setflags(write=False)
    return block
