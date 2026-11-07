#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The checks of the arguments of the backend methods (docs/architecture/backends.md,
"Backends"), the same for every backend, and the primitive checks they and the checks of
``_base`` are made of."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import Link

DTYPES = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float16): "float16",
    np.dtype(np.int8): "int8",
}


def overlaps(a: Any, b: Any) -> bool:
    """``overlaps(a, b)`` (the bounds test, empty arrays share nothing) without the
    numpy call, which releases the GIL and takes it back: a task of a head ran it up to 20
    times (docs/architecture/runtime.md, "Runtime", Tasks)."""
    if a.size == 0 or b.size == 0:
        return False
    low_a, high_a = _extent(a)
    low_b, high_b = _extent(b)
    return low_a < high_b and low_b < high_a


def _extent(a: Any) -> tuple[int, int]:
    """The first and one past the last byte of a non-empty array (strides and offset)."""
    interface = a.__array_interface__
    low = high = interface["data"][0]
    strides = interface["strides"]
    if strides is None:  # C-contiguous
        return low, low + a.nbytes
    for length, stride in zip(interface["shape"], strides, strict=True):
        if stride < 0:
            low += (length - 1) * stride
        else:
            high += (length - 1) * stride
    return low, high + a.itemsize


def check_score_input(stored: object, scales: object, weights: object, bias: object) -> None:
    """The arguments of ``score``, checked the same way by every backend: stored rows of a
    storage dtype, scales exactly for int8, one weight per dimension (any float dtype;
    each backend converts it to float32 as the reference does)."""
    stored = rows_of(stored, "stored")
    if np.asarray(weights).dtype.kind not in "fiu":
        raise TypeError("weights must be numbers (float or int), not strings, objects or bools")
    if np.shape(weights) != (stored.shape[1],):
        raise ValueError(f"weights must hold {stored.shape[1]} values")
    int8 = stored.dtype == np.int8
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 rows")
    if int8:
        float32_vector(scales, "scales", len(stored))
    real(bias, "bias")


def check_stored_input(stored: object, first_row: object = 0) -> None:
    """The arguments of ``inverse_norms``: a C-contiguous matrix of stored rows and the
    row its first row is."""
    rows_of(stored, "stored")
    if isinstance(first_row, bool) or not isinstance(first_row, int | np.integer) or first_row < 0:
        raise ValueError(f"first_row must be a non-negative int, not {first_row!r}")


def check_search_input(stored: object, inverse: object, queries: object) -> None:
    """The arguments of ``cosines``, checked the same way by every backend: stored rows,
    one float32 inverse norm per row, a matrix of queries of the stored dimension."""
    stored = rows_of(stored, "stored")
    float32_vector(inverse, "inverse", len(stored))
    vectors_of(queries, "queries", stored.shape[1])


def check_scores_input(stored: object, scales: object, weights: object, bias: object) -> None:
    """The arguments of ``scores``: as for ``score``, with a matrix of weight vectors and
    one bias for all or one per vector."""
    stored = rows_of(stored, "stored")
    weights = vectors_of(weights, "weights", stored.shape[1])
    int8 = stored.dtype == np.int8
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 rows")
    if int8:
        float32_vector(scales, "scales", len(stored))
    if np.ndim(bias) == 0:
        real(bias, "bias")
    elif np.asarray(bias).dtype.kind not in "fiu" or np.shape(bias) != (len(weights),):
        raise ValueError(f"bias must be one number or {len(weights)} numbers, one per vector")


def vectors_of(value: object, name: str, dim: int) -> ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be numbers (float or int), not strings, objects or bools")
    if value.ndim != 2 or value.shape[1] != dim:
        raise ValueError(
            f"{name} must be a matrix of vectors of {dim} values, not shape {value.shape}"
        )
    return value


def check_store_input(rows: object, codes: object, scales: object, first_row: object) -> None:
    """The arguments of ``store``, checked the same way by every backend: C-contiguous
    float32 rows, writable C-contiguous float16 or int8 codes of their shape, a writable
    float32 scale per row exactly for int8, none sharing memory with another."""
    if not isinstance(rows, ndarray) or rows.dtype != np.float32 or rows.ndim != 2:
        raise ValueError("rows must be a float32 matrix")
    if not rows.flags.c_contiguous:
        raise ValueError("rows must be C-contiguous")
    if (
        not isinstance(codes, ndarray)
        or codes.dtype not in (np.float16, np.int8)
        or codes.shape != rows.shape
        or not codes.flags.c_contiguous
        or not codes.flags.writeable
    ):
        raise ValueError(
            f"codes must be a writable C-contiguous float16 or int8 array of shape {rows.shape}"
        )
    _check_store_scales(scales, codes.dtype == np.int8, len(rows))
    arrays = [a for a in (rows, codes, scales) if a is not None]
    if any(overlaps(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :]):
        raise ValueError("rows, codes and scales must not share memory")
    if isinstance(first_row, bool) or not isinstance(first_row, int | np.integer) or first_row < 0:
        raise ValueError(f"first_row must be an int of at least 0, not {first_row!r}")


def _check_store_scales(scales: object, int8: bool, length: int):
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 codes")
    if int8:
        float32_vector(scales, "scales", length)
        assert isinstance(scales, ndarray)
        if not scales.flags.writeable:
            raise ValueError("scales must be writable")


def int64_vector(value: object, name: str, length: int):
    if (
        not isinstance(value, ndarray)
        or value.dtype != np.int64
        or value.shape != (length,)
        or not value.flags.c_contiguous
        or not value.flags.writeable
    ):
        raise ValueError(f"{name} must be a writable C-contiguous int64 vector of {length} values")


def check_link(link: object, k: int):
    """``link`` must be a Link that takes k logits: sigmoid one, ovr two or more, softmax
    any (of one logit z, softmax of [-z, z])."""
    if not isinstance(link, Link):
        raise ValueError(f"link must be one of {[str(member) for member in Link]}")
    if (link is Link.SIGMOID and k != 1) or (link is Link.OVR and k == 1):
        want = "one logit" if link is Link.SIGMOID else "at least two logits"
        raise ValueError(f"{link} takes {want} per row")


def float32_matrix(
    value: object, name: str, rows: int | None, cols: int | None, writable: bool = False
):
    if (
        not isinstance(value, ndarray)
        or value.dtype != np.float32
        or value.ndim != 2
        or not value.flags.c_contiguous
        or (writable and not value.flags.writeable)
        or (rows is not None and value.shape[0] != rows)
        or (cols is not None and value.shape[1] != cols)
    ):
        shape = (rows if rows is not None else "n", cols if cols is not None else "k")
        kind = "a writable" if writable else "a"
        raise ValueError(f"{name} must be {kind} C-contiguous float32 matrix of shape {shape}")


def rows_of(value: object, name: str) -> ndarray:
    if not isinstance(value, ndarray) or value.ndim != 2 or value.dtype not in DTYPES:
        raise ValueError(f"{name} must be a matrix of float32, float16 or int8 rows")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def float32_vector(value: object, name: str, length: int):
    if not isinstance(value, ndarray) or value.dtype != np.float32 or value.shape != (length,):
        raise ValueError(f"{name} must be a float32 vector of {length} values")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def real(value: object, name: str):
    """A real number, Python or numpy; arrays, strings and bools are refused."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")


def check_pooled_input(rows: object, steps: object, first_row: object) -> None:
    """The arguments of ``postprocess_pooled``, checked the same way by every backend:
    writable C-contiguous float32 rows (already pooled, so finite), the steps of a
    ``Prepared`` and the row the first row is."""
    float32_matrix(rows, "rows", None, None, writable=True)
    if not isinstance(steps, tuple):
        raise ValueError("steps must be a tuple of steps")
    if isinstance(first_row, bool) or not isinstance(first_row, int | np.integer) or first_row < 0:
        raise ValueError(f"first_row must be a non-negative int, not {first_row!r}")
