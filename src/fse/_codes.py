#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Storage formats of 2.0 rows (stage 5): float32, float16 and int8 with a scale per row.

The arithmetic is the one of ARCHITECTURE.md, "Numerical contract", and equals
``fse._reference.quantize_rows`` and ``dequantize`` bit for bit; here it runs in blocks
of rows, so only one block of temporaries exists, also for memory-mapped targets.
Only numpy is imported, so the module can be copied into another project.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

BLOCK_ROWS = 16_384

STORAGE_TYPES = {"float32": np.float32, "float16": np.float16, "int8": np.int8}

_F32 = np.float32
_INT8_TOP = _F32(127)
_FLOAT16_MAX = float(np.finfo(np.float16).max)


def check_dtype(dtype: object) -> str:
    """Return ``dtype`` if it names a storage format, or raise a ValueError."""
    if not isinstance(dtype, str) or dtype not in STORAGE_TYPES:
        raise ValueError(f"dtype must be one of {sorted(STORAGE_TYPES)}, not {dtype!r}")
    return dtype


def quantize_into(
    rows: ndarray,
    codes: ndarray,
    scales: ndarray | None,
    *,
    first_row: int = 0,
    kind: str = "row",
    block_rows: int = BLOCK_ROWS,
):
    """Store float32 ``rows`` into ``codes`` (and ``scales`` for int8), in blocks of rows.

    ``codes`` decides the format. Rows that are not finite, and float16 values beyond
    +-65504, raise a ValueError that names ``kind`` and the row ``first_row + i``.
    """
    if codes.shape != rows.shape:
        raise ValueError(f"codes of shape {codes.shape} cannot hold rows of shape {rows.shape}")
    int8 = codes.dtype == np.int8
    if int8 and (scales is None or scales.shape != (len(rows),)):
        raise ValueError(f"int8 codes need {len(rows)} scales")
    if not int8 and scales is not None:
        raise ValueError(f"{codes.dtype} codes have no scales")
    for begin in range(0, len(rows), block_rows):
        block = np.asarray(rows[begin : begin + block_rows], dtype=_F32)
        end = begin + len(block)
        bad = np.flatnonzero(~np.isfinite(block).all(axis=1))
        if len(bad):
            raise ValueError(
                f"{kind} {first_row + begin + int(bad[0])} contains NaN or infinite values"
            )
        if int8:
            assert scales is not None
            scale, divisor = _int8_scales(block)
            codes[begin:end] = np.clip(np.rint(block / divisor), -127, 127)
            scales[begin:end] = scale
        else:
            if codes.dtype == np.float16:
                _check_float16(block, kind, first_row + begin)
            codes[begin:end] = block


def _int8_scales(block: ndarray) -> tuple[ndarray, ndarray]:
    """The scale of every row of ``block`` and the column of divisors its codes take."""
    absmax = np.abs(block).max(axis=1) if block.shape[1] else np.zeros(len(block), dtype=_F32)
    scale = (absmax / _INT8_TOP).astype(_F32)
    # Rounding can lift 127 * scale past the float32 maximum; step the scale down.
    with np.errstate(over="ignore"):
        beyond = ~np.isfinite(scale * _INT8_TOP)
    scale[beyond] = np.nextafter(scale[beyond], _F32(0))
    # A scale of 0 (a zero row, or one whose absmax / 127 underflows) gives codes 0.
    return scale, np.where(scale > 0, scale, _F32(1))[:, None]


def _check_float16(block: ndarray, kind: str, first: int):
    """Raise for the first row of ``block`` (row ``first`` + i) beyond the float16 range."""
    if not block.size:
        return
    large = np.flatnonzero((np.abs(block) > _FLOAT16_MAX).any(axis=1))
    if len(large):
        raise ValueError(
            f"{kind} {first + int(large[0])} has values beyond the "
            f"float16 range of +-{_FLOAT16_MAX:g}"
        )


def quantize(rows: ndarray, dtype: str, *, kind: str = "row") -> tuple[ndarray, ndarray | None]:
    """New codes (and scales for int8) of float32 ``rows`` in the storage format ``dtype``."""
    codes = np.empty(rows.shape, dtype=STORAGE_TYPES[check_dtype(dtype)])
    scales = np.empty(len(rows), dtype=_F32) if dtype == "int8" else None
    quantize_into(rows, codes, scales, kind=kind)
    return codes, scales


def dequantize(codes: ndarray, scales: ndarray | None) -> ndarray:
    """The float32 rows of ``codes``: ``code * scale`` with one rounding for int8.

    Always a new array, also for float32 codes, so it never keeps a map open.
    """
    rows = np.array(codes, dtype=_F32, copy=True)
    if scales is not None:
        rows *= np.asarray(scales, dtype=_F32)[:, None]
    return rows
