#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers

"""Storage formats of the sentence vectors: float32, float16 and int8.

Sentence vectors are always computed in float32 and only stored in a smaller format.
Reading them gives float32 again. float16 is a plain cast. int8 is symmetric with one
float32 scale per row: ``codes = rint(row / scale)`` with ``scale = absmax(row) / 127``,
so ``codes * scale`` restores the row. As the scale belongs to the row, the cosine of two
rows of codes equals the cosine of the restored rows.

An STS measurement over the 15 vector sets of the README with CBOW, SIF-10 and uSIF found
float16 lossless and int8 within 0.02 points of float32.
"""

from typing import Literal

import numpy
from numpy import ndarray

from fse.models.utils import REMOVAL_BLOCK_ROWS

StorageDtype = Literal["float32", "float16", "int8"]

# The numpy type of each storage format.
STORAGE_TYPES = {"float32": numpy.float32, "float16": numpy.float16, "int8": numpy.int8}

_INT8_TOP = 127
_FLOAT16_MAX = float(numpy.finfo(numpy.float16).max)


def has_scales(dtype: StorageDtype) -> bool:
    """Whether the format stores a float32 scale per row."""
    return dtype == "int8"


def quantize_into(rows: ndarray, codes: ndarray, scales: ndarray | None, first_row: int = 0):
    """Store float32 ``rows`` into ``codes`` (and ``scales``), in blocks of rows.

    ``codes`` and ``scales`` may be memory-mapped; they are written block by block, so
    only one block of temporaries exists. ``first_row`` is the row number of ``rows[0]``
    in error messages. Raises a ValueError for rows that are not finite or, for float16,
    out of its range, instead of storing infinity.
    """
    if codes.shape != rows.shape:
        raise ValueError(f"codes of shape {codes.shape} cannot hold rows of shape {rows.shape}")
    for begin in range(0, len(rows), REMOVAL_BLOCK_ROWS):
        block = numpy.asarray(rows[begin : begin + REMOVAL_BLOCK_ROWS], dtype=numpy.float32)
        end = begin + len(block)
        _check_finite(block, first_row + begin)
        if codes.dtype == numpy.int8:
            absmax = numpy.abs(block).max(axis=1) if block.shape[1] else numpy.zeros(len(block))
            scale = (absmax / _INT8_TOP).astype(numpy.float32)
            # A zero row has scale 0 and codes 0; dividing by 1 instead keeps it zero.
            divisor = numpy.where(scale > 0, scale, numpy.float32(1))[:, None]
            codes[begin:end] = numpy.clip(numpy.rint(block / divisor), -_INT8_TOP, _INT8_TOP)
            cast_scales(scales)[begin:end] = scale
        else:
            if codes.dtype == numpy.float16 and block.size:
                large = numpy.flatnonzero((numpy.abs(block) > _FLOAT16_MAX).any(axis=1))
                if len(large):
                    raise ValueError(
                        f"sentence vector {first_row + begin + int(large[0])} has values "
                        f"beyond the float16 range of +-{_FLOAT16_MAX:g}; use "
                        "sv_dtype='float32' or 'int8'"
                    )
            codes[begin:end] = block


def dequantize(codes: ndarray, scales: ndarray | None) -> ndarray:
    """The float32 rows stored in ``codes`` (and ``scales``), as a new array.

    Always a copy, also for float32 codes: a view would keep a memmap open and let
    writes reach the stored vectors.
    """
    rows = numpy.array(codes, dtype=numpy.float32, copy=True)
    if scales is not None:
        rows *= numpy.asarray(scales, dtype=numpy.float32)[:, None]
    return rows


def cast_scales(scales: ndarray | None) -> ndarray:
    if scales is None:
        raise ValueError("int8 sentence vectors need an array of scales")
    return scales


def _check_finite(block: ndarray, first_row: int):
    bad = numpy.flatnonzero(~numpy.isfinite(block).all(axis=1))
    if len(bad):
        raise ValueError(
            f"sentence vector {first_row + int(bad[0])} contains NaN or infinite values, "
            "which cannot be stored"
        )
