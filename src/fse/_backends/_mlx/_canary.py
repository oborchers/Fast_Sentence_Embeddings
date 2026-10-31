#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The canary of the mlx backend: cases whose bits and flags the GPU must give, checked
against fixed patterns once per process (``_device``), not against a reference computed
in the floating-point environment of the caller.

The inputs are bit patterns, not arithmetic: numpy would compute them in the
floating-point environment of the importing thread. Word 0 is f = 1 + 2**-12, word 1 the
rounded -(f * f), word 2 is 1 + 2**-23, word 3 the largest float32, word 4 is
-(1 + 2**-23). Sentence 0 keeps the rounding error of f * f only if the fma is fused;
sentence 1 is (1 / 3) * acc, not acc / 3 (0x3F800001); sentence 2 overflows (flagged);
sentence 3 cancels exactly to +0.0.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import Batch

from .._base import PoolInput, UnsupportedByBackendError
from . import _kernels, _sources

ROWS = np.array([0x3F800800, 0xBF801000, 0x3F800001, 0x7F7FFFFF, 0xBF800001], np.uint32).view(
    np.float32
)
WEIGHTS = np.array([0x3F800800, *[0x3F800000] * 4], np.uint32).view(np.float32)
OFFSETS = np.array([0, 2, 5, 7, 9], np.int64)
TOKENS = np.array([1, 0, 2, 2, 2, 3, 3, 2, 4], np.uint32)
BITS = (0x33000000, 0x3F800002, 0x00000000)
FLAGS = (0, 0, _sources.NONFINITE, 0)
# Sentence 0 normalized: 2**-25 four times, squares 2**-48, every value 0.5.
NORMALIZED = (0x3F000000, 0x27800000)
# The smallest subnormal float16, 2**-24, which must widen to the float32 2**-24 exactly.
HALF = (0x0001, 0x33800000)


def _rows(mx: ModuleType, stream: Any, rows: ndarray, weights: ndarray, batch: Batch, **options):
    """The rows, flags and bounds of every sentence of ``batch`` from the kernel."""
    t = _kernels.table(mx, PoolInput(rows, weights), None)
    probe = _kernels.task(batch, t)
    assert probe is not None
    launched = _kernels.launch(mx, stream, t, probe, **options)
    mx.eval(*launched)
    view, flags, bounds = _kernels.view(launched)
    return np.array(view), flags, bounds


def _fail(what: str) -> UnsupportedByBackendError:
    return UnsupportedByBackendError(f"the mlx backend cannot run on this GPU: {what}")


def canary(mx: ModuleType, stream: Any) -> None:
    """Raise UnsupportedByBackendError unless this GPU fuses, rounds 1 / n first, widens
    float16 exactly, normalizes and bounds as the kernel must, and gives its flags, for
    one and for four values per thread."""
    batch = Batch(0, OFFSETS, TOKENS)
    for dim in (1, 4):
        rows = np.ascontiguousarray(np.repeat(ROWS[:, None], dim, axis=1))
        got, flags, _ = _rows(mx, stream, rows, WEIGHTS, batch)
        exact = got.view(np.uint32)[[0, 1, 3]]
        if not (exact == np.array(BITS, np.uint32)[:, None]).all():
            bits = [hex(b) for b in exact[:, 0]]
            raise _fail(f"its fma or the rounding of 1 / n differs ({bits})")
        if tuple((flags & _kernels.ALWAYS).tolist()) != FLAGS:
            raise _fail(f"its flags differ ({flags.tolist()})")
    rows = np.ascontiguousarray(np.repeat(ROWS[:, None], 4, axis=1))
    got, _, bounds = _rows(mx, stream, rows, WEIGHTS, batch, normalize=True)
    squares = bounds[:, 1].view(np.uint32)
    if (got[0].view(np.uint32) != NORMALIZED[0]).any() or squares[0] != NORMALIZED[1]:
        raise _fail("it does not normalize as the cpu backend")
    if not float(bounds[1, 0]) >= 3 * 2 * (1 + 2.0**-23):  # sum |f| ||x||, rounded up
        raise _fail(f"its bound of the magnitudes is too small ({bounds[1, 0]!r})")
    half = np.full((1, 4), HALF[0], np.uint16).view(np.float16)
    one = Batch(0, np.array([0, 1], np.int64), np.zeros(1, np.uint32))
    got, flags, _ = _rows(mx, stream, half, np.ones(1, np.float32), one)
    if flags.any() or (got.view(np.uint32) != HALF[1]).any():
        raise _fail("it does not widen float16 exactly")
