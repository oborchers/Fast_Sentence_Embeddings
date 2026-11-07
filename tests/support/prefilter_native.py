"""The comparison and the blocks of the conformance tests of the native int8 prefilter.

The kernels (every variant built and runnable here) and every backend must give, for a block of
stored rows, the status, the first unsafe row, the heaps and the kept rows and bounds of
``fse._reference_prefilter``, bit for bit.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from fse import _native
from fse import _reference_prefilter as prefilter
from support.conformance import assert_bits_equal
from support.prefilter import int8_store, prepared
from support.prefilter_witnesses import int8_rows

F32 = np.float32


def runnable_variants():
    """The variants of the kernel built into this binary that this CPU runs."""
    return [v for v in _native.prefilter_variants() if _native.prefilter_can_run(v)]


class Block(NamedTuple):
    """The arguments of ``int8_pass`` for one block, and the float32 queries of ``bounds``."""

    stored: np.ndarray
    inverse: np.ndarray
    qcodes: np.ndarray
    qscales: np.ndarray
    slack: np.ndarray
    units: np.ndarray

    def args(self, threshold=-np.inf):
        threshold = np.full(len(self.qcodes), threshold)
        return (self.stored, self.inverse, self.qcodes, self.qscales, self.slack, threshold)


def block(rows, dim, queries, seed=0):
    """Gaussian int8 rows (``rows`` may be 0) and ``queries`` Gaussian queries."""
    rng = np.random.default_rng(seed)
    codes = int8_rows(rng.standard_normal((rows, dim))) if rows else np.zeros((0, dim), np.int8)
    stored, inverse = int8_store(codes)
    units, qcodes, qscales, slack, _ = prepared(rng.standard_normal((queries, dim)).astype(F32))
    return Block(stored, inverse, qcodes, qscales, slack, units)


def with_inverse(blk, inverse):
    return blk._replace(inverse=np.ascontiguousarray(inverse, dtype=F32))


def pass_of(variant):
    """``int8_pass`` of the kernel (one variant) as a function of the reference's signature."""

    def run(stored, inverse, qcodes, qscales, slack, threshold, k, capacity=None):
        got = _native.int8_pass(
            stored, inverse, qcodes, qscales, slack, threshold, k, capacity, variant=variant
        )
        return prefilter.Pass(*got)

    return run


def assert_pass_equal(got, want, name=""):
    """Status, first unsafe row, heaps, kept rows and upper bounds are those of ``want``."""
    assert got.status == want.status, (name, got.status, want.status)
    assert got.row == want.row, (name, got.row, want.row)
    if want.status != "ok":
        assert got.heap is None
        assert got.rows is None
        assert got.upper is None
        return
    assert_bits_equal(got.heap, want.heap, name)
    assert len(got.rows) == len(want.rows)
    for j, (rows, upper) in enumerate(zip(got.rows, got.upper, strict=True)):
        assert rows.dtype == np.int64
        np.testing.assert_array_equal(rows, want.rows[j], err_msg=f"{name} query {j}")
        assert_bits_equal(upper, want.upper[j], f"{name} query {j}")
