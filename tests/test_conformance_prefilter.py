"""The conformance suite of the int8 prefilter: every variant and backend, bit for bit.

The slack ``E`` of a query and one pass over a block of stored rows (src/fse/_native_src/kernel/
_native_kernel_prefilter.h and _native_kernel_int8.h) must give the bits, statuses and rows of
``fse._reference_prefilter``: for every variant this binary has and this CPU runs, and for every
backend through its contract. Edges: tile and block remainders of the integer kernels, the
integer range, rows next to pages without access, unsafe rows and overflow, and the
floating-point environment of the caller.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from fse import _native
from fse import _reference_prefilter as prefilter
from support.conformance import BACKENDS, Reference, assert_bits_equal
from support.guarded import Guarded
from support.prefilter import int8_store
from support.prefilter_native import (
    Block,
    assert_pass_equal,
    block,
    pass_of,
    runnable_variants,
    with_inverse,
)

F32 = np.float32
DIMS = [1, 7, 15, 16, 17, 31, 33, 255, 256, 257, 1023, 1024]
VARIANTS = runnable_variants()
REFERENCE = Reference()


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dim", DIMS)
def test_bounds_are_the_reference(backend, dim):
    blk = block(1, dim, 3, seed=dim)
    got = backend.int8_bounds(blk.units, blk.qcodes, blk.qscales)
    want = REFERENCE.int8_bounds(blk.units, blk.qcodes, blk.qscales)
    assert_bits_equal(got[0], want[0], backend.name)
    np.testing.assert_array_equal(got[1], want[1])
    assert got[1].dtype == np.bool_
    assert got[1].all()


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_queries_outside_the_domain_are_the_reference(backend):
    units = np.zeros((3, 256), F32)
    units[0, :2] = [1.0, 0.5]  # ||u||^2 = 1.25, what small_norm_query normalizes to
    units[1] = 0.5
    units[2, 0] = 1.0011**0.5
    codes = np.rint(units * 100).clip(-127, 127).astype(np.int8)
    scales = np.full(3, 0.01, F32)
    got = backend.int8_bounds(units, codes, scales)
    want = REFERENCE.int8_bounds(units, codes, scales)
    assert_bits_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    assert got[1].tolist() == [False, False, True]


def test_bounds_of_float32_subnormals_and_extremes_are_the_reference():
    blk = block(1, 64, 2, seed=3)
    units = blk.units.copy()
    units[0, :4] = [1e-42, -1e-44, 0.0, 3e-39]
    units[1, 5] = np.float32(1e-45)
    got = _native.int8_bounds(units, blk.qcodes, blk.qscales)
    want = REFERENCE.int8_bounds(units, blk.qcodes, blk.qscales)
    assert_bits_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dim", DIMS)
def test_the_pass_is_the_reference(backend, dim):
    """Row counts around the groups of four, any number of queries around the tiles of two."""
    for rows in (0, 1, 2, 3, 4, 5, 6, 7, 9, 37):
        for queries in (1, 2, 3, 8):
            blk = block(rows, dim, queries, seed=rows * 31 + queries + dim)
            for k in (1, 60):
                got = backend.int8_pass(*blk.args(), k)
                assert_pass_equal(got, REFERENCE.int8_pass(*blk.args(), k), f"{rows}/{queries}/{k}")


@pytest.mark.parametrize("variant", VARIANTS)
def test_an_incoming_threshold_is_the_reference(variant):
    blk = block(300, 64, 3, seed=8)
    heap = REFERENCE.int8_pass(*blk.args(), 5).heap
    for threshold in (heap[:, -1], heap[:, 0], np.full(3, 1e9), np.full(3, -1e9)):
        args = (*blk.args(), 5)
        args = (*args[:5], np.ascontiguousarray(threshold, np.float64), 5)
        assert_pass_equal(pass_of(variant)(*args), REFERENCE.int8_pass(*args), variant)


# --- The integer range -----------------------------------------------------------------------


def _extreme_block(dim):
    pattern = np.where(np.arange(dim) % 2, 127, -128).astype(np.int8)
    rows = [np.full(dim, -128), np.full(dim, 127), pattern, -pattern - 1, np.zeros(dim)]
    hot = np.zeros(dim, np.int8)
    hot[0] = -128
    stored, inverse = int8_store(np.stack([*rows, hot]).astype(np.int8))
    qcodes = np.stack([np.full(dim, -127), np.full(dim, 127), -pattern.clip(-127)]).astype(np.int8)
    return Block(stored, inverse, qcodes, np.array([1.0, 0.5, 2.0], F32), np.full(3, 0.5), None)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dim", [1023, 1024])
def test_the_largest_integer_sums_are_the_reference(backend, dim):
    blk = _extreme_block(dim)
    for k in (1, 3):
        assert_pass_equal(backend.int8_pass(*blk.args(), k), REFERENCE.int8_pass(*blk.args(), k))


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_estimate_of_the_extreme_rows_is_their_exact_integer_sum(variant):
    """-128 against -127 at 1024 values: D = 128 * 127 * 1024, S = 2^24, inv = 2^-12."""
    stored, inverse = int8_store(np.full((1, 1024), -128, np.int8))
    qcodes = np.full((1, 1024), -127, np.int8)
    args = (stored, inverse, qcodes, np.ones(1, F32), np.full(1, 0.5), np.full(1, -np.inf), 1)
    got = pass_of(variant)(*args)
    assert got.status == "ok"
    assert got.heap[0, 0] == (128 * 127 * 1024) / 4096
    assert 128 * 127 * 1024 == 16646144
    assert float(inverse[0]) == 2.0**-12


# --- Memory next to the arrays ---------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="mprotect: POSIX only")
@pytest.mark.parametrize("at_end", [True, False], ids=["end", "start"])
@pytest.mark.parametrize("dim", [1, 15, 16, 17, 31, 33, 255, 257])
@pytest.mark.parametrize("variant", VARIANTS)
def test_no_read_beyond_an_array(variant, dim, at_end):
    """Every input array sits against a page without access: a load past the end of a row,
    of a query or of the vectors faults (the integer kernels finish a row with scalar code)."""
    blk = block(9, dim, 3, seed=dim)
    guarded = [Guarded(a.nbytes, at_end).array(a) for a in blk.args()[:6]]
    qcodes_hot = Guarded(blk.units.nbytes, at_end).array(blk.units)
    got = pass_of(variant)(*guarded, 4)
    assert_pass_equal(got, REFERENCE.int8_pass(*blk.args(), 4), variant)
    bounds = _native.int8_bounds(qcodes_hot, guarded[2], guarded[3])
    assert_bits_equal(bounds[0], blk.slack)


# --- Unsafe rows and overflow ----------------------------------------------------------------

BAD = {
    "nan": lambda _inv: np.nan,
    "inf": lambda _inv: np.inf,
    "-inf": lambda _inv: -np.inf,
    "negative": lambda inv: -inv,
    "twice": lambda inv: F32(2) * inv,
    "just outside": lambda inv: F32(1.0000007) * inv,
}


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("position", [0, 3, 4, 7, 8, 11, 12])
@pytest.mark.parametrize("queries", [1, 3])
def test_the_first_unsafe_row_is_reported_and_nothing_else(backend, bad, position, queries):
    blk = block(13, 48, queries, seed=position)
    inverse = blk.inverse.copy()
    with np.errstate(invalid="ignore"):
        inverse[position] = BAD[bad](inverse[position])
        inverse[12] = np.nan  # a later one never changes the answer
    blk = with_inverse(blk, inverse)
    got = backend.int8_pass(*blk.args(), 2)
    assert (got.status, got.row) == ("unsafe", position)
    assert_pass_equal(got, REFERENCE.int8_pass(*blk.args(), 2))


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_zero_row_is_safe_with_a_finite_norm_and_unsafe_with_an_infinite_one(variant):
    blk = block(8, 20, 2, seed=1)
    stored = blk.stored.copy()
    stored[5] = 0
    inverse = blk.inverse.copy()
    for value, status in ((0.0, "ok"), (1e30, "ok"), (np.inf, "unsafe")):
        inverse[5] = value
        args = (stored, inverse, blk.qcodes, blk.qscales, blk.slack, np.full(2, -np.inf), 3)
        got = pass_of(variant)(*args)
        assert got.status == status
        assert_pass_equal(got, REFERENCE.int8_pass(*args))


def _ties(rows, dim=16, queries=1):
    """A block of equal rows: every row stays whatever the threshold, until the list is full."""
    blk = block(1, dim, queries, seed=2)
    stored, inverse = int8_store(np.tile(blk.stored[0], (rows, 1)))
    return blk._replace(stored=stored, inverse=inverse)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("queries", [1, 2, 3])
def test_a_full_list_that_the_floor_cannot_cut_overflows(backend, queries):
    blk = _ties(40, queries=queries)
    got = backend.int8_pass(*blk.args(), 1, 8)
    assert got.status == "overflow"
    assert_pass_equal(got, REFERENCE.int8_pass(*blk.args(), 1, 8))
    ok = backend.int8_pass(*blk.args(), 1, 40)
    assert ok.status == "ok"
    assert all(len(rows) == 40 for rows in ok.rows)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_default_capacity_is_the_larger_of_four_k_and_4096(variant):
    run = pass_of(variant)
    for k, rows, status in ((1, 4096, "ok"), (1, 4097, "overflow"), (2000, 8001, "overflow")):
        blk = _ties(rows)
        got = run(*blk.args(), k)
        assert got.status == status, (k, rows)
        assert prefilter.default_capacity(k) == max(4 * k, 4096)
        assert_pass_equal(got, REFERENCE.int8_pass(*blk.args(), k), (k, rows))


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_full_list_drops_what_the_floor_cuts_before_it_overflows(variant):
    """Rows that come closer to the query one by one raise the floor: the list is cut again
    and again, never overflows, and ends as the reference's list does."""
    rng = np.random.default_rng(5)
    blk = block(1, 64, 2, seed=4)
    unit = blk.units[0].astype(np.float64)
    rows = [
        np.round(127 * (unit + 4.0 * (1 - i / 120) * rng.standard_normal(64)) / 5)
        for i in range(120)
    ]
    stored, inverse = int8_store(np.clip(rows, -128, 127).astype(np.int8))
    blk = blk._replace(stored=stored, inverse=inverse)
    run = pass_of(variant)
    raw = REFERENCE.int8_pass(*blk.args(), 1, 10**6)  # nothing is dropped
    for capacity in (2, 4, 8, 100):
        got = run(*blk.args(), 1, capacity)
        assert_pass_equal(got, REFERENCE.int8_pass(*blk.args(), 1, capacity), capacity)
        assert got.status == "ok"
    cut = run(*blk.args(), 1, 4)
    assert all(len(c) <= 4 < len(r) for c, r in zip(cut.rows, raw.rows, strict=True))


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("position", [2, 36, 39])
def test_an_unsafe_row_wins_over_an_overflow_whatever_their_order(backend, position):
    blk = _ties(40)
    inverse = blk.inverse.copy()
    inverse[position] = F32(2) * inverse[position]
    blk = with_inverse(blk, inverse)
    got = backend.int8_pass(*blk.args(), 1, 8)
    assert (got.status, got.row) == ("unsafe", position)
    assert_pass_equal(got, REFERENCE.int8_pass(*blk.args(), 1, 8))


# --- The floating-point environment of the caller --------------------------------------------

ENVS = [{"flush": True}, {"round_up": True}, {"flush": True, "round_up": True}]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("env", ENVS, ids=lambda e: "-".join(e))
def test_the_environment_of_the_caller_changes_no_bit_and_is_restored(variant, env):
    """FTZ, DAZ and rounding toward +inf are the caller's: the kernel runs in round to
    nearest, no flush, whatever was set, and puts the caller's environment back, also on
    the unsafe and overflow exits."""
    blk = block(37, 40, 3, seed=6)
    inverse = blk.inverse.copy()
    inverse[[3, 9]] = [F32(1e-40), F32(3e-45)]  # float32 subnormals that DAZ would read as 0
    blk = with_inverse(blk, inverse)
    units = blk.units.copy()
    units[1, :3] = [1e-42, 5e-45, -2e-40]
    want_bounds = REFERENCE.int8_bounds(units, blk.qcodes, blk.qscales)
    want_ok = REFERENCE.int8_pass(*blk.args(), 4)
    unsafe_inverse = blk.inverse.copy()
    unsafe_inverse[20] = np.nan
    unsafe = with_inverse(blk, unsafe_inverse)
    saved = _native._test_fpenv_set(**env)
    try:
        before = _native._test_fpenv_get()
        bounds = _native.int8_bounds(units, blk.qcodes, blk.qscales)
        got_ok = pass_of(variant)(*blk.args(), 4)
        got_unsafe = pass_of(variant)(*unsafe.args(), 4)
        got_overflow = pass_of(variant)(*_ties(40).args(), 1, 8)
        after = _native._test_fpenv_get()
    finally:
        _native._test_fpenv_restore(saved)
    assert before == after
    assert_bits_equal(bounds[0], want_bounds[0])
    np.testing.assert_array_equal(bounds[1], want_bounds[1])
    assert_pass_equal(got_ok, want_ok)
    assert got_unsafe.status == "unsafe"
    assert got_overflow.status == "overflow"
