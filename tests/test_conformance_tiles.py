"""The conformance suite of the tiled weighted mean: block and tile edges, bit for bit.

The register-blocked loop (src/fse/_native_kernel_tiles.h) keeps the accumulators of a
block of the dimension in registers over a tile of terms. The cases of
tests/support/tiles.py put witnesses into every block, sub-block and the tail and across
the tile boundaries of every K up to 64; every CPU kernel must give the bits of
fse._reference on them, also when out holds garbage before, when a late row overflows,
and when the last row of a table ends at a page that cannot be read.
"""

from __future__ import annotations

import ctypes
import functools
import mmap
import sys

import numpy as np
import pytest

from fse import _native, _reference
from fse._contract import OOV_TOKEN
from support.conformance import BACKENDS, BIG, F32, Case, Contract, Native, _rng, assert_bits_equal
from support.tiles import dirty, model_pool, tile_cases

CASES = tile_cases()
IDS = [f"{c.dtype}-{c.name}" for c in CASES]
CPU = [
    b
    for b in BACKENDS
    if isinstance(b, Native) or (isinstance(b, Contract) and b.backend.name == "cpu")
]
ISAS = _native.available_isas()


@functools.cache
def expected(index: int, kind: str):
    case = CASES[index]
    if kind == "pool":
        return _reference.pool_weighted_mean(case.batch, *case.args())
    return _reference.accumulate(case.batch, *case.args())


@pytest.mark.parametrize("backend", CPU, ids=lambda b: b.name)
@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_pool(backend, index):
    assert_bits_equal(backend.pool(CASES[index]), expected(index, "pool"), CASES[index].name)


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_accumulate(isa, index):
    accs, counts = Native(isa).accumulate(CASES[index])
    want_accs, want_counts = expected(index, "accumulate")
    assert_bits_equal(accs, want_accs, CASES[index].name)
    np.testing.assert_array_equal(counts, want_counts)


def _kernel(case, isa, **arrays):
    rows, weights, scales, ngram_rows, ngram_scales, oov_weight = case.args()
    rows = arrays.get("rows", rows)
    ngram_rows = arrays.get("ngram_rows", ngram_rows)
    return _native.WeightedMean(
        rows, weights, scales, ngram_rows, ngram_scales, oov_weight, isa=isa
    )


@pytest.mark.parametrize("isa", ISAS)
@pytest.mark.parametrize("index", range(len(CASES)), ids=IDS)
def test_out_is_written_not_read(isa, index):
    """out holds NaN payloads and -0.0 before: the first tile starts from +0.0."""
    case = CASES[index]
    out = dirty((len(case.sentences), case.base.shape[1]))
    _kernel(case, isa).pool(case.batch, out)
    assert_bits_equal(out, expected(index, "pool"), case.name)


# --- A late overflow -----------------------------------------------------------------------


def _overflow_case(column):
    """Sentence 1 overflows in ``column`` only, at terms 80 and 90 (a late tile for every
    K below 80); sentences 0, 2 and 3 are finite."""
    rng = _rng(3)
    dim = 301
    base = np.vstack([rng.standard_normal((10, dim)).astype(F32), np.zeros((1, dim), F32)])
    base[10, column] = BIG
    long = [int(t) for t in rng.integers(0, 10, 100)]
    long[80] = long[90] = 10
    sentences = ((1, 2, 3, 4, 5), tuple(long), (6, 7, 8), (9, 0))
    return Case(f"overflow {column}", "float32", base, np.ones(11, F32), sentences, first_row=5)


@pytest.mark.parametrize("column", [0, 290, 300])
@pytest.mark.parametrize("isa", ISAS)
def test_a_late_overflow_writes_its_row_and_stops(isa, column):
    case = _overflow_case(column)
    kernel = _kernel(case, isa)
    out = dirty((4, 301))
    with pytest.raises(ValueError, match=r"^row 6 contains NaN or infinite values$"):
        kernel.pool(case.batch, out)
    first = Case(case.name, case.dtype, case.base, case.weights, case.sentences[:1])
    assert_bits_equal(out[:1], _reference.pool_weighted_mean(first.batch, *first.args()))
    faulty = Case(case.name, case.dtype, case.base, case.weights, case.sentences[1:2])
    want = model_pool(faulty, 1)  # the fused chain and the scaling of the whole row
    assert np.isinf(want[0, column])
    assert np.isfinite(np.delete(want[0], column)).all()
    assert_bits_equal(out[1:2], want, "the faulty row, finished whole")
    assert_bits_equal(out[2:], dirty((4, 301))[2:], "the rows after it are untouched")
    with pytest.raises(ValueError, match=r"^row 6 contains NaN or infinite values$"):
        kernel.accumulate(case.batch)


# --- Rows next to pages that cannot be read ------------------------------------------------

GUARD_DIMS = (1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 31, 33, 47, 63, 65, 301)


class _Guarded:
    """Memory of nbytes between two pages without access, at the end or at the start."""

    def __init__(self, nbytes: int, at_end: bool):
        page = mmap.PAGESIZE
        pages = max(1, -(-nbytes // page))
        self.map = mmap.mmap(-1, (pages + 2) * page)
        char = ctypes.c_char.from_buffer(self.map)
        base = ctypes.addressof(char)
        del char
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        for index in (0, pages + 1):
            assert libc.mprotect(base + index * page, page, 0) == 0, ctypes.get_errno()
        self.offset = page + (pages * page - nbytes if at_end else 0)
        self.nbytes = nbytes

    def array(self, values: np.ndarray) -> np.ndarray:
        view = np.frombuffer(self.map, np.uint8, self.nbytes, self.offset)
        view = view.view(values.dtype).reshape(values.shape)
        view[...] = values
        return view


def _guard_case(dtype, dim):
    rng = _rng(dim)
    base = rng.standard_normal((3, dim)).astype(F32)
    ngram_base = rng.standard_normal((4, dim)).astype(F32)
    sentences = ((2,), (0, 2, OOV_TOKEN, 1), (OOV_TOKEN, 2, 0))
    ngrams = ((3, 0, 3), (0, 3))
    weights = rng.random(3).astype(F32)
    return Case("guard", dtype, base, weights, sentences, ngrams, ngram_base, 0.5)


@pytest.mark.skipif(sys.platform == "win32", reason="mprotect: POSIX only")
@pytest.mark.parametrize("at_end", [True, False], ids=["end", "start"])
@pytest.mark.parametrize("dim", GUARD_DIMS)
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("isa", ISAS)
def test_no_read_or_write_beyond_a_row(isa, dtype, dim, at_end):
    """The first and the last row of the word and ngram tables, and out, touch a page
    without access: a load or a store past a row (or before it) faults."""
    case = _guard_case(dtype, dim)
    a = case.arrays
    rows, ngram_rows = a["rows"], a["ngram_rows"]
    guards = [_Guarded(x.nbytes, at_end) for x in (rows, ngram_rows)]
    kernel = _kernel(case, isa, rows=guards[0].array(rows), ngram_rows=guards[1].array(ngram_rows))
    out_guard = _Guarded(3 * dim * 4, at_end)
    out = out_guard.array(np.zeros((3, dim), F32))
    kernel.pool(case.batch, out)
    assert_bits_equal(out, _reference.pool_weighted_mean(case.batch, *case.args()))
