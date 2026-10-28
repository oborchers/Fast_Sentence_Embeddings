"""Shared parts of the conformance suites: the case record, the backends, the comparison.

ARCHITECTURE.md, "Backends": a backend is done when it passes the conformance suites.
The kernels run directly (Reference, Native) and every backend of fse._backends runs
through its contract (Contract); PR 9 adds the MLX backend, later CUDA, by appending
to BACKENDS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from fse import _native, _reference
from fse._backends import PoolInput, get_backend
from fse._contract import Batch

F32 = np.float32
TINY = F32(np.finfo(F32).smallest_subnormal)
BIG = F32(np.finfo(F32).max)


@dataclass(frozen=True)
class Case:
    """Word rows, weights and sentences of one conformance case."""

    name: str
    dtype: str
    base: np.ndarray  # float32 rows before storage
    weights: np.ndarray
    sentences: tuple  # tuples of token ids; OOV_TOKEN for unknown FastText words
    ngrams: tuple | None = None  # bucket ids per unknown word, in order
    ngram_base: np.ndarray | None = None
    oov_weight: float | None = None
    first_row: int = 0
    stored: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def arrays(self):
        if not self.stored:
            rows, scales = _reference.quantize_rows(self.base, self.dtype)
            self.stored.update(rows=rows, scales=scales)
            if self.ngram_base is not None:
                nrows, nscales = _reference.quantize_rows(self.ngram_base, self.dtype)
                self.stored.update(ngram_rows=nrows, ngram_scales=nscales)
        return self.stored

    @property
    def batch(self) -> Batch:
        offsets = np.cumsum([0, *map(len, self.sentences)]).astype(np.int64)
        tokens = np.array([t for s in self.sentences for t in s], dtype=np.uint32)
        if self.ngrams is None:
            return Batch(self.first_row, offsets, tokens)
        ngram_offsets = np.cumsum([0, *map(len, self.ngrams)]).astype(np.int64)
        flat = np.array([b for n in self.ngrams for b in n], dtype=np.uint32)
        return Batch(self.first_row, offsets, tokens, ngram_offsets, flat)

    def args(self):
        a = self.arrays
        return (
            a["rows"],
            self.weights,
            a["scales"],
            a.get("ngram_rows"),
            a.get("ngram_scales"),
            self.oov_weight,
        )


def _rng(seed):
    return np.random.default_rng(seed)


class Reference:
    name = "reference"

    def postprocess(self, rows, steps, first_row=0):
        return _reference.postprocess(rows, steps, first_row)

    def score(self, stored, scales, weights, bias):
        return _reference.score(stored, scales, weights, bias)

    def dot16(self, rows, other):
        return _reference.dot16(rows, other)

    def pool(self, case):
        return self.pool_args(case.batch, case.args())

    def pool_args(self, batch, args):
        return _reference.pool_weighted_mean(batch, *args)

    def accumulate(self, case):
        return _reference.accumulate(case.batch, *case.args())


class Native:
    def __init__(self, isa):
        self.isa = isa
        self.name = f"native-{isa}"

    def pool(self, case):
        return self.pool_args(case.batch, case.args())

    def pool_args(self, batch, args):
        return _native.pool_weighted_mean(batch, *args, isa=self.isa)

    def accumulate(self, case):
        return _native.accumulate(case.batch, *case.args(), isa=self.isa)

    def postprocess(self, rows, steps, first_row=0):
        return _native.postprocess(rows, steps, first_row, isa=self.isa)

    def score(self, stored, scales, weights, bias):
        return _native.score(stored, scales, weights, bias, isa=self.isa)

    def dot16(self, rows, other):
        return _native.dot16(rows, other, isa=self.isa)


class Contract:
    """A backend of fse._backends through its contract: prepare once, then ``rows``.

    The steps run on pooled rows, as in the runtime: ``postprocess`` pools every row by
    itself with weight 1, which gives the row back bit for bit (fma(1, x, 0) = x and
    1 * x = x), except that -0.0 would become 0.0, so such input is refused. accumulate
    and dot16 are no operations of the contract; the kernel tests above cover them
    directly.
    """

    def __init__(self, name, **options):
        self.backend = get_backend(name, **options)
        self.name = "contract-" + "-".join([name, *map(str, options.values())])

    def pool(self, case):
        return self.pool_args(case.batch, case.args())

    def pool_args(self, batch, args, steps=()):
        prepared = self.backend.prepare(PoolInput(*args), steps)
        out = np.empty((batch.n_sentences, prepared.output_dim), dtype=F32)
        self.backend.rows(batch, prepared, out)
        return out

    def postprocess(self, rows, steps, first_row=0):
        assert not (np.signbit(rows) & (rows == 0)).any(), "-0.0 does not pool to itself"
        n = len(rows)
        batch = Batch(first_row, np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.uint32))
        return self.pool_args(batch, (np.ascontiguousarray(rows), np.ones(n, F32)), steps)

    def score(self, stored, scales, weights, bias):
        return self.backend.score(stored, scales, weights, bias)

    def accumulate(self, *_case):
        pytest.skip(f"{self.name}: accumulate is no contract operation")

    def dot16(self, *_rows):
        pytest.skip(f"{self.name}: dot16 is no contract operation")


BACKENDS = [
    Reference(),
    *(Native(isa) for isa in _native.available_isas()),
    *(Contract("cpu", isa=isa) for isa in _native.available_isas()),
    Contract("numpy"),
]


_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def assert_bits_equal(got, want, name=""):
    """``got`` and ``want`` have the same shape, dtype and bits (so -0.0 is not 0.0)."""
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
    unsigned = _UINT[got.dtype.itemsize]
    g = np.ascontiguousarray(got).view(unsigned)
    w = np.ascontiguousarray(want).view(unsigned)
    if not np.array_equal(g, w):
        where = tuple(int(i) for i in np.argwhere(g != w)[0])
        digits = 2 * got.dtype.itemsize
        raise AssertionError(
            f"{name}: at {where}: {got[where]!r} ({int(g[where]):#0{digits + 2}x}) "
            f"!= {want[where]!r} ({int(w[where]):#0{digits + 2}x})"
        )
