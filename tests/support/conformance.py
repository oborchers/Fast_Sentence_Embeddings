"""Shared parts of the conformance suites: the case record, the backends, the comparison.

ARCHITECTURE.md, "Backends": a backend is done when it passes the conformance suites.
PR 4 adds the CPU runtime backend, PR 9 the MLX backend, later CUDA, by appending to
BACKENDS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fse import _native, _reference
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
        return _reference.pool_weighted_mean(case.batch, *case.args())

    def accumulate(self, case):
        return _reference.accumulate(case.batch, *case.args())


class Native:
    def __init__(self, isa):
        self.isa = isa
        self.name = f"native-{isa}"

    def pool(self, case):
        return _native.pool_weighted_mean(case.batch, *case.args(), isa=self.isa)

    def accumulate(self, case):
        return _native.accumulate(case.batch, *case.args(), isa=self.isa)

    def postprocess(self, rows, steps, first_row=0):
        return _native.postprocess(rows, steps, first_row, isa=self.isa)

    def score(self, stored, scales, weights, bias):
        return _native.score(stored, scales, weights, bias, isa=self.isa)

    def dot16(self, rows, other):
        return _native.dot16(rows, other, isa=self.isa)


BACKENDS = [Reference(), *(Native(isa) for isa in _native.available_isas())]


def assert_bits_equal(got, want, name=""):
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
    g = got.view(np.uint32) if got.dtype == F32 else got
    w = want.view(np.uint32) if want.dtype == F32 else want
    if not np.array_equal(g, w):
        row, col = (int(i) for i in np.argwhere(g != w)[0])
        raise AssertionError(
            f"{name}: row {row}, column {col}: {got[row, col]!r} ({g[row, col]:#010x}) "
            f"!= {want[row, col]!r} ({w[row, col]:#010x})"
        )
