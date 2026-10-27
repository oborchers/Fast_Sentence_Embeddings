"""The conformance suite: every backend gives the bits of fse._reference.

ARCHITECTURE.md, "Backends": a backend is done when it passes this suite. Every case runs
for every backend in BACKENDS; a backend that cannot run a case skips it and names the
capability. PR 4 adds the CPU runtime backend, PR 9 the MLX backend, later CUDA, by
appending to BACKENDS. Meta-tests at the end check that the suite catches typical
mistakes (no fma, acc / n, an EPS start), so it cannot pass by accident.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import pytest

from fse import _native, _reference
from fse._contract import OOV_TOKEN, Batch

F32 = np.float32
TINY = F32(np.finfo(F32).smallest_subnormal)
BIG = F32(np.finfo(F32).max)


# --- Cases --------------------------------------------------------------------------------


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


def _random_case(name, dtype, seed, *, words=40, dim=13, n=25, length=12, empty=0.15, ft=False):
    rng = _rng(seed)
    base = rng.standard_normal((words, dim)).astype(F32)
    weights = rng.random(words).astype(F32)
    sentences, ngrams = [], []
    for _ in range(n):
        size = 0 if rng.random() < empty else int(rng.integers(1, length + 1))
        sentence = []
        for _ in range(size):
            if ft and rng.random() < 0.3:
                sentence.append(OOV_TOKEN)
                ngrams.append(tuple(int(b) for b in rng.integers(0, 17, rng.integers(0, 6))))
            else:
                sentence.append(int(rng.integers(0, words)))
        sentences.append(tuple(sentence))
    if not ft:
        return Case(name, dtype, base, weights, tuple(sentences))
    ngram_base = rng.standard_normal((17, dim)).astype(F32)
    return Case(name, dtype, base, weights, tuple(sentences), tuple(ngrams), ngram_base, 0.75)


def _cases():
    out = []
    for dtype in ("float32", "float16", "int8"):
        rng = _rng(hash(dtype) % 2**32)
        base = rng.standard_normal((30, 9)).astype(F32)
        w = rng.random(30).astype(F32)
        out += [
            Case("no sentences", dtype, base, w, ()),
            Case("all empty", dtype, base, w, ((), (), ())),
            Case("empty first middle last", dtype, base, w, ((), (1, 2), (), (3,), ())),
            Case("single tokens", dtype, base, w, tuple((i,) for i in range(30))),
            Case(
                "1000 tokens", dtype, base, w, (tuple(int(i) for i in rng.integers(0, 30, 1000)),)
            ),
            Case(
                "5000 tokens",
                dtype,
                base,
                w,
                (tuple(int(i) for i in rng.integers(0, 30, 5000)), (1,)),
            ),
            Case("one word repeated", dtype, base, w, ((7,) * 1000, (7,) * 3)),
        ]
        for dim in (1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 300, 301):
            out.append(_random_case(f"dim {dim}", dtype, dim, dim=dim, n=6))
        out.append(_random_case("fasttext", dtype, 11, ft=True))
        out.append(
            Case(
                "only unknown words",
                dtype,
                base,
                w,
                ((OOV_TOKEN,), (OOV_TOKEN, OOV_TOKEN), (OOV_TOKEN,) * 3),
                ((), (0,), (5, 5, 5), (0, 16), (16,), tuple(range(17)) * 12),
                rng.standard_normal((17, 9)).astype(F32),
                0.5,
            )
        )
        out.append(
            Case(
                "fasttext without unknown words",
                dtype,
                base,
                w,
                ((1, 2), ()),
                (),
                rng.standard_normal((17, 9)).astype(F32),
                0.5,
            )
        )
        sub = np.full((4, 9), TINY, dtype=F32) * np.arange(1, 37, dtype=F32).reshape(4, 9)
        out.append(
            Case("subnormal rows", dtype, sub, np.full(4, 0.5, F32), ((0, 1, 2, 3), (0, 0), (3,)))
        )
        small = rng.standard_normal((5, 9)).astype(F32) * F32(1e-30)
        out.append(
            Case("subnormal products", dtype, small, np.full(5, 1e-10, F32), ((0, 1, 2), (4,) * 50))
        )
        cross = np.vstack([np.full(9, TINY * 64, F32), np.full(9, F32(1e-37), F32)])
        out.append(Case("subnormal to normal", dtype, cross, np.ones(2, F32), ((0,) * 9 + (1,),)))
        signed = np.array([[-0.0, 1.0, -2.0], [0.0, -1.0, 2.0]], dtype=F32)
        out.append(
            Case("signed zeros", dtype, signed, np.array([-0.0, 1.0], F32), ((0,), (0, 1), (1, 0)))
        )
        out.append(Case("large first row", dtype, base, w, ((1, 2), (3,)), first_row=2**40))
        for seed in range(12):
            out.append(
                _random_case(
                    f"random {seed}",
                    dtype,
                    100 + seed,
                    ft=seed % 3 == 0,
                    dim=int(_rng(seed).integers(1, 40)),
                )
            )
    # float32 only: exact constructions.
    f, x = F32(1 + 2**-12), F32(1 + 2**-12)
    fused = np.array([[x], [-(f * x)]], dtype=F32)
    out.append(
        Case("fused, not rounded products", "float32", fused, np.array([f, 1], F32), ((0, 1),))
    )
    v = F32(1 + 2**-23)
    out.append(
        Case(
            "inverse length, not division",
            "float32",
            np.array([[v]], F32),
            np.ones(1, F32),
            ((0, 0, 0),),
        )
    )
    tiny_scale = np.array([[1e-38, 2e-38, 0.0], [127.0, -127.0, 1.0]], dtype=F32)
    out.append(
        Case("int8 tiny scales", "int8", tiny_scale, np.array([1e-3, 0.5], F32), ((0, 1), (0,)))
    )
    zero_row = np.zeros((2, 5), dtype=F32)
    out.append(Case("int8 zero rows", "int8", zero_row, np.ones(2, F32), ((0, 1),)))
    half = np.array([[65504.0, -65504.0, 6e-8, 1.0]], dtype=F32)
    out.append(Case("float16 extremes", "float16", half, np.ones(1, F32) * F32(0.25), ((0, 0),)))
    return out


CASES = _cases()
OVERFLOW = Case(
    "overflow", "float32", np.array([[BIG, 1.0]], F32), np.ones(1, F32), ((0,), (0, 0)), first_row=7
)


@functools.cache
def expected(index: int, kind: str):
    case = CASES[index]
    if kind == "pool":
        return _reference.pool_weighted_mean(case.batch, *case.args())
    return _reference.accumulate(case.batch, *case.args())


# --- Backends -------------------------------------------------------------------------------


class Reference:
    name = "reference"

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


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("index", range(len(CASES)), ids=[f"{c.dtype}-{c.name}" for c in CASES])
def test_pool(backend, index):
    assert_bits_equal(backend.pool(CASES[index]), expected(index, "pool"), CASES[index].name)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("index", range(len(CASES)), ids=[f"{c.dtype}-{c.name}" for c in CASES])
def test_accumulate(backend, index):
    accs, counts = backend.accumulate(CASES[index])
    want_accs, want_counts = expected(index, "accumulate")
    assert_bits_equal(accs, want_accs, CASES[index].name)
    np.testing.assert_array_equal(counts, want_counts)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_overflow_raises_like_the_reference(backend):
    with pytest.raises(ValueError, match="row 8 contains NaN or infinite values"):
        backend.pool(OVERFLOW)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_batch_splits_give_the_same_rows(backend):
    case = _random_case("split", "int8", 5, n=20, ft=True)
    whole = backend.pool(case)
    for cut in (0, 1, 7, 19, 20):
        first = _split(case, 0, cut)
        second = _split(case, cut, 20)
        parts = [backend.pool(first), backend.pool(second)]
        assert_bits_equal(np.concatenate(parts), whole, f"cut {cut}")


def _split(case, begin, end):
    unknown_before = sum(s.count(OOV_TOKEN) for s in case.sentences[:begin])
    unknown_in = sum(s.count(OOV_TOKEN) for s in case.sentences[begin:end])
    return Case(
        case.name,
        case.dtype,
        case.base,
        case.weights,
        case.sentences[begin:end],
        case.ngrams[unknown_before : unknown_before + unknown_in],
        case.ngram_base,
        case.oov_weight,
        first_row=begin,
        stored=case.arrays,
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_v1_fixture_batches(backend, test_data_dir):
    """The batches and the learned rows of the 1.1.0 fixture (tests/test_data/v1_reference)."""
    with np.load(test_data_dir / "v1_reference" / "outputs.npz") as data:
        arrays = dict(data)
    for name in ("Average", "SIF", "uSIF", "AverageFT"):
        kind = "ft" if name == "AverageFT" else "kv"
        keys = ("offsets", "tokens", "ngram_offsets", "ngrams")
        batch = Batch(
            0, **{k: arrays[f"batch.{kind}.{k}"] for k in keys if f"batch.{kind}.{k}" in arrays}
        )
        for dtype in ("float32", "float16", "int8"):
            prefix = f"{name}.{dtype}"
            if dtype == "float32":
                rows = arrays["ft.vectors_vocab" if kind == "ft" else "kv.vectors"]
                ngram_rows = arrays["ft.vectors_ngrams"] if kind == "ft" else None
            else:
                rows = arrays[f"{prefix}.rows"]
                ngram_rows = arrays.get(f"{prefix}.ngram_rows")
            weights = arrays[f"{prefix}.weights"]
            args = (
                rows,
                weights,
                arrays.get(f"{prefix}.scales"),
                ngram_rows,
                arrays.get(f"{prefix}.ngram_scales"),
                float(weights.max()) if kind == "ft" else None,
            )
            want = _reference.pool_weighted_mean(batch, *args)
            if isinstance(backend, Native):
                got = _native.pool_weighted_mean(batch, *args, isa=backend.isa)
            else:
                got = want
            assert_bits_equal(got, want, prefix)


# --- Meta-tests: the suite catches typical mistakes ----------------------------------------


def _numpy_pool(case, *, fused=True, divide=False, start=0.0):
    a = case.arrays
    rows = _reference.dequantize(a["rows"], None)
    scales = a["scales"]
    out = np.zeros((len(case.sentences), case.base.shape[1]), dtype=F32)
    for s, sentence in enumerate(case.sentences):
        acc = np.full(case.base.shape[1], start, dtype=F32)
        for t in sentence:
            factor = case.weights[t] if scales is None else F32(case.weights[t] * scales[t])
            if fused:
                acc = _reference.fma32(factor, rows[t], acc)
            else:
                acc = (factor * rows[t]).astype(F32) + acc
        if sentence or start:
            out[s] = (
                acc / F32(len(sentence) or 1) if divide else F32(1 / F32(len(sentence) or 1)) * acc
            )
    return out


@pytest.mark.parametrize(
    ("mistake", "kwargs"),
    [
        ("multiply and add without fma", {"fused": False}),
        ("acc / n instead of (1/n) * acc", {"divide": True}),
        ("an EPS start", {"start": float(np.finfo(F32).eps)}),
    ],
)
def test_the_suite_catches(mistake, kwargs):
    failures = 0
    for index, case in enumerate(CASES):
        if case.ngrams is not None:
            continue
        try:
            assert_bits_equal(_numpy_pool(case, **kwargs), expected(index, "pool"))
        except AssertionError:
            failures += 1
    assert failures, f"no case catches {mistake}"
