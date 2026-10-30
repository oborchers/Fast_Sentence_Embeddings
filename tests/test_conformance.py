"""The conformance suite: every backend gives the bits of fse._reference.

ARCHITECTURE.md, "Backends": a backend is done when it passes this suite. Every case runs
for every backend in BACKENDS (the pooled rows of a device backend, mlx, are compared
under the bound of its tolerance contract in tests/test_conformance_device.py instead);
a backend that cannot run a case skips it and names the capability. PR 4 adds the CPU
runtime backend, PR 9 the MLX backend, later CUDA, by appending to BACKENDS in
tests/support/conformance.py; dot16, the steps and score are in
tests/test_conformance_steps.py. Meta-tests at the end check that the suite catches typical
mistakes (no fma, acc / n, an EPS start), so it cannot pass by accident.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from fse import _native, _reference
from fse._contract import OOV_TOKEN, Batch, Normalize
from support.cases import CASES, OVERFLOW, random_case
from support.conformance import (
    BACKENDS,
    BIG,
    EXACT,
    F32,
    TINY,
    Case,
    Contract,
    Native,
    Reference,
    _rng,
    assert_bits_equal,
)

# The mlx backend warns at the first fallback of an encoder, which these suites cause.
pytestmark = pytest.mark.expected_log("backend.fallback")


@functools.cache
def expected(index: int, kind: str):
    case = CASES[index]
    if kind == "pool":
        return _reference.pool_weighted_mean(case.batch, *case.args())
    return _reference.accumulate(case.batch, *case.args())


# --- Backends -------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", EXACT, ids=lambda b: b.name)
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
    case = random_case("split", "int8", 5, n=20, ft=True)
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
        batch = _v1_batch(arrays, kind)
        for dtype in ("float32", "float16", "int8"):
            prefix = f"{name}.{dtype}"
            args = _v1_args(arrays, prefix, kind, dtype)
            want = _reference.pool_weighted_mean(batch, *args)
            assert_bits_equal(backend.pool_args(batch, args), want, prefix)


def _v1_batch(arrays, kind):
    keys = ("offsets", "tokens", "ngram_offsets", "ngrams")
    return Batch(
        0, **{k: arrays[f"batch.{kind}.{k}"] for k in keys if f"batch.{kind}.{k}" in arrays}
    )


def _v1_args(arrays, prefix, kind, dtype):
    """The arguments of the weighted mean for one model and dtype of the fixture."""
    if dtype == "float32":
        rows = arrays["ft.vectors_vocab" if kind == "ft" else "kv.vectors"]
        ngram_rows = arrays["ft.vectors_ngrams"] if kind == "ft" else None
    else:
        rows = arrays[f"{prefix}.rows"]
        ngram_rows = arrays.get(f"{prefix}.ngram_rows")
    weights = arrays[f"{prefix}.weights"]
    return (
        rows,
        weights,
        arrays.get(f"{prefix}.scales"),
        ngram_rows,
        arrays.get(f"{prefix}.ngram_scales"),
        float(weights.max()) if kind == "ft" else None,
    )


# --- Meta-tests: the suite catches typical mistakes ----------------------------------------


def _numpy_pool(case, *, fused=True, divide=False, start=0.0):
    a = case.arrays
    rows = _reference.dequantize(a["rows"], None)
    scales = a["scales"]
    out = np.zeros((len(case.sentences), case.base.shape[1]), dtype=F32)
    for s, sentence in enumerate(case.sentences):
        acc = _numpy_acc(case, sentence, (rows, scales), fused, start)
        if sentence or start:
            out[s] = (
                acc / F32(len(sentence) or 1) if divide else F32(1 / F32(len(sentence) or 1)) * acc
            )
    return out


def _numpy_acc(case, sentence, table, fused, start):
    """The sum of one sentence, with a fused or a separate multiply and add."""
    rows, scales = table
    acc = np.full(case.base.shape[1], start, dtype=F32)
    for t in sentence:
        factor = case.weights[t] if scales is None else F32(case.weights[t] * scales[t])
        if fused:
            acc = _reference.fma32(factor, rows[t], acc)
        else:
            acc = (factor * rows[t]).astype(F32) + acc
    return acc


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


def test_the_fused_case_catches_a_missing_fma():
    index = next(i for i, c in enumerate(CASES) if c.name == "fused, not rounded products")
    with pytest.raises(AssertionError):
        assert_bits_equal(_numpy_pool(CASES[index], fused=False), expected(index, "pool"))


def test_the_suite_catches_zero_padded_dot16():
    """Padding a tail with zeros turns a -0.0 lane into +0.0."""
    rows = np.full((1, 17), -(2.0**-149), dtype=F32)
    vector = np.full(17, 0.5, dtype=F32)
    padded = _reference.dot16(np.pad(rows, ((0, 0), (0, 15))), np.pad(vector, (0, 15)))
    with pytest.raises(AssertionError):
        assert_bits_equal(padded[None, :], _reference.dot16(rows, vector)[None, :])


# --- The floating-point environment ----------------------------------------------------------

# The kernels, the cpu backend and the mlx backend (the GPU and its cpu host) set the
# environment themselves; the numpy backend, like the reference, computes in the
# environment of its thread.
NATIVE = [
    b
    for b in BACKENDS
    if isinstance(b, Native) or (isinstance(b, Contract) and b.backend_name in ("cpu", "mlx"))
]
# Flush-to-zero changes subnormal results, rounding up changes inexact ones.
ENVIRONMENTS = {"flush to zero": {"flush": True}, "round up": {"round_up": True}}


def _environment_calls(backend):
    """Every native entry point, on inputs whose results depend on the environment."""
    rows = np.full((3, 20), TINY, dtype=F32) * np.arange(1, 61, dtype=F32).reshape(3, 20)
    inexact = _rng(5).standard_normal((3, 20)).astype(F32)
    # Normal values whose sum of squares is subnormal.
    small = np.full((1, 4), F32(2.0**-70), dtype=F32)
    # Inputs are made here, before the environment changes: the calls must only run kernels.
    tiny, shifted = float(TINY), {"subnormal": rows + F32(1), "inexact": inexact + F32(1)}
    third = np.full(20, 0.3, F32)
    pool_cases = [
        c for c in CASES if c.dtype == "float32" and c.name in ("subnormal rows", "random 1")
    ]
    calls = {}
    for case in pool_cases:
        calls[f"pool {case.name}"] = lambda case=case: backend.pool(case)
        calls[f"accumulate {case.name}"] = lambda case=case: backend.accumulate(case)[0]
    for label, x in (("subnormal", rows), ("inexact", inexact)):
        calls[f"normalize {label}"] = lambda y=shifted[label]: backend.postprocess(y, [Normalize()])
        calls[f"dot16 {label}"] = lambda x=x: backend.dot16(x, third)[:, None]
        calls[f"score {label}"] = lambda x=x: backend.score(x, None, third, tiny)[:, None]
    calls["normalize small"] = lambda: backend.postprocess(small, [Normalize()])
    zeros, ones = np.zeros((1, 3), F32), np.ones(3, F32)
    calls["score subnormal bias"] = lambda: backend.score(zeros, None, ones, 2.0**-149)[:, None]
    return calls


@pytest.mark.parametrize("env", ENVIRONMENTS, ids=list(ENVIRONMENTS))
def test_the_environment_changes_numpy(env):
    """The switches the next tests use really change results in this process."""
    a, b, c = np.array([TINY * 64], F32), np.ones(1, F32), np.array([2.0**-30], F32)
    before = (a * F32(0.5), b + c)
    saved = _native._test_fpenv_set(**ENVIRONMENTS[env])
    try:
        after = (a * F32(0.5), b + c)
    finally:
        _native._test_fpenv_restore(saved)
    assert after[0][0] != before[0][0] or after[1][0] != before[1][0]


ENVIRONMENT_CALLS = list(_environment_calls(Reference()))


def test_the_environment_cases_depend_on_the_environment():
    want = {name: call() for name, call in _environment_calls(Reference()).items()}
    assert np.any(want["pool subnormal rows"] != 0)
    assert want["score subnormal bias"][0, 0] != 0


@pytest.mark.parametrize("name", ENVIRONMENT_CALLS)
@pytest.mark.parametrize("env", ENVIRONMENTS, ids=list(ENVIRONMENTS))
@pytest.mark.parametrize("backend", NATIVE, ids=lambda b: b.name)
def test_native_ignores_and_restores_the_environment(backend, env, name):
    """One entry point per test, so a backend that skips one still runs the others."""
    want = _environment_calls(Reference())[name]()
    call = _environment_calls(backend)[name]
    saved = _native._test_fpenv_set(**ENVIRONMENTS[env])
    try:
        changed = _native._test_fpenv_get()
        got = call()
        after = _native._test_fpenv_get()
    finally:
        _native._test_fpenv_restore(saved)
    assert after == changed, f"{name} did not restore the floating-point environment"
    assert_bits_equal(got, want, name)


@pytest.mark.parametrize("backend", NATIVE, ids=lambda b: b.name)
def test_native_restores_the_environment_after_an_error(backend):
    saved = _native._test_fpenv_set(flush=True, round_up=True)
    try:
        changed = _native._test_fpenv_get()
        with pytest.raises(ValueError, match="score of row 0 overflows"):
            backend.score(np.array([[BIG, 0]], dtype=F32), None, np.array([2, 0], F32), 0.0)
        after = _native._test_fpenv_get()
    finally:
        _native._test_fpenv_restore(saved)
    assert after == changed


def _oov_case(oov_weight, unknown):
    tokens = np.array([OOV_TOKEN, 0] if unknown else [0], np.uint32)
    batch = Batch(
        0,
        np.array([0, len(tokens)], np.int64),
        tokens,
        np.array([0, 1] if unknown else [0], np.int64),
        np.zeros(1 if unknown else 0, np.uint32),
    )
    args = (np.ones((1, 2), F32), np.ones(1, F32), None, np.ones((1, 2), F32), None, oov_weight)
    return batch, args


def test_an_oov_weight_beyond_float32_behaves_as_in_the_reference():
    batch, args = _oov_case(1e100, unknown=False)
    want = _reference.pool_weighted_mean(batch, *args)
    for isa in _native.available_isas():
        assert_bits_equal(_native.pool_weighted_mean(batch, *args, isa=isa), want, isa)
    batch, args = _oov_case(1e100, unknown=True)
    with pytest.raises(ValueError, match="row 0 contains NaN or infinite"):
        _reference.pool_weighted_mean(batch, *args)
    for isa in _native.available_isas():
        with pytest.raises(ValueError, match="row 0 contains NaN or infinite"):
            _native.pool_weighted_mean(batch, *args, isa=isa)
