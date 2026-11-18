"""Observe both tile builders, including NULL buckets with non-NULL offsets."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from fse import _native, _reference
from fse._contract import OOV_TOKEN, Batch
from support.conformance import F32, Case, _rng, assert_bits_equal
from support.tiles import prefetch_addresses, prefetch_digest


@pytest.fixture
def instrumented():
    if _native._test_builder_trace() is None:
        assert os.environ.get("FSE_TEST_TILES") != "1", "rebuild with test instrumentation"
        pytest.skip("production build has no builder counters")


def _case(kind, dim, dtype="float32"):
    rng = _rng(dim)
    sentences = ((), tuple(range(10)) * 13, (1, 2, 3), ())
    ngrams = None
    ngram_base = None
    if kind != "known":
        ngram_base = rng.standard_normal((80, dim)).astype(F32)
    if kind == "known offsets":
        ngrams = ()  # Supplied offsets [0], empty bucket array becomes NULL in C.
    elif kind in ("empty buckets", "mixed buckets"):
        sentences = ((), (1, OOV_TOKEN, 2) * 45, (OOV_TOKEN, 3), ())
        ngrams = ((),) * 46 if kind == "empty buckets" else (tuple(range(80)),) + ((),) * 45
    return Case(
        kind,
        dtype,
        rng.standard_normal((10, dim)).astype(F32),
        rng.random(10).astype(F32),
        sentences,
        ngrams,
        ngram_base,
        0.75 if ngram_base is not None else None,
    )


@pytest.mark.parametrize("isa", _native.available_isas())
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dim", [33, 127, 300, 1024, 1025])
@pytest.mark.parametrize(
    "kind", ["known", "known FastText", "known offsets", "empty buckets", "mixed buckets"]
)
@pytest.mark.usefixtures("instrumented")
def test_offsets_select_builder_with_identical_bits_counts_and_hints(isa, dtype, dim, kind):
    """All-empty ranges must use general, even though their bucket pointer is NULL."""
    case = _case(kind, dim, dtype)
    kernel = _native.WeightedMean(*case.args(), isa=isa)
    batch = case.batch
    stride = _native._test_prefetch_trace()[0]
    tiled = dtype == "float32" and isa in ("avx2", "neon") and dim <= 1024
    known_only = batch.ngram_offsets is None
    for method in ("pool", "accumulate", "pool"):
        got = getattr(kernel, method)(batch)
        known, general = _native._test_builder_trace()
        if tiled:
            assert (known > 0, general > 0) == (known_only, not known_only)
        else:
            assert (known, general) == (0, 0)
        if method == "pool":
            assert_bits_equal(got, _reference.pool_weighted_mean(batch, *case.args()))
        else:
            want, counts = _reference.accumulate(batch, *case.args())
            assert_bits_equal(got[0], want)
            np.testing.assert_array_equal(got[1], counts)
        assert _native._test_prefetch_trace() == (
            stride,
            *prefetch_digest(prefetch_addresses(case, stride)),
        )


@pytest.mark.parametrize("isa", _native.available_isas())
@pytest.mark.usefixtures("instrumented")
def test_builder_counters_reset_between_paths_and_are_thread_local(isa):
    barrier = Barrier(2)

    def probe(kind):
        case = _case(kind, 127)
        kernel = _native.WeightedMean(*case.args(), isa=isa)
        kernel.pool(case.batch)
        before = _native._test_builder_trace()
        barrier.wait(timeout=10)
        assert _native._test_builder_trace() == before
        kernel.pool(Batch(0, np.array([0, 0], np.int64), np.array([], np.uint32)))
        after = _native._test_builder_trace()
        return before, after

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(probe, ("known", "empty buckets")))
    if isa in ("avx2", "neon"):
        assert results[0][0][0] > 0
        assert results[0][0][1] == 0
        assert results[1][0][0] == 0
        assert results[1][0][1] > 0
        assert all(after == (1, 0) for _, after in results)
    else:
        assert all(before == after == (0, 0) for before, after in results)


@pytest.mark.parametrize("isa", _native.available_isas())
def test_empty_batch_output_is_unchanged(isa):
    case = Case("empty", "float32", np.ones((1, 127), F32), np.ones(1, F32), ())
    kernel = _native.WeightedMean(*case.args(), isa=isa)
    assert_bits_equal(
        kernel.pool(case.batch), _reference.pool_weighted_mean(case.batch, *case.args())
    )
    got, counts = kernel.accumulate(case.batch)
    want, want_counts = _reference.accumulate(case.batch, *case.args())
    assert_bits_equal(got, want)
    np.testing.assert_array_equal(counts, want_counts)


@pytest.mark.parametrize("isa", _native.available_isas())
@pytest.mark.parametrize("kind", ["known", "empty buckets"])
@pytest.mark.usefixtures("instrumented")
def test_zero_sentence_batch_resets_builder_counters(instrumented, isa, kind):
    case = _case(kind, 127)
    kernel = _native.WeightedMean(*case.args(), isa=isa)
    empty = Batch(0, np.array([0], np.int64), np.array([], np.uint32))
    for method in ("pool", "accumulate"):
        kernel.pool(case.batch)
        before = _native._test_builder_trace()
        assert sum(before) > 0 if isa in ("avx2", "neon") else before == (0, 0)
        getattr(kernel, method)(empty)
        assert _native._test_builder_trace() == (0, 0)
