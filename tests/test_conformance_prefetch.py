"""Observe actual hints separately from the numerical conformance proof."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from fse import _native, _reference
from fse._contract import OOV_TOKEN
from support.conformance import F32, Case, _rng, assert_bits_equal
from support.tiles import prefetch_addresses, prefetch_digest, prefetch_lines


@pytest.fixture
def stride():
    trace = _native._test_prefetch_trace()
    if trace is None:
        assert os.environ.get("FSE_TEST_TILES") != "1", "rebuild with test instrumentation"
        pytest.skip("production build has no address recording")
    expected = int(os.environ.get("FSE_EXPECT_PREFETCH_LINE_STRIDE", "1"))
    assert trace[0] == expected, "wrong compiled prefetch stride"
    return expected


def _case(dim, dtype):
    rng = _rng(dim)
    sentences = (
        (),
        (1,) * 10 + (OOV_TOKEN, 2, OOV_TOKEN, 3),
        (OOV_TOKEN,) + tuple(range(20)) * 7,
        (),
        (OOV_TOKEN, 1, 2),
    )
    case = Case(
        "prefetch",
        dtype,
        rng.standard_normal((20, dim)).astype(F32),
        rng.random(20).astype(F32),
        sentences,
        (tuple(range(100)) * 2, (), tuple(range(99, -1, -1)), (1,) * 7),
        rng.standard_normal((100, dim)).astype(F32),
        0.75,
    )
    # Distinct, deliberately unaligned table bases. Keep element alignment for SIMD.
    for name in ("rows", "ngram_rows"):
        array = case.arrays[name]
        storage = np.empty(array.nbytes + 128, np.uint8)
        start = (-storage.ctypes.data % 64) + array.itemsize
        shifted = storage[start : start + array.nbytes].view(array.dtype).reshape(array.shape)
        shifted[...] = array
        case.stored[name] = shifted
    return case


@pytest.mark.parametrize("isa", _native.available_isas())
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dim", [1, 15, 16, 17, 63, 64, 65, 127, 256, 257, 300, 1024, 1025])
def test_native_hints_match_row_model(stride, isa, dtype, dim):
    case = _case(dim, dtype)
    kernel = _native.WeightedMean(*case.args(), isa=isa)
    for _ in range(2):  # Every call resets its own trace.
        got = kernel.pool(case.batch)
        assert _native._test_prefetch_trace() == (
            stride,
            *prefetch_digest(prefetch_addresses(case, stride)),
        )
        assert_bits_equal(got, _reference.pool_weighted_mean(case.batch, *case.args()))
    kernel.pool(Case("empty", dtype, case.base, case.weights, ((),)).batch)
    assert _native._test_prefetch_trace() == (stride, 0, 0)


@pytest.mark.parametrize("size", [0, 1, 4, 63, 64, 65, 1024, 1200, 4096])
def test_helper_alignment_and_tail(stride, size):
    storage = np.empty(size + 128, np.uint8)
    aligned = -storage.ctypes.data % 64
    for offset in range(64):
        row = storage[aligned + offset : aligned + offset + size]
        assert _native._test_prefetch_row(row) == (
            stride,
            *prefetch_digest(prefetch_lines(row.ctypes.data, size, stride)),
        )


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 16, 65])
@pytest.mark.parametrize("mistake", ["omit", "duplicate", "reverse", "wrong table"])
def test_trace_detects_wrong_addresses(stride, mistake):
    case = _case(257, "float32")
    addresses = list(prefetch_addresses(case, stride))
    wrong = {
        "omit": addresses[1:],
        "duplicate": [addresses[0], *addresses],
        "reverse": addresses[::-1],
        "wrong table": [address + 64 for address in addresses],
    }[mistake]
    assert prefetch_digest(wrong) != prefetch_digest(addresses)


def test_trace_is_thread_local(stride):
    barrier = Barrier(4)

    def probe(offset):
        data = np.empty(2048, np.uint8)[offset : offset + 1024]
        want = stride, *prefetch_digest(prefetch_lines(data.ctypes.data, data.size, stride))
        _native._test_prefetch_row(data)
        barrier.wait(timeout=10)
        return _native._test_prefetch_trace(), want

    with ThreadPoolExecutor(4) as pool:
        for got, want in pool.map(probe, range(32)):
            assert got == want
