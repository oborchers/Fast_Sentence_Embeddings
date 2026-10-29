"""Meta-tests of the conformance suite for a device backend (mlx): it must pass the suite
on the GPU, not by computing everything on the host, and the suite must catch a device
kernel that flushes subnormal values, divides or does not fuse.

tests/test_conformance.py checks the bits of every case; these check how the mlx backend
got them: the cases without subnormal values never fall back to the host, the subnormal
ones do, and without its checks the GPU gets them wrong.
"""

import numpy as np
import pytest

from fse import _reference
from fse._backends import PoolInput
from fse._backends._mlx import _kernels, _sources, fallbacks
from support.cases import CASES
from support.conformance import Contract

# The cases the GPU cannot compute because a value of the arithmetic is subnormal.
SUBNORMAL = {
    ("float32", "subnormal rows"),
    ("float32", "subnormal products"),
    ("float32", "subnormal to normal"),
    ("int8", "subnormal products"),
    ("int8", "subnormal to normal"),
    ("int8", "int8 tiny scales"),
    ("float32", "tiny oov weight"),
    ("float32", "cancels to subnormal"),
    # The product 2**-140 inside an fma whose sum is normal: this GPU keeps it, one that
    # flushed it would not, so it is flagged too.
    ("float32", "subnormal product"),
}


@pytest.fixture
def mlx():
    return Contract("mlx").backend  # skips where it cannot run


def _pool(backend, case):
    prepared = backend.prepare(PoolInput(*case.args()), ())
    out = np.empty((case.batch.n_sentences, prepared.output_dim), np.float32)
    backend.rows(case.batch, prepared, out)
    return out, sum(fallbacks(prepared).values())


def _differs(out, case):
    want = _reference.pool_weighted_mean(case.batch, *case.args())
    return out.tobytes() != want.tobytes()


def _check_fallbacks(backend):
    """The bits of every case, and a host fallback exactly for the subnormal ones."""
    fell_back = set()
    for case in CASES:
        out, count = _pool(backend, case)
        assert not _differs(out, case), case.name
        if count:
            fell_back.add((case.dtype, case.name))
    assert fell_back == SUBNORMAL


def test_only_the_subnormal_cases_fall_back(mlx):
    _check_fallbacks(mlx)


def test_a_backend_that_always_falls_back_is_caught(mlx, monkeypatch):
    """Its bits are right (the host computes them), so only the count catches it."""
    real = _kernels.read
    monkeypatch.setattr(_kernels, "read", lambda *a: real(*a) | _sources.SUBNORMAL)
    with pytest.raises(AssertionError):
        _check_fallbacks(mlx)


def test_without_the_checks_the_subnormal_cases_fail(mlx, monkeypatch):
    """The cases do reach the flush: without the flags the GPU's bits are wrong."""
    monkeypatch.setattr(mlx, "_checks", False)
    wrong = set()
    for case in CASES:
        out, count = _pool(mlx, case)
        assert count == 0
        if _differs(out, case):
            wrong.add((case.dtype, case.name))
    assert wrong == SUBNORMAL - {("float32", "subnormal product")}


def _variant(monkeypatch, old, new):
    """A pooling kernel with ``old`` replaced by ``new``, built apart from the real one."""
    assert old in _kernels.POOL + _kernels.HEADER
    monkeypatch.setattr(_kernels, "POOL", _kernels.POOL.replace(old, new))
    monkeypatch.setattr(_kernels, "HEADER", _kernels.HEADER.replace(old, new))
    monkeypatch.setattr(_kernels, "_KERNELS", {})


@pytest.mark.parametrize(
    ("old", "new", "case"),
    [
        ("fse_mul(inverse, acc, flag)", "acc / float(n)", "inverse length, not division"),
        ("metal::fma(a, b, c)", "a * b + c", "fused, not rounded products"),
    ],
    ids=["acc / n", "no fma"],
)
def test_the_suite_catches_a_kernel_that(mlx, monkeypatch, old, new, case):
    """With the checks off: they would send the unfused case to the host (its rounding
    error is a subnormal value), where the bits are right."""
    _variant(monkeypatch, old, new)
    monkeypatch.setattr(mlx, "_checks", False)
    (found,) = [c for c in CASES if c.name == case]
    out, count = _pool(mlx, found)
    assert count == 0
    assert _differs(out, found)
