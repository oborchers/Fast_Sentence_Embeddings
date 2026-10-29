"""Meta-tests of the conformance suite of a device backend: the comparison under the bound
(support.bound) fails for each mistake the tolerance contract names, on a witness case
that the mistaken kernel computes on the GPU (a launch, no host fallback).

The mistakes are variants of the real kernel's source (built apart from it), a read that
swaps rows, and rows moved by 10 times the bound. Two more show that the bit comparison
of tests/test_conformance.py catches what the bound allows (no fma, acc / n).
"""

import numpy as np
import pytest

from fse import _reference
from fse._backends._mlx import _kernels, _sources
from support.bound import assert_within_bound, bounds
from support.cases import CASES, random_case
from support.conformance import Contract, assert_bits_equal, pool_on_device


@pytest.fixture
def mlx():
    return Contract("mlx").backend  # skips where it cannot run


def _pool(backend, case):
    """The rows of ``case``, requiring that the GPU computed them."""
    out, fell_back, launched = pool_on_device(backend, case)
    assert launched, case.name
    assert not fell_back, case.name
    return out


def _variant(monkeypatch, *replacements):
    """A kernel with each ``old`` of its sources replaced by ``new``, built apart."""
    for old, new in replacements:
        for name in ("HEADER", "_POOL", "PREPARE"):
            text = getattr(_sources, name)
            if old in text:
                monkeypatch.setattr(_sources, name, text.replace(old, new))
                break
        else:
            raise AssertionError(f"not in the sources: {old!r}")
    monkeypatch.setattr(_kernels, "_KERNELS", {})


# Witnesses: distinct tokens of random words, no empty sentence, both kernel widths.
WITNESSES = [random_case(f"witness {dim}", "float32", 70 + dim, dim=dim, empty=0) for dim in (8, 9)]
KNOWN_FMA = "            acc = metal::fma(acc_t(f), x, acc);\n            flag |="
MISTAKES = {
    "drops a token": [
        ("for (uint t = begin; t < end; ++t)", "for (uint t = begin + 1u; t < end; ++t)")
    ],
    "repeats every token": [
        (
            KNOWN_FMA,
            "            acc = metal::fma(acc_t(f), x, metal::fma(acc_t(f), x, acc));\n            flag |=",
        )
    ],
    "divides by the wrong count": [("divide(1.0f, float(n))", "divide(1.0f, float(n + 1u))")],
    "skips 1 / n": [("fse_mul(inverse, acc, flag)", "acc")],
    "accumulates in float16": [
        (
            KNOWN_FMA,
            "            acc = fse_half(metal::fma(acc_t(f), x, acc));\n            flag |=",
        ),
        (
            "inline bool fse_nonfinite(uint bits)",
            "inline float fse_half(float x) { return float(half(x)); }\n"
            "inline float4 fse_half(float4 x) { return float4(half4(x)); }\n"
            "inline bool fse_nonfinite(uint bits)",
        ),
    ],
}


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda c: c.name)
@pytest.mark.parametrize("mistake", list(MISTAKES))
def test_the_bound_catches_a_kernel_that(mlx, monkeypatch, mistake, witness):
    _variant(monkeypatch, *MISTAKES[mistake])
    with pytest.raises(AssertionError, match="misses the bound"):
        assert_within_bound(_pool(mlx, witness), witness, name=witness.name)


@pytest.mark.parametrize("name", ["cancels to subnormal", "subnormal rows"])
def test_the_bound_catches_a_kernel_that_flushes_without_detecting(mlx, monkeypatch, name):
    _variant(
        monkeypatch,
        ("flag |= fse_step(f, x, before, acc);", ""),
        ("as_type<uint>(flagged ? NAN : f)", "as_type<uint>(f)"),
    )
    (case,) = [c for c in CASES if c.name == name and c.dtype == "float32"]
    with pytest.raises(AssertionError, match="misses the bound"):
        assert_within_bound(_pool(mlx, case), case, name=name)


def test_the_bound_catches_a_device_that_swaps_rows(mlx, monkeypatch):
    def swapped(launched, target):
        flags = real(launched, target)
        target[[0, 1]] = target[[1, 0]]
        return flags

    real = _kernels.read
    monkeypatch.setattr(_kernels, "read", swapped)
    witness = WITNESSES[0]
    with pytest.raises(AssertionError, match="row 0 misses the bound"):
        assert_within_bound(_pool(mlx, witness), witness, name=witness.name)


@pytest.mark.parametrize(("factor", "fails"), [(10.0, True), (0.5, False)])
def test_the_bound_catches_rows_10_times_beyond_it(mlx, factor, fails):
    witness = WITNESSES[1]
    out = _pool(mlx, witness)
    _, bound = bounds(witness)
    moved = (out.astype(np.float64) + factor * bound[:, None] / np.sqrt(out.shape[1])).astype(
        out.dtype
    )
    if fails:
        with pytest.raises(AssertionError, match="misses the bound"):
            assert_within_bound(moved, witness, name=witness.name)
    else:
        assert assert_within_bound(moved, witness, name=witness.name) < 1


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        ("fse_mul(inverse, acc, flag)", "acc / float(n)", "inverse length, not division"),
        # Not the case "fused, not rounded products": unfused, it cancels to 0.0 from a
        # rounded product, which the kernel flags.
        ("metal::fma(acc_t(f), x, acc)", "acc_t(f) * x + acc", "witness 8"),
    ],
    ids=["acc / n", "no fma"],
)
def test_the_bits_catch_what_the_bound_allows(mlx, monkeypatch, old, new, name):
    _variant(monkeypatch, (old, new))
    (case,) = [c for c in [*CASES, *WITNESSES] if c.name == name]
    out = _pool(mlx, case)
    assert_within_bound(out, case, name=name)
    with pytest.raises(AssertionError):
        assert_bits_equal(out, _reference.pool_weighted_mean(case.batch, *case.args()), name)
