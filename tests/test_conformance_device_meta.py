"""Meta-tests of the conformance suite of a device backend: the comparison under the bound
(support.bound) fails for each mistake the tolerance contract names, on a witness case
that the mistaken kernel computes on the GPU (a launch, no host fallback).

The mistakes are variants of the real kernel's source (built apart from it), a backend
that does not refuse subnormal inputs at prepare, a read that swaps rows, and rows moved
by 10 times the bound; and the flushed cases need the absolute term of the bound. Two more show that the bit comparison
of tests/test_conformance.py catches what the bound allows (no fma, acc / n).
"""

import numpy as np
import pytest

from fse import _reference
from fse._backends._mlx import _bounds, _kernels, _sources
from fse._contract import Normalize, Project
from support import bound
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
        for name in ("HEADER", "_POOL"):
            text = getattr(_sources, name)
            if old in text:
                monkeypatch.setattr(_sources, name, text.replace(old, new))
                break
        else:
            raise AssertionError(f"not in the sources: {old!r}")
    monkeypatch.setattr(_kernels, "_KERNELS", {})


# Witnesses: distinct tokens of random words, no empty sentence, both kernel widths.
WITNESSES = [random_case(f"witness {dim}", "float32", 70 + dim, dim=dim, empty=0) for dim in (8, 9)]
# The fma of a known word (the ngram one is indented further), and its replacements.
KNOWN_FMA = "acc = metal::fma(acc_t(f), x, acc);\n            if (BOUND) {\n                magnitude = metal::fma(metal::fabs(f), norms[w]"
MISTAKES = {
    "drops a token": [
        ("for (uint t = begin; t < end; ++t)", "for (uint t = begin + 1u; t < end; ++t)")
    ],
    "repeats every token": [
        (KNOWN_FMA, KNOWN_FMA.replace("acc);", "metal::fma(acc_t(f), x, acc));", 1))
    ],
    "divides by the wrong count": [("divide(1.0f, float(n))", "divide(1.0f, float(n + 1u))")],
    "skips 1 / n": [("acc_t(inverse) * acc", "acc")],
    "accumulates in float16": [
        (
            KNOWN_FMA,
            KNOWN_FMA.replace(
                "acc = metal::fma(acc_t(f), x, acc);",
                "acc = fse_half(metal::fma(acc_t(f), x, acc));",
            ),
        ),
        (
            "inline bool fse_nonfinite(float x)",
            "inline float fse_half(float x) { return float(half(x)); }\n"
            "inline float4 fse_half(float4 x) { return float4(half4(x)); }\n"
            "inline bool fse_nonfinite(float x)",
        ),
    ],
}


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda c: c.name)
@pytest.mark.parametrize("mistake", list(MISTAKES))
def test_the_bound_catches_a_kernel_that(mlx, monkeypatch, mistake, witness):
    _variant(monkeypatch, *MISTAKES[mistake])
    with pytest.raises(AssertionError, match="misses the bound"):
        assert_within_bound(_pool(mlx, witness), witness, name=witness.name)


@pytest.mark.parametrize(
    ("dtype", "name"),
    [
        ("float32", "subnormal weight, large row"),
        ("int8", "subnormal scale, large weight"),
        ("int8", "subnormal int8 factor"),
        ("float32", "subnormal ngram factor"),
    ],
)
def test_the_bound_catches_a_backend_that_takes_subnormal_inputs(mlx, monkeypatch, dtype, name):
    """Without the check at prepare, the GPU flushes a subnormal factor of normal products."""
    monkeypatch.setattr(_kernels, "subnormal_input", lambda *_: False)
    (case,) = [c for c in CASES if c.name == name and c.dtype == dtype]
    with pytest.raises(AssertionError, match="misses the bound"):
        assert_within_bound(_pool(mlx, case), case, name=name)


def test_the_flushed_cases_need_the_absolute_term(mlx, monkeypatch):
    """The flush of a sum that cancels into the subnormal range is beyond TAU m alone."""
    (case,) = [c for c in CASES if c.name == "cancels to subnormal"]
    out = _pool(mlx, case)
    assert_within_bound(out, case, name=case.name)
    monkeypatch.setattr(bound, "DELTA", 0.0)
    with pytest.raises(AssertionError, match="misses the bound"):
        assert_within_bound(out, case, name=case.name)


def test_a_flush_that_turns_a_rounding_needs_the_rounding_term(mlx):
    """The witness of the review of PR #121: on the GPU within the bound, beyond the
    bound without ``2 gamma_K``."""
    (case,) = [c for c in CASES if c.name == "a flush turns a rounding"]
    out = _pool(mlx, case)
    assert_within_bound(out, case, name=case.name)
    with pytest.raises(AssertionError, match="misses the bound"):
        assert_within_bound(out, case, name=case.name, rounding=False)


def test_the_bound_catches_a_device_that_swaps_rows(mlx, monkeypatch):
    def swapped(launched):
        rows, flags, bounds = real(launched)
        return rows[[1, 0, *range(2, len(rows))]], flags, bounds

    real = _kernels.view
    monkeypatch.setattr(_kernels, "view", swapped)
    witness = WITNESSES[0]
    with pytest.raises(AssertionError, match="row 0 misses the bound"):
        assert_within_bound(_pool(mlx, witness), witness, name=witness.name)


@pytest.mark.parametrize(("factor", "fails"), [(10.0, True), (0.5, False)])
def test_the_bound_catches_rows_10_times_beyond_it(mlx, factor, fails):
    witness = WITNESSES[1]
    out = _pool(mlx, witness)
    _, bound, _ = bounds(witness)
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
        ("acc_t(inverse) * acc", "acc / float(n)", "inverse length, not division"),
        ("metal::fma(acc_t(f), x, acc)", "acc_t(f) * x + acc", "fused, not rounded products"),
    ],
    ids=["acc / n", "no fma"],
)
def test_the_bits_catch_what_the_bound_allows(mlx, monkeypatch, old, new, name):
    _variant(monkeypatch, (old, new))
    (case,) = [c for c in CASES if c.name == name]
    out = _pool(mlx, case)
    assert_within_bound(out, case, name=name)
    with pytest.raises(AssertionError):
        assert_bits_equal(out, _reference.pool_weighted_mean(case.batch, *case.args()), name)


@pytest.mark.parametrize("steps", ["normalize", "project normalize"])
def test_the_suite_catches_a_backend_without_the_check_before_normalize(mlx, monkeypatch, steps):
    """Without it, the device row of "zero after a flush" is normalized to about -1
    where the cpu row stays zero (or, after an identity projection, is normalized on the
    host from the device row)."""
    monkeypatch.setattr(_bounds, "unnormalizable", lambda low, *_: np.zeros(len(low), bool))
    (case,) = [c for c in CASES if c.name == "zero after a flush"]
    chosen = (
        (Normalize(),)
        if steps == "normalize"
        else (Project(np.eye(1, dtype=np.float32)), Normalize())
    )
    out, _, _ = pool_on_device(mlx, case, chosen)
    with pytest.raises(AssertionError, match="must be the cpu's"):
        assert_within_bound(out, case, chosen, case.name)
