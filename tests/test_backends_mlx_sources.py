"""fse._backends._mlx._sources: the Metal sources keep to the rules of the contract.

Read as text, so it runs on every machine; the kernel tests run them on the GPU.
"""

import re

import pytest

from fse._backends._mlx import _sources
from fse._contract import OOV_TOKEN

POOLS = {vec: _sources.pool(vec) for vec in (1, 4)}
CODE = _sources.HEADER + POOLS[4]


def test_contraction_is_off_and_fast_math_refused():
    assert _sources.HEADER.lstrip().startswith("#pragma METAL fp contract(off)")
    assert "#if defined(__FAST_MATH__)\n#error" in _sources.HEADER


def test_only_correctly_rounded_operations():
    """Explicit fma, precise divide and sqrt; no division operator, other sqrt, rsqrt or
    SIMD sums."""
    assert "metal::fma(" in CODE
    assert "metal::precise::divide(1.0f, float(n))" in CODE
    body = re.sub(r"//[^\n]*", "", CODE)
    assert not re.search(r"[^/]/[^/*]", body.replace("q / 32u", "").replace("DIM / VEC", ""))
    assert "metal::precise::sqrt(l[0])" in body
    body = body.replace("metal::precise::sqrt(", "")
    for banned in ("rsqrt", "sqrt(", "simd_sum", "simdgroup_matrix", "fast::"):
        assert banned not in body, banned


@pytest.mark.parametrize("vec", [1, 4])
def test_the_order_and_the_factors_of_the_cpu(vec):
    """One fma per word and per bucket in order, the factors as the cpu computes them,
    and the flags written per sentence."""
    pool = POOLS[vec]
    assert pool.count("acc = metal::fma(acc_t(f), x, acc);") == 2
    assert "INT8 ? weights[w] * scales[w] : weights[w]" in pool
    assert "metal::precise::divide(1.0f, float(b1 - b0)) * oov_weight[0]" in pool
    assert "INT8 ? shared * ngram_scales[g] : shared" in pool
    assert "n ? acc_t(inverse) * acc : acc_t(0.0f)" in pool
    assert f"constexpr uint VEC = {vec}u;" in pool
    assert "flags[(ulong)s * FLAG_WORDS + q / 32u] = any_flag;" in pool


def test_the_flag_bits_match_the_header():
    for name in ("NONFINITE", "LONG"):
        assert f"FSE_{name} = {getattr(_sources, name)}u;" in _sources.HEADER
    assert (_sources.NONFINITE, _sources.LONG) == (2, 4)


def test_the_bound_sums_every_contribution():
    """With BOUND, one fma of |f| and the row's norm per token and per bucket."""
    pool = POOLS[4]
    assert "magnitude = metal::fma(metal::fabs(f), norms[w], magnitude);" in pool
    assert "magnitude = metal::fma(metal::fabs(f), ngram_norms[g], magnitude);" in pool
    assert "bounds[(ulong)s * 2u] = magnitude;" in pool
    assert "bounds[(ulong)s * 2u + 1u] = l[0];" in pool


def test_the_unknown_token_is_the_contract_s():
    assert int(_sources.OOV_TOKEN_HEX.rstrip("u"), 16) == OOV_TOKEN


def test_the_inputs_are_those_the_source_reads():
    for name in (*_sources.POOL_INPUTS, *_sources.POOL_OUTPUTS):
        assert re.search(rf"\b{name}\b", POOLS[1]), name


def test_normalize_follows_dot16():
    """16 lanes by fma in index order, added 8, 4, 2 and 1 lanes apart, 1 / sqrt."""
    pool = POOLS[4]
    assert "for (uint i = q; i < DIM; i += 16u)" in pool
    assert "lane = metal::fma(values[ly * DIM + i], values[ly * DIM + i], lane);" in pool
    assert "for (uint width = 8u; width > 0u; width >>= 1u)" in pool
    assert "l[j] = l[j] + l[j + width];" in pool
    assert "metal::precise::divide(1.0f, metal::precise::sqrt(l[0]))" in pool
