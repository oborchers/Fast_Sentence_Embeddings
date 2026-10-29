"""fse._backends._mlx._sources: the Metal sources keep to the rules of the contract.

Read as text, so it runs on every machine; the kernel tests run them on the GPU.
"""

import re

import pytest

from fse._backends._mlx import _sources
from fse._contract import OOV_TOKEN

POOLS = {vec: _sources.pool(vec) for vec in (1, 4)}
CODE = _sources.HEADER + _sources.PREPARE + POOLS[4]


def test_contraction_is_off_and_fast_math_refused():
    assert _sources.HEADER.lstrip().startswith("#pragma METAL fp contract(off)")
    assert "#if defined(__FAST_MATH__)\n#error" in _sources.HEADER


def test_only_correctly_rounded_operations():
    """Explicit fma, precise divide; no division operator, sqrt, rsqrt or SIMD sums."""
    assert "metal::fma(" in CODE
    assert "metal::precise::divide(1.0f, float(n))" in CODE
    body = re.sub(r"//[^\n]*", "", CODE)
    assert not re.search(r"[^/]/[^/*]", body.replace("q / 32u", "").replace("DIM / VEC", ""))
    for banned in ("rsqrt", "sqrt(", "simd_sum", "simdgroup_matrix", "fast::"):
        assert banned not in body, banned


@pytest.mark.parametrize("vec", [1, 4])
def test_every_fma_is_checked_and_every_product_flagged(vec):
    """Each accumulation is followed by fse_step; the products of factors go through
    fse_mul; the word factors come from the prepare kernel."""
    pool = POOLS[vec]
    assert pool.count("acc = metal::fma(acc_t(f), x, acc);") == 2
    assert pool.count("flag |= fse_step(f, x, before, acc);") == 2
    assert "fse_mul(shared, ngram_scales[g], flag)" in pool
    assert "fse_mul(inverse, oov_weight[0], flag)" in pool
    assert "fse_mul(inverse, acc, flag)" in pool
    assert "as_type<float>(factors[w])" in pool
    assert "fse_small_factor(f, ngram_low[g])" in pool
    assert f"constexpr uint VEC = {vec}u;" in pool


def test_the_prepare_kernel_poisons_flagged_words():
    assert "factors[w] = as_type<uint>(flagged ? NAN : f);" in _sources.PREPARE
    assert "fse_mul(weight, scale, flag)" in _sources.PREPARE


def test_the_flag_bits_match_the_header():
    names = ("SUBNORMAL", "NONFINITE", "LONG", "HUGE", "TINY")
    for name in names:
        assert f"FSE_{name} = {getattr(_sources, name)}u;" in _sources.HEADER
    assert sorted(getattr(_sources, name) for name in names) == [1, 2, 4, 8, 16]
    assert f"FSE_NO_EXPONENT = {_sources.NO_EXPONENT}u;" in _sources.HEADER


def test_the_unknown_token_is_the_contract_s():
    assert int(_sources.OOV_TOKEN_HEX.rstrip("u"), 16) == OOV_TOKEN


def test_the_inputs_are_those_the_sources_read():
    for name in (*_sources.POOL_INPUTS, *_sources.POOL_OUTPUTS):
        assert re.search(rf"\b{name}\b", POOLS[1]), name
    for name in (*_sources.PREPARE_INPUTS, *_sources.PREPARE_OUTPUTS):
        assert re.search(rf"\b{name}\[", _sources.PREPARE), name
