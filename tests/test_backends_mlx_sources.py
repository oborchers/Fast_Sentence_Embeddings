"""fse._backends._mlx._sources: the Metal source keeps to the rules of the contract.

Read as text, so it runs on every machine; the kernel tests run it on the GPU.
"""

import re

from fse._backends._mlx import _sources
from fse._contract import OOV_TOKEN

CODE = _sources.HEADER + _sources.POOL


def test_contraction_is_off_and_fast_math_refused():
    assert _sources.HEADER.lstrip().startswith("#pragma METAL fp contract(off)")
    assert "#if defined(__FAST_MATH__)\n#error" in _sources.HEADER


def test_only_correctly_rounded_operations():
    """Explicit fma, precise divide; no division operator, sqrt, rsqrt or SIMD sums."""
    assert "metal::fma(" in CODE
    assert "metal::precise::divide(1.0f, float(n))" in CODE
    body = re.sub(r"//[^\n]*", "", CODE)
    assert not re.search(r"[^/]/[^/*]", body.replace("d / 32u", "")), "a division operator"
    for banned in ("rsqrt", "sqrt(", "simd_sum", "simdgroup_matrix", "fast::"):
        assert banned not in body, banned


def test_every_multiply_and_fma_of_values_is_checked():
    """Values meet only in fse_mul and fse_fma; the plain products are those wrappers'."""
    pool = _sources.POOL
    assert "fse_fma(f, float(vectors[" in pool
    assert "fse_fma(f, float(ngram_vectors[" in pool
    assert "fse_mul(weights[w], scales[w], flag)" in pool
    assert "fse_mul(shared, ngram_scales[q], flag)" in pool
    assert "fse_mul(inverse, acc, flag)" in pool
    assert "fse_mul(inverse, oov_weight[0], flag)" in pool


def test_the_flag_bits_match_the_header():
    for name in ("SUBNORMAL", "NONFINITE", "LONG"):
        assert f"FSE_{name} = {getattr(_sources, name)}u;" in _sources.HEADER
    bits = [_sources.SUBNORMAL, _sources.NONFINITE, _sources.LONG]
    assert sorted(bits) == [1, 2, 4]


def test_the_unknown_token_is_the_contract_s():
    assert int(_sources.OOV_TOKEN_HEX.rstrip("u"), 16) == OOV_TOKEN


def test_the_inputs_are_those_the_source_reads():
    for name in (*_sources.POOL_INPUTS, *_sources.POOL_OUTPUTS):
        assert re.search(rf"\b{name}\[", _sources.POOL), name
