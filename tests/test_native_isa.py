"""Raw-register eligibility and exact import-time ISA selection, including emulated children."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from fse import _native

LEAF1 = sum(1 << bit for bit in (12, 27, 28, 29))
LEAF7 = sum(1 << bit for bit in (5, 16, 17, 30, 31))


@pytest.fixture
def detect():
    if _native._test_isa_detection(7, LEAF1, LEAF7, 0xE6) is None:
        assert not os.environ.get("FSE_REQUIRE_ISA"), "required detection hook missing"
        assert os.environ.get("FSE_TEST_TILES") != "1", "rebuild instrumented extension"
        pytest.skip("synthetic detection exists only in instrumented builds")
    return _native._test_isa_detection


@pytest.mark.parametrize("bit", [12, 27, 28, 29])
def test_each_shared_avx_prerequisite(detect, bit):
    assert detect(7, LEAF1 & ~(1 << bit), LEAF7, 0xE6) == 0


@pytest.mark.parametrize("bit", [5, 16, 17, 30, 31])
def test_each_leaf7_feature(detect, bit):
    assert detect(7, LEAF1, LEAF7 & ~(1 << bit), 0xE6) == (0 if bit == 5 else 1)


@pytest.mark.parametrize("bit", [1, 2, 5, 6, 7])
def test_each_os_state_component(detect, bit):
    assert detect(7, LEAF1, LEAF7, 0xE6 & ~(1 << bit)) == (0 if bit in (1, 2) else 1)


@pytest.mark.parametrize("leaf", [0, 1, 6])
def test_missing_leaf(detect, leaf):
    assert detect(leaf, LEAF1, LEAF7, 0xE6) == 0


@pytest.mark.parametrize("vendor", [0, 0x68747541, 0x756E6547])
@pytest.mark.parametrize("family", [6, 0x19, 0x1A, 0xFF])
def test_availability_does_not_enable_default_policy(detect, vendor, family):
    assert detect(7, LEAF1, LEAF7, 0xE6, vendor=vendor, family=family) == 3
    assert detect(7, LEAF1, LEAF7, 0xE6, compiled=0, vendor=vendor, family=family) == 1
    assert detect(8, LEAF1 | 1, LEAF7 | 1, 0xE7, leaf7_ecx=0xFFFFFFFF) == 3


def _child(choice):
    env = os.environ.copy()
    env.pop("FSE_ISA", None)
    if choice is not None:
        env["FSE_ISA"] = choice
    # Explicit wrapping instruments every exec, independently of Pin's fork policy.
    prefix = json.loads(env.get("FSE_TEST_PYTHON_PREFIX", "[]"))
    script = """
import json, os, numpy as np
from fse import _native as n
available = n.available_isas()
profile = os.environ.get('FSE_SDE_PROFILE')
if profile == 'spr':
    assert 'avx512' in available, available
elif profile == 'hsw':
    assert 'avx2' in available and 'avx512' not in available, available
explicit = n.WeightedMean(np.ones((1, 1), np.float32), np.ones(1, np.float32), isa='portable')
implicit = n.WeightedMean(np.ones((1, 1), np.float32), np.ones(1, np.float32))
result = [n.default_isa(), available, explicit.isa_name, implicit.isa_name]
os.environ['FSE_ISA'] = 'invalid after import'
assert n.default_isa() == result[0]
print(json.dumps(result))
"""
    return subprocess.run(
        [*prefix, sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("choice", [None, "portable", "avx2", "avx512", "neon"])
def test_environment_choice_and_explicit_precedence(choice):
    result = _child(choice)
    if choice is not None and choice not in _native.available_isas():
        assert result.returncode != 0
        assert "ValueError: FSE_ISA" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    default, available, explicit, implicit = json.loads(result.stdout)
    expected = choice or next(isa for isa in reversed(available) if isa != "avx512")
    assert default == implicit == expected
    assert tuple(available) == _native.available_isas()
    assert explicit == "portable"


@pytest.mark.parametrize("choice", ["", "AVX2", " avx2", "avx2 ", "avx512\n", "sse", "0"])
def test_invalid_environment_fails_at_extension_import(choice):
    result = _child(choice)
    assert result.returncode != 0
    assert "ValueError: FSE_ISA" in result.stderr


def test_required_isa_is_executed():
    required = os.environ.get("FSE_REQUIRE_ISA")
    if required is None:
        pytest.skip("no required ISA in this run")
    assert required in _native.available_isas()
    assert _native._test_pool_trace() is not None
    kernel = _native.WeightedMean(
        np.ones((1, 255), np.float32), np.ones(1, np.float32), isa=required
    )
    from fse._contract import Batch

    kernel.pool(Batch(0, np.array([0, 1], np.int64), np.array([0], np.uint32)))
    assert _native._test_pool_trace() == (f"{required}-tiles", 1)
    assert _native._avx512_layout() == int(os.environ.get("FSE_EXPECT_AVX512_LAYOUT", "0"))


def test_avx512_selection_reaches_every_vector_dispatch_site():
    if "avx512" not in _native.available_isas():
        pytest.skip("AVX-512 unavailable on this host")
    if _native._test_route_trace(reset=True) is None:
        assert not os.environ.get("FSE_REQUIRE_ISA"), "required dispatch counters missing"
        pytest.skip("production build has no dispatch counters")
    from fse._contract import Batch, Link

    rows = np.ones((1, 255), np.float32)
    batch = Batch(0, np.array([0, 1], np.int64), np.array([0], np.uint32))
    for dtype in (np.float32, np.float16, np.int8):
        scales = np.ones(1, np.float32) if dtype == np.int8 else None
        _native.WeightedMean(rows.astype(dtype), np.ones(1, np.float32), scales, isa="avx512").pool(
            batch
        )
    _native.dot16(rows, rows, isa="avx512")
    _native.store_rows(rows, np.empty_like(rows, dtype=np.float16), isa="avx512")
    _native.store_rows(
        rows, np.empty_like(rows, dtype=np.int8), np.empty(1, np.float32), isa="avx512"
    )
    logits = np.ones((1, 17), np.float32)
    _native.exp1(logits, isa="avx512")
    for link in (Link.SIGMOID, Link.SOFTMAX):
        _native.head_outputs(logits, "proba", link, np.empty_like(logits), isa="avx512")
    assert _native._test_route_trace() == 1023  # all ten vector dispatch sites, no AVX2 f32 loop


def test_cpu_registers_are_read_once_for_all_dispatch_paths():
    reads = _native._test_cpu_reads()
    if reads is None:
        assert not os.environ.get("FSE_REQUIRE_ISA"), "required CPU read counter missing"
        pytest.skip("production build has no read counter")
    assert reads == 1
    from fse._backends._cpu import CpuBackend
    from fse._contract import Batch

    rows = np.ones((1, 17), np.float32)
    weights = np.ones(1, np.float32)
    batch = Batch(0, np.array([0, 1], np.int64), np.array([0], np.uint32))
    available = _native.available_isas()
    for _ in range(3):
        assert _native.available_isas() is available
        _native.default_isa()
        for isa in (None, *available):
            _native.WeightedMean(rows, weights, isa=isa).pool(batch)
            _native.dot16(rows, rows, isa=isa)
            CpuBackend(isa=isa).prefilter_variant()
    assert _native._test_cpu_reads() == 1
