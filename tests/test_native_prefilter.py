"""The routines of fse._native for the int8 prefilter: variants, checks and what they return.

Bit identity with the reference is the conformance suite (tests/test_conformance_prefilter.py).
"""

import numpy as np
import pytest

from fse import _native
from support.prefilter_native import block, pass_of, runnable_variants

F32 = np.float32


def test_variants_are_built_and_runnable_as_separate_facts():
    built = _native.prefilter_variants()
    assert built[0] == "portable"
    assert set(built) <= {"portable", "avx2", "neon-sdot"}
    assert _native.prefilter_can_run("portable")
    for name in ("avx2", "neon-sdot"):
        if name not in built:
            assert not _native.prefilter_can_run(name)
    assert not _native.prefilter_can_run("vnni")
    assert not _native.prefilter_can_run(None)
    assert set(runnable_variants()) <= set(built)


def test_the_default_variant_is_the_fastest_the_cpu_runs():
    b = block(5, 32, 1, seed=1)
    default = _native.int8_pass(*b.args(), 2)
    explicit = _native.int8_pass(*b.args(), 2, variant=runnable_variants()[-1])
    assert default[2].tobytes() == explicit[2].tobytes()


def test_an_unknown_or_unavailable_variant_is_refused():
    b = block(5, 32, 1, seed=1)
    with pytest.raises(ValueError, match="variant must be one of"):
        _native.int8_pass(*b.args(), 2, variant="vnni")
    with pytest.raises(ValueError, match="variant must be one of"):
        _native.int8_pass(*b.args(), 2, variant=3)
    for name in ("avx2", "neon-sdot"):
        if name not in _native.prefilter_variants():
            with pytest.raises(ValueError, match=f"this build has no {name} prefilter"):
                _native.int8_pass(*b.args(), 2, variant=name)
        elif not _native.prefilter_can_run(name):
            with pytest.raises(ValueError, match=f"this CPU cannot run the {name} prefilter"):
                _native.int8_pass(*b.args(), 2, variant=name)


def test_the_kernel_checks_its_arrays_itself():
    b = block(5, 32, 2, seed=1)
    stored, inverse, qcodes, qscales, slack, threshold = b.args()
    with pytest.raises(ValueError, match="qcodes must be a nonempty C-contiguous int8 matrix"):
        _native.int8_bounds(b.units, qcodes[:0], qscales[:0])
    with pytest.raises(ValueError, match="at most 1024 dimensions"):
        _native.int8_bounds(np.ones((1, 1025), F32), np.ones((1, 1025), np.int8), np.ones(1, F32))
    with pytest.raises(ValueError, match="units must be a C-contiguous float32 array"):
        _native.int8_bounds(b.units.astype(np.float64), qcodes, qscales)
    with pytest.raises(ValueError, match="qscales must be a C-contiguous float32 array"):
        _native.int8_bounds(b.units, qcodes, qscales[:1])
    with pytest.raises(ValueError, match="stored must be a C-contiguous int8 matrix"):
        _native.int8_pass(stored[:, :31].copy(), inverse, qcodes, qscales, slack, threshold, 2)
    with pytest.raises(ValueError, match="stored must be a C-contiguous int8 matrix"):
        _native.int8_pass(stored.T.copy().T, inverse, qcodes, qscales, slack, threshold, 2)
    with pytest.raises(ValueError, match="inverse must be a C-contiguous float32 array"):
        _native.int8_pass(stored, inverse[:4], qcodes, qscales, slack, threshold, 2)
    with pytest.raises(ValueError, match="slack must be a C-contiguous float64 array"):
        _native.int8_pass(stored, inverse, qcodes, qscales, slack[:1], threshold, 2)
    with pytest.raises(ValueError, match="threshold must be a C-contiguous float64 array"):
        _native.int8_pass(stored, inverse, qcodes, qscales, slack, threshold.astype(F32), 2)
    for bad in (0, True, 2**31, 1.0):
        with pytest.raises(ValueError, match="k must be an integer from 1"):
            _native.int8_pass(*b.args(), bad)
        with pytest.raises(ValueError, match="capacity must be an integer from 1"):
            _native.int8_pass(*b.args(), 2, bad)


def test_what_the_pass_returns_does_not_keep_the_big_buffers_alive():
    b = block(40, 16, 2, seed=2)
    status, row, heap, rows, upper = _native.int8_pass(*b.args(), 2)
    assert (status, row) == ("ok", None)
    assert heap.shape == (2, 2)
    assert all(a.base is None for a in (*rows, *upper))
    assert all(a.flags.owndata for a in (*rows, *upper))


@pytest.mark.parametrize("variant", runnable_variants())
def test_the_dimension_limit_of_the_proof_is_1024(variant):
    top = block(5, 1024, 1, seed=3)
    assert pass_of(variant)(*top.args(), 1).status == "ok"
    wide = (
        np.ones((5, 1025), np.int8),
        np.ones(5, F32),
        np.ones((1, 1025), np.int8),
        np.ones(1, F32),
    )
    with pytest.raises(ValueError, match="at most 1024 dimensions"):
        pass_of(variant)(*wide, np.ones(1), np.full(1, -np.inf), 1)


def test_the_kernel_checks_the_content_of_its_arrays_too():
    b = block(5, 32, 2, seed=1)
    stored, inverse, qcodes, qscales, slack, threshold = b.args()
    with pytest.raises(ValueError, match=r"\[-127, 127\]"):
        _native.int8_bounds(b.units, np.full_like(qcodes, -128), qscales)
    for bad in (np.nan, np.inf, -1.0):
        scales = np.array([1.0, bad], F32)
        with pytest.raises(ValueError, match="finite and not negative"):
            _native.int8_bounds(b.units, qcodes, scales)
        with pytest.raises(ValueError, match="finite and not negative"):
            _native.int8_pass(stored, inverse, qcodes, scales, slack, threshold, 2)
    for bad in (np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="slack of every query"):
            _native.int8_pass(stored, inverse, qcodes, qscales, np.array([0.5, bad]), threshold, 2)
    with pytest.raises(ValueError, match="must not be NaN"):
        _native.int8_pass(stored, inverse, qcodes, qscales, slack, np.array([0.0, np.nan]), 2)
    n = 2**20 + 1
    many = (np.ones((n, 1), F32), np.ones((n, 1), np.int8), np.ones(n, F32))
    with pytest.raises(ValueError, match="at most 1048576 queries"):
        _native.int8_bounds(*many)
    with pytest.raises(ValueError, match="capacity must be an integer from 1"):
        _native.int8_pass(stored, inverse, qcodes, qscales, slack, threshold, 2**29)
