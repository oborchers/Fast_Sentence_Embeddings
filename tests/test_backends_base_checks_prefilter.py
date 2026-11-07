"""The checks of the arguments of ``int8_bounds`` and ``int8_pass``, the same for every backend."""

import numpy as np
import pytest

from fse._backends import get_backend
from support.prefilter_native import block

F32 = np.float32
BACKENDS = ["cpu", "numpy"]


def blk():
    return block(6, 16, 2, seed=1)


@pytest.mark.parametrize("name", BACKENDS)
def test_bounds_of_a_block_work_and_are_bool_and_float64(name):
    b = blk()
    slack, ok = get_backend(name).int8_bounds(b.units, b.qcodes, b.qscales)
    assert slack.dtype == np.float64
    assert ok.dtype == np.bool_
    assert ok.all()


@pytest.mark.parametrize("name", BACKENDS)
def test_bounds_check_their_arguments(name):
    backend, b = get_backend(name), blk()
    call = backend.int8_bounds
    with pytest.raises(ValueError, match="nonempty C-contiguous int8 matrix"):
        call(b.units, b.qcodes.astype(np.int16), b.qscales)
    with pytest.raises(ValueError, match="nonempty C-contiguous int8 matrix"):
        call(b.units[:0], b.qcodes[:0], b.qscales[:0])
    with pytest.raises(ValueError, match="nonempty C-contiguous int8 matrix"):
        call(b.units, np.asfortranarray(b.qcodes), b.qscales)
    with pytest.raises(ValueError, match="at most 1024 dimensions"):
        call(np.ones((1, 1025), F32), np.ones((1, 1025), np.int8), np.ones(1, F32))
    with pytest.raises(ValueError, match=r"\[-127, 127\]"):
        call(b.units, np.full_like(b.qcodes, -128), b.qscales)
    with pytest.raises(ValueError, match="qscales must be a C-contiguous float32 vector of 2"):
        call(b.units, b.qcodes, b.qscales.astype(np.float64))
    with pytest.raises(ValueError, match="qscales must be a C-contiguous float32 vector of 2"):
        call(b.units, b.qcodes, b.qscales[:1])
    with pytest.raises(ValueError, match="finite and not negative"):
        call(b.units, b.qcodes, np.array([1.0, -1.0], F32))
    with pytest.raises(ValueError, match="finite and not negative"):
        call(b.units, b.qcodes, np.array([1.0, np.inf], F32))
    with pytest.raises(ValueError, match="units must be a C-contiguous float32 matrix of shape"):
        call(b.units.astype(np.float64), b.qcodes, b.qscales)
    with pytest.raises(ValueError, match="units must be a C-contiguous float32 matrix of shape"):
        call(b.units[:1], b.qcodes, b.qscales)
    with pytest.raises(ValueError, match="qcodes must be"):
        call(b.units, "codes", b.qscales)


@pytest.mark.parametrize("name", BACKENDS)
def test_the_pass_checks_its_arguments(name):
    backend, b = get_backend(name), blk()
    stored, inverse, qcodes, qscales, slack, threshold = b.args()

    def call(**changes):
        args = dict(
            stored=stored,
            inverse=inverse,
            qcodes=qcodes,
            qscales=qscales,
            slack=slack,
            threshold=threshold,
            k=2,
            capacity=None,
        )
        return backend.int8_pass(*{**args, **changes}.values())

    assert call().status == "ok"
    assert call(capacity=3).status == "ok"
    with pytest.raises(ValueError, match="stored must be a C-contiguous int8 matrix"):
        call(stored=stored.astype(np.int16))
    with pytest.raises(ValueError, match="stored must be a C-contiguous int8 matrix"):
        call(stored=np.ones((6, 15), np.int8))
    with pytest.raises(ValueError, match="stored must be a C-contiguous int8 matrix"):
        call(stored=np.asfortranarray(stored))
    with pytest.raises(ValueError, match="inverse must be a C-contiguous float32 vector of 6"):
        call(inverse=inverse[:5])
    with pytest.raises(ValueError, match="slack must be a C-contiguous float64 vector of 2"):
        call(slack=slack.astype(F32))
    with pytest.raises(ValueError, match="finite and not negative"):
        call(slack=np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="finite and not negative"):
        call(slack=np.array([np.inf, 1.0]))
    with pytest.raises(ValueError, match="threshold must be a C-contiguous float64 vector of 2"):
        call(threshold=np.zeros(3))
    with pytest.raises(ValueError, match="must not be NaN"):
        call(threshold=np.array([np.nan, 0.0]))
    for bad in (0, -1, True, 1.5, "2", 2**31):
        with pytest.raises(ValueError, match="k must be an integer from 1"):
            call(k=bad)
        with pytest.raises(ValueError, match="capacity must be an integer from 1"):
            call(capacity=bad)
    assert call(k=np.int64(2)).status == "ok"


@pytest.mark.parametrize("name", BACKENDS)
def test_a_block_of_no_rows_keeps_nothing_and_a_heap_of_minus_infinity(name):
    b = block(0, 16, 2, seed=1)
    got = get_backend(name).int8_pass(*b.args(), 3)
    assert got.status == "ok"
    assert np.isneginf(got.heap).all()
    assert [len(rows) for rows in got.rows] == [0, 0]
