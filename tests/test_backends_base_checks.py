"""The checks of the arguments of the backend methods, the same for every backend."""

import numpy as np
import pytest

F32 = np.float32


@pytest.mark.parametrize("name", ["cpu", "numpy"])
def test_score_input_is_checked_the_same_way(name):
    from fse._backends import get_backend

    backend = get_backend(name)
    rows = np.ones((3, 2), F32)
    with pytest.raises(ValueError, match="scales are given exactly for int8 rows"):
        backend.score(rows, np.ones(3, F32), np.ones(2, F32), 0.0)
    with pytest.raises(ValueError, match="scales are given exactly for int8 rows"):
        backend.score(np.ones((3, 2), np.int8), None, np.ones(2, F32), 0.0)
    with pytest.raises(ValueError, match="weights must hold 2 values"):
        backend.score(rows, None, np.ones(3, F32), 0.0)
    with pytest.raises(ValueError, match="matrix of float32"):
        backend.score(rows.astype(np.float64), None, np.ones(2, F32), 0.0)


@pytest.mark.parametrize("name", ["cpu", "numpy"])
@pytest.mark.parametrize("bias", [np.ones((3, 1), F32), [0.5], True, "0"])
def test_score_bias_must_be_a_real_number(name, bias):
    from fse._backends import get_backend

    with pytest.raises(TypeError, match="bias must be a real number"):
        get_backend(name).score(np.ones((3, 2), F32), None, np.ones(2, F32), bias)


@pytest.mark.parametrize("name", ["cpu", "numpy"])
def test_score_of_no_rows(name):
    from fse._backends import get_backend

    out = get_backend(name).score(np.ones((0, 2), F32), None, np.ones(2, F32), np.float32(0.5))
    assert out.shape == (0,)


@pytest.mark.parametrize("name", ["cpu", "numpy"])
@pytest.mark.parametrize(
    "weights", [np.array(["1", "2"]), np.array([True, False]), np.array([None, None])]
)
def test_score_weights_must_be_numbers(name, weights):
    from fse._backends import get_backend

    with pytest.raises(TypeError, match="weights must be numbers"):
        get_backend(name).score(np.ones((3, 2), F32), None, weights, 0.0)


def _overlap_cases():
    """Arrays over one buffer: views with an offset, strides (negative, zero, steps), 0-d,
    empty, broadcast and non-contiguous ones, and arrays of other buffers."""
    base = np.arange(240, dtype=F32).reshape(12, 20)
    other = np.arange(30, dtype=np.int64)
    return [
        base,
        base[3:7],
        base[:, 5:9],
        base[::2],
        base[::-1],
        base[:, ::-3],
        base[4, 2:5],
        base[2:3, 2:3],
        base[5:6, 5:6].reshape(()),  # 0-d view
        np.ones(()),  # 0-d of its own
        base[3:3],  # empty, with an offset
        np.empty((0, 4), F32),
        np.broadcast_to(base[0, :4], (6, 4)),  # stride 0
        base.T,
        base.reshape(-1)[7::13],
        base.view(np.uint8)[::5],
        base[1:5, 2:4].T,
        other,
        other[::-2],
        other[10:12],
    ]


def test_overlaps_is_the_bounds_test_of_may_share_memory():
    from fse._backends._base_checks import overlaps

    cases = _overlap_cases()
    for a in cases:
        for b in cases:
            assert overlaps(a, b) is bool(np.may_share_memory(a, b)), (a.shape, b.shape)


def test_the_native_overlaps_is_the_bounds_test_of_may_share_memory():
    from fse import _native

    cases = _overlap_cases()
    for a in cases:
        for b in cases:
            assert _native._test_overlaps(a, b) is bool(np.may_share_memory(a, b)), (
                a.shape,
                b.shape,
            )


def test_overlaps_makes_no_numpy_call_that_could_release_the_gil(monkeypatch):
    """Every check of a head task ran np.may_share_memory, which releases the GIL (DR-241)."""
    from fse._backends._base_checks import overlaps

    def forbidden(*_args, **_kwargs):
        raise AssertionError("np.may_share_memory called")

    monkeypatch.setattr(np, "may_share_memory", forbidden)
    assert overlaps(np.ones(4, F32), np.ones(4, F32)) is False
