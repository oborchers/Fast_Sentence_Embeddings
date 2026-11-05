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
