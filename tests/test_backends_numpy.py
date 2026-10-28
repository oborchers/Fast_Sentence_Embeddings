import numpy as np
import pytest

from fse import _reference
from fse._backends import NumpyBackend, PoolInput, Prepared, UnsupportedByBackendError
from fse._contract import Batch, Normalize

F32 = np.float32


def _batch():
    return Batch(5, np.array([0, 2, 2, 3], np.int64), np.array([0, 1, 2], np.uint32))


def test_rows_are_the_reference_pool_and_steps():
    rng = np.random.default_rng(0)
    pool = PoolInput(rng.standard_normal((3, 4)).astype(F32), rng.random(3).astype(F32))
    backend = NumpyBackend()
    prepared = backend.prepare(pool, [Normalize()])
    assert (prepared.pooled_dim, prepared.output_dim, prepared.steps) == (4, 4, (Normalize(),))
    out = np.empty((3, 4), F32)
    backend.rows(_batch(), prepared, out)
    want = _reference.postprocess(
        _reference.pool_weighted_mean(_batch(), pool.vectors, pool.weights), [Normalize()], 5
    )
    assert out.tobytes() == want.tobytes()


def test_errors_name_the_rows_of_the_batch():
    pool = PoolInput(np.array([[1, 0], [0, 0], [np.inf, 0]], F32), np.ones(3, F32))
    backend = NumpyBackend()
    with pytest.raises(ValueError, match="row 7 contains NaN"):
        backend.rows(_batch(), backend.prepare(pool, ()), np.empty((3, 2), F32))
    tiny = PoolInput(np.array([[1, 0], [1e-30, 0], [1, 0]], F32), np.ones(3, F32))
    batch = Batch(5, np.array([0, 1, 2, 3], np.int64), np.array([0, 1, 2], np.uint32))
    with pytest.raises(ValueError, match="row 6 cannot be normalized"):
        backend.rows(batch, backend.prepare(tiny, [Normalize()]), np.empty((3, 2), F32))


def test_prepared_of_another_backend_is_refused():
    pool = PoolInput(np.ones((3, 2), F32), np.ones(3, F32))
    other = Prepared("cpu", pool, (), 2, 2, None)
    with pytest.raises(TypeError, match="prepared by the numpy backend"):
        NumpyBackend().rows(_batch(), other, np.empty((3, 2), F32))


def test_prepare_checks_input_and_capabilities():
    with pytest.raises(ValueError, match="weights"):
        NumpyBackend().prepare(PoolInput(np.ones((3, 2), F32), np.ones(2, F32)), ())
    caps = NumpyBackend().capabilities()
    assert caps.concurrent
    assert caps.fasttext
    assert {"float32", "float16", "int8"} == caps.word_dtypes
    assert issubclass(UnsupportedByBackendError, ValueError)


def test_gram_block_and_score():
    rows = np.arange(12, dtype=F32).reshape(4, 3)
    backend = NumpyBackend()
    assert backend.gram_block(rows).tobytes() == (rows.T @ rows).tobytes()
    with pytest.raises(ValueError, match="float32 matrix"):
        backend.gram_block(rows.astype(np.float64))
    weights = np.ones(3, F32)
    assert (
        backend.score(rows, None, weights, 0.5).tobytes()
        == _reference.score(rows, None, weights, 0.5).tobytes()
    )
