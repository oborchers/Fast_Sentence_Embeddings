import numpy as np
import pytest

from fse._backends import Capabilities, PoolInput, UnsupportedByBackendError
from fse._backends._base import check_out, check_pool_input, dims
from fse._contract import Aggregation, Batch, Normalize, Op, Project

F32 = np.float32


def _pool(dtype=np.float32, n=4, dim=3, **kwargs):
    vectors = np.ones((n, dim), dtype=dtype)
    scales = np.ones(n, F32) if dtype == np.int8 else None
    return PoolInput(vectors, np.ones(n, F32), scales, **kwargs)


ALL = Capabilities(
    word_dtypes=frozenset({"float32", "float16", "int8"}),
    aggregations=frozenset({Aggregation.WEIGHTED_MEAN}),
    steps=frozenset(Op),
    fasttext=True,
    concurrent=True,
    max_dim=8,
)


@pytest.mark.parametrize(
    ("change", "pool", "steps", "match"),
    [
        ({"word_dtypes": frozenset({"float32"})}, _pool(np.int8), (), "word vectors of int8"),
        ({"aggregations": frozenset()}, _pool(), (), "the aggregation weighted_mean@1"),
        ({"fasttext": False}, _pool(ngram_vectors=np.ones((2, 3), F32)), (), "FastText"),
        ({"max_dim": 2}, _pool(), (), "rows of 3 values"),
        ({"steps": frozenset({Op.PROJECT})}, _pool(), (Normalize(),), "the step normalize@1"),
    ],
)
def test_capabilities_name_what_is_missing(change, pool, steps, match):
    caps = Capabilities(**{**ALL.__dict__, **change})
    with pytest.raises(UnsupportedByBackendError, match=f"the gpu backend cannot run {match}"):
        caps.check("gpu", pool, steps)
    ALL.check("gpu", pool, steps)


def test_unsupported_is_a_value_error():
    assert issubclass(UnsupportedByBackendError, ValueError)


@pytest.mark.parametrize(
    ("pool", "match"),
    [
        (PoolInput(np.ones((2, 3)), np.ones(2, F32)), "matrix of float32, float16 or int8"),
        (PoolInput(np.ones((3, 2), F32).T, np.ones(2, F32)), "vectors must be C-contiguous"),
        (PoolInput(np.ones((2, 0), F32), np.ones(2, F32)), "at least one dimension"),
        (PoolInput(np.ones((2, 3), F32), np.ones(3, F32)), "weights must be a float32 vector of 2"),
        (PoolInput(np.ones((2, 3), F32), np.ones(4, F32)[::2]), "weights must be C-contiguous"),
        (PoolInput(np.ones((2, 3), np.int8), np.ones(2, F32)), "scales are given exactly"),
        (PoolInput(np.ones((2, 3), F32), np.ones(2, F32), np.ones(2, F32)), "scales are given"),
        (_pool(ngram_scales=np.ones(2, F32)), "ngram_scales without ngram_vectors"),
        (_pool(ngram_vectors=np.ones((2, 4), F32)), "dtype and dimension of vectors"),
        (_pool(ngram_vectors=np.ones((2, 3), np.float16)), "dtype and dimension of vectors"),
        (_pool(np.int8, ngram_vectors=np.ones((2, 3), np.int8)), "ngram_scales are given"),
    ],
)
def test_pool_input_is_checked(pool, match):
    with pytest.raises(ValueError, match=match):
        check_pool_input(pool)


def test_pool_input_must_be_a_pool_input():
    with pytest.raises(TypeError, match="PoolInput"):
        check_pool_input((np.ones((2, 3), F32), np.ones(2, F32)))


def test_dims_follow_the_aggregation_and_the_steps():
    basis = np.ascontiguousarray(np.eye(2, 3, dtype=F32))
    steps, pooled, out = dims(_pool(), [Project(basis), Normalize()])
    assert (pooled, out) == (3, 2)
    assert isinstance(steps, tuple)


def test_out_is_checked():
    from fse._backends import Prepared

    pool = _pool()
    prepared = Prepared("numpy", pool, (), 3, 3, None)
    batch = Batch(0, np.array([0, 1, 2], np.int64), np.array([0, 1], np.uint32))
    check_out(np.empty((2, 3), F32), batch, prepared)
    for bad in (
        np.empty((2, 3), np.float64),
        np.empty((3, 2), F32).T,
        np.empty((1, 3), F32),
        [[0.0] * 3] * 2,
    ):
        with pytest.raises(ValueError, match="writable C-contiguous float32"):
            check_out(bad, batch, prepared)
    frozen = np.empty((2, 3), F32)
    frozen.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        check_out(frozen, batch, prepared)
    with pytest.raises(ValueError, match="vectors or weights"):
        check_out(pool.vectors[:2], batch, prepared)


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
def test_steps_may_be_any_iterable_and_are_used(name):
    from fse import _reference
    from fse._backends import get_backend

    backend = get_backend(name)
    pool = PoolInput(np.arange(6, dtype=F32).reshape(3, 2) + 1, np.ones(3, F32))
    prepared = backend.prepare(pool, iter([Normalize()]))
    assert prepared.steps == (Normalize(),) or [type(s) for s in prepared.steps] == [Normalize]
    batch = Batch(0, np.array([0, 1, 2, 3], np.int64), np.array([0, 1, 2], np.uint32))
    out = np.empty((3, 2), F32)
    backend.rows(batch, prepared, out)
    want = _reference.postprocess(
        _reference.pool_weighted_mean(batch, pool.vectors, pool.weights), [Normalize()]
    )
    assert out.tobytes() == want.tobytes()


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
