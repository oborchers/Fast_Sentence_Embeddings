import numpy as np
import pytest

from fse import _native, _reference
from fse._backends import CpuBackend, PoolInput, Prepared, get_backend
from fse._contract import Batch, Normalize, Project

F32 = np.float32


def _batch():
    return Batch(5, np.array([0, 2, 2, 3], np.int64), np.array([0, 1, 2], np.uint32))


def _pool(dtype="float32"):
    rng = np.random.default_rng(0)
    rows, scales = _reference.quantize_rows(rng.standard_normal((3, 4)).astype(F32), dtype)
    return PoolInput(rows, rng.random(3).astype(F32), scales)


@pytest.mark.parametrize("isa", _native.available_isas())
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_rows_are_the_reference(isa, dtype):
    pool = _pool(dtype)
    backend = get_backend("cpu", isa=isa)
    basis = np.ascontiguousarray(np.eye(2, 4, dtype=F32))
    for steps in ((), (Normalize(),), (Project(basis), Normalize())):
        prepared = backend.prepare(pool, steps)
        out = np.empty((3, prepared.output_dim), F32)
        backend.rows(_batch(), prepared, out)
        want = _reference.pool_weighted_mean(_batch(), pool.vectors, pool.weights, pool.scales)
        want = _reference.postprocess(want, steps, 5) if steps else want
        assert out.tobytes() == want.tobytes(), (isa, dtype, steps)


def test_prepare_builds_the_kernel_once():
    prepared = CpuBackend().prepare(_pool(), ())
    kernel = prepared.state.kernel
    for _ in range(3):
        CpuBackend().rows(_batch(), prepared, np.empty((3, 4), F32))
    assert prepared.state.kernel is kernel
    assert prepared.state.isa == _native.default_isa()


def test_out_and_prepared_are_checked():
    backend = CpuBackend()
    prepared = backend.prepare(_pool(), ())
    with pytest.raises(ValueError, match="shape"):
        backend.rows(_batch(), prepared, np.empty((3, 5), F32))
    with pytest.raises(ValueError, match="vectors or weights"):
        backend.rows(_batch(), prepared, prepared.pool.vectors[:3])
    other = Prepared("numpy", prepared.pool, (), 4, 4, None)
    with pytest.raises(TypeError, match="prepared by the cpu backend"):
        backend.rows(_batch(), other, np.empty((3, 4), F32))


def test_errors_name_the_rows_of_the_batch():
    pool = PoolInput(np.array([[1, 0], [0, 0], [np.inf, 0]], F32), np.ones(3, F32))
    backend = CpuBackend()
    with pytest.raises(ValueError, match="row 7 contains NaN"):
        backend.rows(_batch(), backend.prepare(pool, ()), np.empty((3, 2), F32))
    with pytest.raises(ValueError, match="row 7 contains NaN"):
        backend.rows(_batch(), backend.prepare(pool, [Normalize()]), np.empty((3, 2), F32))


def test_isa_selection():
    with pytest.raises(ValueError, match="isa must be one of the kernels"):
        CpuBackend("sse9")
    for isa in _native.available_isas():
        assert CpuBackend(isa).isa == isa


def test_gram_block_and_score():
    rows = np.arange(12, dtype=F32).reshape(4, 3)
    backend = CpuBackend()
    assert backend.gram_block(rows).tobytes() == (rows.T @ rows).tobytes()
    with pytest.raises(ValueError, match="float32 matrix"):
        backend.gram_block(rows.astype(np.float64))
    weights = np.ones(3, F32)
    assert (
        backend.score(rows, None, weights, 0.5).tobytes()
        == _reference.score(rows, None, weights, 0.5).tobytes()
    )


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_a_square_project_is_applied(dtype):
    # A project of as many dimensions as it reads keeps the dimension, and still makes a
    # new array: it must reach out (a permutation shows whether it did).
    pool = _pool(dtype)
    backend = get_backend("cpu")
    basis = np.ascontiguousarray(np.eye(4, dtype=F32)[[2, 0, 3, 1]])
    for steps in ((Project(basis),), (Project(basis), Normalize())):
        prepared = backend.prepare(pool, steps)
        out = np.empty((3, 4), F32)
        backend.rows(_batch(), prepared, out)
        rows = _reference.pool_weighted_mean(_batch(), pool.vectors, pool.weights, pool.scales)
        want = _reference.postprocess(rows, steps, 5)
        assert out.view(np.uint32).tolist() == want.view(np.uint32).tolist()


def test_pooled_rows_must_not_overlap_a_step():
    from fse._contract import RemoveComponents

    components = np.zeros((1, 4), F32)
    components[0, 0] = 1
    step = RemoveComponents(components)
    held = step.components  # the step's own frozen copy, made writable by a caller
    held.setflags(write=True)
    with pytest.raises(ValueError, match="overlap the components of a step"):
        _native.postprocess_pooled(held, (step,))
