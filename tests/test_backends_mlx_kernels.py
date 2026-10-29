"""fse._backends._mlx._kernels on the GPU: the rounding of 1 / n, widening, the bounds
and squares of the pooling kernel and the kernel cache.

They skip where the mlx backend cannot run (support.conformance.require_mlx); the arrays
of a task, the split of a batch and the flags (plain numpy) are in
tests/test_backends_mlx_kernels_task.py.
"""

import numpy as np
import pytest

from fse import _reference
from fse._backends import PoolInput
from fse._backends._mlx import _kernels, _sources
from fse._contract import Batch, Normalize
from support.conformance import mlx_module, require_mlx

F32 = np.float32


def _device():
    """The device thread of the process (the backend imports MLX)."""
    return mlx_module("fse._backends._mlx._backend").DEVICE


@pytest.fixture(scope="module")
def gpu():
    """Run ``function(mx, stream)`` in the device thread, after the canary."""
    require_mlx()
    _device().ready()
    return _device().call


_PRIMITIVES = """
    const uint i = thread_position_in_grid.x;
    uint flag = 0u;
    out[i] = metal::precise::divide(1.0f, a[i]);
    flags[i] = flag;
"""


def _inverse(a):
    """``precise::divide(1, a)`` on the GPU, elementwise."""

    def run(mx, stream):
        kernel = mx.fast.metal_kernel(
            name="fse_test_primitives",
            input_names=["a", "b", "c"],
            output_names=["out", "flags"],
            source=_PRIMITIVES,
            header=_sources.HEADER,
            compile_options={"math_mode": "safe"},
        )
        n = len(a)
        out, flags = kernel(
            inputs=[mx.array(a)] * 3,
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.uint32],
            stream=stream,
        )
        mx.eval(out, flags)
        return np.array(out), np.array(flags)

    return _device().call(run)


def _bits(x):
    return np.ascontiguousarray(x).view(np.uint32)


@pytest.mark.usefixtures("gpu")
def test_the_inverse_of_every_count_is_correctly_rounded():
    """1 / n for every n of [1, 2**24), as the kernel rounds it before the product."""
    n = np.arange(1, 1 << 24, dtype=F32)
    got, _ = _inverse(n)
    np.testing.assert_array_equal(_bits(got), _bits(F32(1) / n))


def _widen(rows):
    """Every value of ``rows`` (float16 or int8) read by the pooling kernel with weight 1,
    and the flags of its sentences."""
    n = rows.size
    batch = Batch(0, np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.uint32))
    scales = np.ones(n, F32) if rows.dtype == np.int8 else None
    pool = PoolInput(rows.reshape(n, 1), np.ones(n, F32), scales)
    table = _device().call(lambda mx, _s: _kernels.table(mx, pool, None))
    task = _kernels.task(batch, table)
    out = np.empty((n, 1), F32)
    flags = _device().run(
        lambda mx, s: _kernels.launch(mx, s, table, task),
        lambda launched: _kernels.read(launched, out),
    )
    return out.ravel(), flags & _kernels.ALWAYS


@pytest.mark.usefixtures("gpu")
def test_every_half_and_int8_value_widens_exactly():
    halves = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    halves = halves[np.isfinite(halves)]
    got, flags = _widen(halves)
    assert not flags.any()
    # Pooling adds to +0.0, which turns -0.0 into +0.0, as in the reference.
    np.testing.assert_array_equal(_bits(got), _bits(halves.astype(F32) + F32(0)))
    codes = np.arange(-128, 128, dtype=np.int8)
    got, flags = _widen(codes)
    assert not flags.any()
    np.testing.assert_array_equal(got, codes.astype(F32))


def test_the_kernel_is_built_once(gpu, monkeypatch):
    built = []

    def count(mx, _stream):
        real = mx.fast.metal_kernel
        monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: built.append(1) or real(**kw))
        monkeypatch.setattr(_kernels, "_KERNELS", {})
        first = _kernels.kernel(mx, 4)
        assert _kernels.kernel(mx, 4) is first
        return first

    gpu(count)
    assert built == [1]


@pytest.mark.usefixtures("gpu")
@pytest.mark.parametrize("dim", [8, 9])
def test_the_kernel_bounds_the_magnitudes_and_gives_the_squares(dim):
    """With bound and normalize: per row sum |f| ||x|| (the norms rounded up, so at least
    the float64 value) and the dot16 of the row with itself, for one and four values per
    thread; the rows normalized as the cpu does."""
    rng = np.random.default_rng(dim)
    rows = rng.standard_normal((5, dim)).astype(F32)
    weights = rng.random(5).astype(F32)
    batch = Batch(0, np.array([0, 3, 3, 5], np.int64), np.array([0, 1, 2, 3, 4], np.uint32))
    pool = PoolInput(rows, weights)
    table = _device().call(lambda mx, _s: _kernels.table(mx, pool, None))
    task = _kernels.task(batch, table)
    out = np.empty((3, dim), F32)

    def launch(mx, s):
        return _kernels.launch(mx, s, table, task, normalize=True)

    flags, bounds = _device().run(
        launch, _kernels.view, 0, lambda v: (np.copyto(out, v[0]), v[1], v[2])[1:]
    )
    assert not flags.any()
    norms = np.linalg.norm(rows.astype(np.float64), axis=1)
    exact = [np.dot(np.abs(weights[t]), norms[t]) for t in ([0, 1, 2], [], [3, 4])]
    assert (bounds[:, 0].astype(np.float64) >= np.array(exact) * (1 - 2.0**-20)).all()
    pooled = _reference.pool_weighted_mean(batch, rows, weights)
    np.testing.assert_array_equal(bounds[[0, 2], 1], _reference.dot16(pooled, pooled)[[0, 2]])
    want = _reference.postprocess(pooled, (Normalize(),))
    np.testing.assert_array_equal(_bits(out), _bits(want))
