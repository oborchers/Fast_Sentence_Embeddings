"""fse._backends._mlx._kernels on the GPU: the rounding of 1 / n, widening, the flags of
the pooling kernel, the kernel cache and the canary.

They skip where the mlx backend cannot run (support.conformance.require_mlx); the arrays
of a task, the split of a batch and the flags (plain numpy) are in
tests/test_backends_mlx_kernels_task.py.
"""

import subprocess
import sys

import numpy as np
import pytest

from fse._backends import PoolInput, UnsupportedByBackendError
from fse._backends._mlx import _kernels, _sources
from fse._contract import Batch
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


def test_the_canary_passes_here(gpu):
    gpu(_kernels.canary)


def test_the_canary_rejects_other_bits(gpu, monkeypatch):
    monkeypatch.setattr(_kernels, "CANARY_BITS", (0x33000000, 0x3F800001, 0))
    with pytest.raises(UnsupportedByBackendError, match="fma or the rounding of 1 / n"):
        gpu(_kernels.canary)


def test_the_canary_rejects_a_kernel_without_the_nonfinite_flag(gpu, monkeypatch):
    without = _sources._POOL.replace("flag |= FSE_NONFINITE;", "")
    monkeypatch.setattr(_sources, "_POOL", without)
    monkeypatch.setattr(_kernels, "_KERNELS", {})
    with pytest.raises(UnsupportedByBackendError, match="its flags differ"):
        gpu(_kernels.canary)


def test_the_canary_rejects_a_gpu_that_flushes_halves(gpu, monkeypatch):
    monkeypatch.setattr(_kernels, "CANARY_HALF", (0x0001, 0))
    with pytest.raises(UnsupportedByBackendError, match="does not widen float16 exactly"):
        gpu(_kernels.canary)


@pytest.mark.parametrize("env", ["flush=True", "round_up=True"])
def test_the_canary_passes_when_fse_is_imported_in_another_environment(env):
    """In a fresh process whose main thread flushes or rounds up before fse loads."""
    require_mlx()
    code = (
        "from fse import _native\n"
        f"_native._test_fpenv_set({env})\n"
        "import numpy as np\n"
        "from fse._backends import PoolInput, get_backend\n"
        "get_backend('mlx').prepare(PoolInput(np.ones((1, 2), np.float32), np.ones(1, np.float32)), ())\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.usefixtures("gpu")
@pytest.mark.parametrize("dim", [8, 9])
def test_the_flags_near_the_limits_of_normalize(dim):
    """HUGE from 2**60, TINY for a row all below 2**-50 (and an empty one), for one and
    four values per thread; a row with one value above is not TINY."""
    values = [2.0**60, 2.0**60 * 0.99, 2.0**-50 * 0.99, 2.0**-50, 0.0]
    rows = np.zeros((len(values), dim), F32)
    for i, v in enumerate(values):
        rows[i, :] = 2.0**-60
        rows[i, dim - 1] = v
    rows[4] = 0
    n = len(values)
    batch = Batch(0, np.array([*range(n + 1), n], np.int64), np.arange(n, dtype=np.uint32))
    pool = PoolInput(rows, np.ones(n, F32))
    table = _device().call(lambda mx, _s: _kernels.table(mx, pool, None))
    task = _kernels.task(batch, table)
    out = np.empty((n + 1, dim), F32)
    flags = _device().run(
        lambda mx, s: _kernels.launch(mx, s, table, task),
        lambda launched: _kernels.read(launched, out),
    )
    huge, tiny = _sources.HUGE, _sources.TINY
    assert flags.tolist() == [huge, 0, tiny, 0, tiny, tiny]
