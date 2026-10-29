"""fse._backends._mlx._kernels on the GPU: the checks of the pooling kernel, the prepare
kernel, widening, the kernel cache and the canary.

They skip where the mlx backend cannot run (support.conformance.require_mlx); the arrays
of a task, the split of a batch and the flags (plain numpy) are in
tests/test_backends_mlx_kernels_task.py.
"""

import subprocess
import sys

import numpy as np
import pytest

from fse import _reference
from fse._backends import PoolInput, UnsupportedByBackendError
from fse._backends._mlx import _kernels, _sources
from fse._contract import Batch
from support.conformance import TINY, mlx_module, require_mlx

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
    if (MODE == 0) {
        out[i] = metal::fma(a[i], b[i], c[i]);
        flag = fse_step(a[i], b[i], c[i], out[i]);
    } else if (MODE == 1) {
        out[i] = fse_mul(a[i], b[i], flag);
    } else {
        out[i] = metal::precise::divide(1.0f, a[i]);
    }
    flags[i] = flag;
"""


def _primitive(mode, a, b=None, c=None):
    """``fma`` with ``fse_step`` (mode 0), ``fse_mul`` (1) or ``precise::divide(1, a)``
    (2) on the GPU, elementwise, with the flags."""

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
            inputs=[mx.array(x if x is not None else a) for x in (a, b, c)],
            template=[("MODE", mode)],
            grid=(n, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32, mx.uint32],
            stream=stream,
        )
        mx.eval(out, flags)
        return np.array(out), np.array(flags)

    return _device().call(run)


def _floats(rng, n, low, high):
    """float32 values with exponent fields from ``low`` to ``high`` (0 is subnormal) and
    random signs and mantissas."""
    bits = rng.integers(low, high + 1, n).astype(np.uint32) << 23
    bits |= rng.integers(0, 1 << 23, n).astype(np.uint32)
    bits |= rng.integers(0, 2, n).astype(np.uint32) << 31
    return bits.view(F32)


def _bits(x):
    return np.ascontiguousarray(x).view(np.uint32)


def _exponent(x):
    return (_bits(x) >> 23) & 0xFF


def _unflagged_inputs(seed, n=1 << 20):
    """fma operands as the pooling kernel meets them in a task it does not flag: factors
    and values normal or zero whose exponent fields sum to 128 or more, accumulators
    normal above 2**-126 or zero; products near the subnormal range, and sums that
    cancel exactly or within the rounding of the product."""
    rng = np.random.default_rng(seed)
    a = _floats(rng, n, 1, 140)
    b = _floats(rng, n, 1, 140)
    small = _exponent(a).astype(int) + _exponent(b) < 128
    b[small] = _floats(rng, int(small.sum()), 128, 140)
    c = _floats(rng, n, 2, 40)
    c[::7] = -(a[::7] * b[::7])  # cancels within the rounding of the product
    c[::13] = 0
    a[::11] = 0
    c[_bits(c) & 0x7FFFFFFF <= 0x00800000] = 0  # no accumulator at or below 2**-126
    return a, b, c


@pytest.mark.usefixtures("gpu")
def test_an_unflagged_fma_has_the_reference_bits():
    a, b, c = _unflagged_inputs(0)
    got, flags = _primitive(0, a, b, c)
    want = _reference.fma32(a, b, c)
    clear = flags == 0
    assert 0.5 < clear.mean() < 0.99, clear.mean()  # both kinds occur
    np.testing.assert_array_equal(_bits(got)[clear], _bits(want)[clear])
    # Flagged where IEEE gives a subnormal result or 2**-126, which the GPU may flush.
    low = (_bits(want) & 0x7FFFFFFF) - 1 < 0x00800000
    assert low.sum() > 1000
    assert (flags[low] == _sources.SUBNORMAL).all()
    assert (_bits(got) != _bits(want)).any(), "the GPU kept every subnormal: no flush to test"


@pytest.mark.usefixtures("gpu")
def test_a_sum_that_cancels_exactly_is_no_subnormal():
    """An exact product that cancels gives +0.0 on the GPU too; a rounded one that equals
    -c leaves its rounding error, which may be subnormal; zero from zero is no flag."""
    f = F32(1 + 2**-12)
    a = np.array([1.5, -3.0, 2.0**-60 * f, 3.0, f, 0.0], F32)
    b = np.array([2.0, 2.0**-100, 2.0**-60 * f, 0.0, f, 5.0], F32)
    c = -(a * b)
    c[3] = c[5] = 0
    got, flags = _primitive(0, a, b, c)
    assert flags.tolist() == [0, 0, _sources.SUBNORMAL, 0, 0, 0]
    want = _reference.fma32(a, b, c)
    np.testing.assert_array_equal(_bits(got)[[0, 1, 3, 4, 5]], _bits(want)[[0, 1, 3, 4, 5]])


@pytest.mark.usefixtures("gpu")
def test_unflagged_products_have_the_reference_bits():
    rng = np.random.default_rng(1)
    a, b = _floats(rng, 1 << 20, 0, 140), _floats(rng, 1 << 20, 0, 140)
    a[::11] = 0
    got, flags = _primitive(1, a, b)
    with np.errstate(under="ignore", over="ignore"):
        want = a * b
    clear = flags == 0
    assert 0.1 < clear.mean() < 0.9
    np.testing.assert_array_equal(_bits(got)[clear], _bits(want)[clear])
    assert (flags[(np.abs(want) < np.finfo(F32).tiny) & (want != 0)] != 0).all()


@pytest.mark.usefixtures("gpu")
def test_the_inverse_of_every_count_is_correctly_rounded():
    """1 / n for every n of [1, 2**24), as the kernel rounds it before the product."""
    n = np.arange(1, 1 << 24, dtype=F32)
    got, _ = _primitive(2, n)
    np.testing.assert_array_equal(_bits(got), _bits(F32(1) / n))


def _table(pool):
    return _device().call(lambda mx, stream: _kernels.table(mx, stream, pool, None))


def _factors(table):
    """The factors the prepare kernel made, and the ngram exponents."""
    return _device().call(
        lambda _mx, _s: (np.asarray(table.arrays[1]).view(F32), np.asarray(table.arrays[4]))
    )


@pytest.mark.usefixtures("gpu")
def test_the_prepare_kernel_flags_the_words_a_flush_could_change():
    rows = np.array(
        [
            [1.0, 0.0, -2.0, 4.0],  # 0: normal
            [1.0, TINY, 0.0, 1.0],  # 1: a subnormal value
            [2.0**-63, 1.0, 0.0, 1.0],  # 2: with a weight of 2**-63, exponents sum to 128
            [2.0**-64, 1.0, 0.0, 1.0],  # 3: ... to 127: a product may be subnormal
            [0.0, 0.0, 0.0, 0.0],  # 4: no value, any weight
            [1.0, 2.0, 3.0, 4.0],  # 5: a subnormal weight
            [1.0, 2.0, 3.0, 4.0],  # 6: weight zero
        ],
        F32,
    )
    weights = np.array([0.5, 1.0, 2.0**-63, 2.0**-63, TINY, TINY * 3, 0.0], F32)
    factors, _ = _factors(t := _table(PoolInput(rows, weights)))
    flagged = np.isnan(factors)
    assert flagged.tolist() == [False, True, False, True, True, True, False]
    assert _bits(factors[~flagged]).tolist() == _bits(weights[~flagged]).tolist()
    assert t.flagged.tolist() == flagged.tolist()


@pytest.mark.usefixtures("gpu")
def test_the_prepare_kernel_flags_int8_factors_that_may_underflow():
    codes = np.array([[1, -127], [127, 0], [5, 5], [1, 1]], np.int8)
    # Factors 2**-3; 2**-120 (exponent fields 67 + 67); a subnormal scale; and 2**-61 times
    # 2**-66, whose exponent fields sum to 127, so the product may be subnormal.
    weights = np.array([0.5, 2.0**-60, 1.0, 2.0**-61], F32)
    scales = np.array([0.25, 2.0**-60, TINY, 2.0**-66], F32)
    factors, _ = _factors(_table(PoolInput(codes, weights, scales)))
    assert np.isnan(factors).tolist() == [False, False, True, True]
    assert factors[:2].tolist() == [0.125, 2.0**-120]


@pytest.mark.usefixtures("gpu")
def test_the_prepare_kernel_gives_the_smallest_exponent_of_every_ngram_row():
    grams = np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0], [3.0, TINY, 1.0], [-(2.0**-100), 8, 0]])
    pool = PoolInput(
        np.ones((1, 3), F32), np.ones(1, F32), ngram_vectors=grams.astype(F32), oov_weight=1.0
    )
    t = _device().call(lambda mx, stream: _kernels.table(mx, stream, pool, np.ones(1, F32)))
    _, low = _factors(t)
    assert low.tolist() == [126, _sources.NO_EXPONENT, 0, 27]


def _widen(rows):
    """Every value of ``rows`` (float16 or int8) read by the pooling kernel with weight 1,
    and the flags of its sentences."""
    n = rows.size
    batch = Batch(0, np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.uint32))
    scales = np.ones(n, F32) if rows.dtype == np.int8 else None
    table = _table(PoolInput(rows.reshape(n, 1), np.ones(n, F32), scales))
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
        first = _kernels.kernel(mx, "pool4")
        assert _kernels.kernel(mx, "pool4") is first
        return first

    gpu(count)
    assert built == [1]


def test_the_canary_passes_here(gpu):
    gpu(_kernels.canary)


def test_the_canary_rejects_other_bits(gpu, monkeypatch):
    monkeypatch.setattr(_kernels, "CANARY_BITS", (0x33000000, 0x3F800001, 0))
    with pytest.raises(UnsupportedByBackendError, match="fma or the rounding of 1 / n"):
        gpu(_kernels.canary)


def test_the_canary_rejects_a_kernel_without_checks(gpu, monkeypatch):
    monkeypatch.setattr(
        _sources, "_POOL", _sources._POOL.replace("fse_step(f, x, before, acc)", "0u")
    )
    monkeypatch.setattr(_kernels, "_KERNELS", {})
    with pytest.raises(UnsupportedByBackendError, match="subnormal check does not work"):
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
