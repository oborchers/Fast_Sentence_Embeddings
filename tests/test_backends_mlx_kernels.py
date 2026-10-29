"""fse._backends._mlx._kernels: the primitives of the pooling kernel on the GPU, the
arrays of a task, the kernel cache and the canary.

The GPU tests skip where the mlx backend cannot run (support.conformance.require_mlx);
the tests of ``task`` are plain numpy and run everywhere.
"""

import functools

import numpy as np
import pytest

from fse import _reference
from fse._backends import PoolInput, UnsupportedByBackendError
from fse._backends._mlx import _kernels, _sources
from fse._backends._mlx._device import DEVICE
from fse._contract import OOV_TOKEN, Batch
from support.conformance import require_mlx

F32 = np.float32


@pytest.fixture(scope="module")
def gpu():
    """Run ``function(mx, stream)`` in the device thread."""
    require_mlx()
    return DEVICE.call


_PRIMITIVES = """
    const uint i = thread_position_in_grid.x;
    uint flag = 0u;
    if (MODE == 0) {
        out[i] = fse_fma(a[i], b[i], c[i], flag);
    } else if (MODE == 1) {
        out[i] = fse_mul(a[i], b[i], flag);
    } else {
        out[i] = metal::precise::divide(1.0f, a[i]);
    }
    flags[i] = flag;
"""


def _primitive(mode, a, b=None, c=None):
    """``fse_fma`` (mode 0), ``fse_mul`` (1) or ``precise::divide(1, a)`` (2) on the GPU,
    elementwise, with the flags."""

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

    return DEVICE.call(run)


def _floats(rng, n, low, high):
    """float32 values with exponent fields from ``low`` to ``high`` (0 is subnormal) and
    random signs and mantissas."""
    bits = rng.integers(low, high + 1, n).astype(np.uint32) << 23
    bits |= rng.integers(0, 1 << 23, n).astype(np.uint32)
    bits |= rng.integers(0, 2, n).astype(np.uint32) << 31
    return bits.view(F32)


def _near_underflow(seed, n=1 << 20):
    """Triples around the subnormal range: products whose exponents sum near 127, and
    sums that cancel; some operands are subnormal or zero."""
    rng = np.random.default_rng(seed)
    a = _floats(rng, n, 0, 140)
    b = _floats(rng, n, 0, 140)
    c = _floats(rng, n, 0, 40)
    c[::7] = -(a[::7] * b[::7])  # cancels within the rounding of the product
    a[::11] = 0
    return a, b, c


def _bits(x):
    return np.ascontiguousarray(x).view(np.uint32)


@pytest.mark.usefixtures("gpu")
def test_unflagged_fma_has_the_reference_bits():
    a, b, c = _near_underflow(0)
    got, flags = _primitive(0, a, b, c)
    want = _reference.fma32(a, b, c)
    clear = flags == 0
    assert 0.1 < clear.mean() < 0.9, clear.mean()  # both kinds occur
    np.testing.assert_array_equal(_bits(got)[clear], _bits(want)[clear])
    # Flagged where IEEE gives a subnormal result: the GPU flushes some of them.
    subnormal = (np.abs(want) < np.finfo(F32).tiny) & (want != 0)
    assert (flags[subnormal] & _sources.SUBNORMAL).all()
    assert (_bits(got) != _bits(want)).any(), "the GPU kept every subnormal: no flush to test"


@pytest.mark.usefixtures("gpu")
def test_a_sum_that_cancels_exactly_is_no_subnormal():
    """An exact product that cancels gives +0.0 on the GPU too; a rounded one that equals
    -c leaves its rounding error, which may be subnormal."""
    f = F32(1 + 2**-12)
    a = np.array([1.5, -3.0, 2.0**-70, 3.0, f, 2.0**-60 * f], F32)
    b = np.array([2.0, 2.0**-100, 2.0**-70, 0.0, f, 2.0**-60 * f], F32)
    c = -(a * b)
    c[3] = 0
    got, flags = _primitive(0, a, b, c)
    assert flags.tolist() == [0, 0, _sources.SUBNORMAL, 0, 0, _sources.SUBNORMAL]
    want = _reference.fma32(a, b, c)
    np.testing.assert_array_equal(_bits(got)[[0, 1, 3, 4]], _bits(want)[[0, 1, 3, 4]])


@pytest.mark.usefixtures("gpu")
def test_unflagged_products_have_the_reference_bits():
    a, b, _ = _near_underflow(1)
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


def _widen(rows):
    """Every value of ``rows`` (float16 or int8) read by the pooling kernel with weight 1."""
    n = rows.size
    batch = Batch(0, np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.uint32))
    scales = np.ones(n, F32) if rows.dtype == np.int8 else None
    pool = PoolInput(rows.reshape(n, 1), np.ones(n, F32), scales)
    table = DEVICE.call(lambda mx, _s: _kernels.table(mx, pool, None))
    task = _kernels.task(batch, table)
    out = np.empty((n, 1), F32)
    flags = DEVICE.run(
        lambda mx, s: _kernels.launch(mx, s, table, task),
        lambda launched: _kernels.read(launched, out),
    )
    return out.ravel(), flags


@pytest.mark.usefixtures("gpu")
def test_every_half_and_int8_value_widens_exactly():
    halves = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    halves = halves[np.isfinite(halves)]
    got, flags = _widen(halves)
    assert flags == 0
    # Pooling adds to +0.0, which turns -0.0 into +0.0, as in the reference.
    np.testing.assert_array_equal(_bits(got), _bits(halves.astype(F32) + F32(0)))
    codes = np.arange(-128, 128, dtype=np.int8)
    got, flags = _widen(codes)
    assert flags == 0
    np.testing.assert_array_equal(got, codes.astype(F32))


def test_the_kernel_is_built_once(gpu, monkeypatch):
    built = []

    def count(mx, _stream):
        real = mx.fast.metal_kernel
        monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: built.append(1) or real(**kw))
        first = _kernels.kernel(mx, "fse_test_cache")
        assert _kernels.kernel(mx, "fse_test_cache") is first
        return first

    gpu(count)
    assert built == [1]


def test_the_canary_passes_here(gpu):
    gpu(_kernels.canary)


def test_the_canary_rejects_other_bits(gpu, monkeypatch):
    monkeypatch.setattr(_kernels, "CANARY_BITS", (0x33000000, 0x3F800001))
    with pytest.raises(UnsupportedByBackendError, match="fma or the rounding of 1 / n"):
        gpu(_kernels.canary)


def test_the_canary_rejects_a_kernel_without_checks(gpu, monkeypatch):
    monkeypatch.setattr(_kernels, "launch", functools.partial(_kernels.launch, checks=False))
    with pytest.raises(UnsupportedByBackendError, match="subnormal check does not work"):
        gpu(_kernels.canary)


# --- The arrays of a task (numpy only) ------------------------------------------------------

TABLE = _kernels.Table(
    (), dim=3, n_words=5, n_buckets=4, int8=False, fasttext=True, oov_weight=True
)
U = OOV_TOKEN


def _batch(sentences, ngrams=None):
    offsets = np.cumsum([0, *map(len, sentences)]).astype(np.int64)
    tokens = np.array([t for s in sentences for t in s], np.uint32)
    if ngrams is None:
        return Batch(3, offsets, tokens)
    ngram_offsets = np.cumsum([0, *map(len, ngrams)]).astype(np.int64)
    flat = np.array([b for n in ngrams for b in n], np.uint32)
    return Batch(3, offsets, tokens, ngram_offsets, flat)


def test_a_task_numbers_the_first_unknown_word_of_every_sentence():
    batch = _batch(((U, 1), (), (2,), (U, U, 4), (U,)), ((0, 1), (), (3,), (2, 2)))
    task = _kernels.task(batch, TABLE)
    assert task is not None
    assert task.unknown
    assert (task.tokens, task.n_sentences, task.first_row) == (7, 5, 3)
    assert task.arrays[2].tolist() == [0, 1, 1, 1, 3]
    assert task.arrays[2].dtype == np.int64


def test_a_task_without_unknown_words_reads_no_ngrams():
    task = _kernels.task(_batch(((0, 4), (1,))), TABLE)
    assert task is not None
    assert not task.unknown
    assert [len(a) for a in task.arrays[2:]] == [1, 1, 1]


@pytest.mark.parametrize(
    ("batch", "table"),
    [
        (_batch(((0, 5),)), TABLE),  # a word beyond the rows
        (_batch(((U,),), ((4,),)), TABLE),  # a bucket beyond the ngram rows
        (_batch(((U,),), ((0,),)), _kernels.Table((), 3, 5, 0, False, False, False)),
        (_batch(((U,),), ((0,),)), _kernels.Table((), 3, 5, 4, False, True, False)),
    ],
    ids=["word", "bucket", "no-fasttext", "no-oov-weight"],
)
def test_a_batch_the_cpu_refuses_is_no_task(batch, table):
    assert _kernels.task(batch, table) is None
