"""fse._backends._mlx._backend: rows on the GPU, the rest on the cpu host, the fallback
of flagged rows, and encoders on the mlx backend with the bits of the cpu backend.

The bits of every case are in the conformance suites; these test the parts around them.
They skip where the mlx backend cannot run.
"""

import logging

import numpy as np
import pytest

import fse
from fse import _reference
from fse._backends import CpuBackend, PoolInput, Prepared, UnsupportedByBackendError
from fse._contract import Batch, Normalize, Project, RemoveComponents
from support.conformance import TINY, mlx_module, require_mlx
from support.runtime import texts, word_vectors

_backend = mlx_module("fse._backends._mlx._backend")
_kernels = mlx_module("fse._backends._mlx._kernels")
MlxBackend, fallbacks = _backend.MlxBackend, _backend.fallbacks
F32 = np.float32


@pytest.fixture(scope="module")
def mlx():
    require_mlx()
    return MlxBackend(CpuBackend())


def _batch(first_row=0):
    offsets = np.array([0, 3, 3, 5, 9], np.int64)
    return Batch(first_row, offsets, np.array([0, 1, 2, 2, 3, 4, 0, 1, 4], np.uint32))


def _pool(dim=7, seed=0):
    rng = np.random.default_rng(seed)
    return PoolInput(rng.standard_normal((5, dim)).astype(F32), rng.random(5).astype(F32))


def _rows(backend, pool, steps=(), batch=None):
    batch = batch or _batch()
    prepared = backend.prepare(pool, steps)
    out = np.empty((batch.n_sentences, prepared.output_dim), F32)
    backend.rows(batch, prepared, out)
    return out, prepared


def test_capabilities(mlx):
    caps = mlx.capabilities()
    assert caps.word_dtypes == {"float32", "float16", "int8"}
    assert caps.fasttext
    assert caps.concurrent
    assert caps.max_dim == _backend.MAX_DIM == 1 << 16


def test_rows_wider_than_its_limit_are_refused(mlx):
    pool = PoolInput(np.zeros((1, _backend.MAX_DIM + 1), F32), np.ones(1, F32))
    with pytest.raises(UnsupportedByBackendError, match="rows of 65537 values"):
        mlx.prepare(pool, ())


@pytest.mark.parametrize(
    "steps",
    [
        (Normalize(),),
        (RemoveComponents(np.eye(1, 7, dtype=F32)), Normalize()),
        (Project(np.ascontiguousarray(np.eye(3, 7, dtype=F32))),),
        (
            RemoveComponents(np.eye(2, 7, dtype=F32), np.array([1, 0.5], F32)),
            Project(np.ascontiguousarray(np.eye(4, 7, k=1, dtype=F32))),
            Normalize(),
        ),
    ],
    ids=["normalize", "remove-normalize", "project", "all"],
)
def test_the_steps_run_on_the_host_with_its_bits(mlx, steps):
    got, prepared = _rows(mlx, _pool(), steps)
    want, _ = _rows(CpuBackend(), _pool(), steps)
    assert got.tobytes() == want.tobytes()
    assert sum(fallbacks(prepared).values()) == 0


def test_a_batch_the_kernel_cannot_read_raises_the_cpu_error(mlx):
    batch = Batch(0, np.array([0, 2], np.int64), np.array([0, 5], np.uint32))
    with pytest.raises(ValueError, match="token ids must be below 5"):
        _rows(mlx, _pool(), batch=batch)


def test_an_empty_batch_writes_nothing(mlx):
    out, _ = _rows(mlx, _pool(), batch=Batch(0, np.zeros(1, np.int64), np.zeros(0, np.uint32)))
    assert out.shape == (0, 7)


def _nudged(monkeypatch):
    """A device whose rows are one ulp above the real ones (still within the bound)."""
    real = _kernels.view

    def view(launched):
        rows, flags = real(launched)
        return np.where(rows != 0, np.nextafter(rows, F32(np.inf)), rows), flags

    monkeypatch.setattr(_kernels, "view", view)


def _limit_pool():
    """Word 3 with values of 2**-62: alone (sentence 2 of ``_limit_batch``) it is a row
    whose sum of squares may be tiny, flagged before normalize."""
    pool = _pool()
    pool.vectors[3] = 2.0**-62
    return pool


def _limit_batch(first_row=0):
    offsets = np.array([0, 3, 3, 4, 8], np.int64)
    return Batch(first_row, offsets, np.array([0, 1, 2, 3, 4, 0, 1, 4], np.uint32))


@pytest.mark.parametrize(
    "steps",
    [
        (Normalize(),),
        (RemoveComponents(np.eye(1, 7, dtype=F32)), Project(np.eye(3, 7, dtype=F32)), Normalize()),
    ],
    ids=["normalize", "all"],
)
def test_only_the_flagged_rows_come_from_the_host(mlx, caplog, monkeypatch, steps):
    """Sentence 2 comes from the host, the others keep the device's rows (one ulp above
    the cpu's here) through the steps; a task counts once per call, the warning once per
    encoder."""
    caplog.set_level(logging.WARNING, logger="fse")
    _nudged(monkeypatch)
    got, prepared = _rows(mlx, _limit_pool(), steps, _limit_batch())
    want, _ = _rows(CpuBackend(), _limit_pool(), steps, _limit_batch())
    assert got[2].tobytes() == want[2].tobytes()
    assert got[0].tobytes() != want[0].tobytes()
    assert got[3].tobytes() != want[3].tobytes()
    assert not got[1].any()  # the empty sentence
    mlx.rows(_limit_batch(7), prepared, np.empty_like(got))
    counts = {"subnormal_input": 0, "long": 0, "nonfinite": 0, "large": 0, "limit": 2}
    assert fallbacks(prepared) == counts
    warnings = [r for r in caplog.records if "computes a task on the CPU" in r.message]
    assert len(warnings) == 1
    assert "sum of squares" in warnings[0].message


def test_an_encoder_with_subnormal_inputs_runs_on_the_host(mlx, caplog):
    pool = _pool()
    pool.vectors[3, 2] = TINY * 5
    caplog.set_level(logging.WARNING, logger="fse")
    device = mlx.device
    before = device.launches
    got, prepared = _rows(mlx, pool)
    want, _ = _rows(CpuBackend(), pool)
    assert got.tobytes() == want.tobytes()
    assert device.launches == before
    assert fallbacks(prepared)["subnormal_input"] == 1
    assert any("subnormal" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("vectors", "reason"),
    [([[2.0**-62, 0.0], [1.0, 1.0]], "limit"), ([[2.0**61, 1.0], [1.0, 1.0]], "limit")],
    ids=["tiny", "huge"],
)
def test_rows_near_a_limit_of_normalize_come_from_the_host(mlx, vectors, reason):
    """Its sum of squares may be subnormal (or overflow): flagged only before normalize."""
    pool = PoolInput(np.array(vectors, F32), np.ones(2, F32))
    batch = Batch(0, np.array([0, 1, 2], np.int64), np.array([0, 1], np.uint32))
    for steps, counted in (((), 0), ((Normalize(),), 1)):
        prepared = mlx.prepare(pool, steps)
        out = np.empty((2, 2), F32)
        mlx.rows(batch, prepared, out)
        assert fallbacks(prepared)[reason] == counted
        want, _ = _rows(CpuBackend(), pool, steps, batch)
        assert out.tobytes() == want.tobytes()


def test_a_normalize_error_of_a_flagged_row_is_the_cpus(mlx):
    """1e-25 squared underflows on the cpu: the host raises for the task."""
    pool = PoolInput(np.array([[1.0, 1.0], [1e-25, 0.0]], F32), np.ones(2, F32))
    batch = Batch(3, np.array([0, 1, 2], np.int64), np.array([0, 1], np.uint32))
    prepared = mlx.prepare(pool, (Normalize(),))
    with pytest.raises(ValueError, match=r"^row 4 cannot be normalized"):
        mlx.rows(batch, prepared, np.empty((2, 2), F32))


def test_a_row_that_is_not_finite_raises_the_cpu_error_through_the_host(mlx):
    pool = PoolInput(np.array([[3e38], [3e38]], F32), np.ones(2, F32))
    batch = Batch(4, np.array([0, 1, 3], np.int64), np.array([0, 0, 1], np.uint32))
    prepared = mlx.prepare(pool, ())
    with pytest.raises(ValueError, match=r"^row 5 contains NaN or infinite values"):
        mlx.rows(batch, prepared, np.empty((2, 1), F32))
    assert fallbacks(prepared)["nonfinite"] == 1


def test_fallbacks_need_an_mlx_prepared():
    with pytest.raises(TypeError, match="prepared by the mlx backend"):
        fallbacks(Prepared("cpu", _pool(), (), 7, 7, None))


def test_the_host_runs_everything_else(mlx):
    host, rng = CpuBackend(), np.random.default_rng(3)
    stored = rng.standard_normal((6, 5)).astype(F32)
    weights, queries = rng.standard_normal(5).astype(F32), rng.standard_normal((2, 5))
    inverse = host.inverse_norms(stored)
    pairs = [
        (mlx.inverse_norms(stored, 3), host.inverse_norms(stored, 3)),
        (mlx.score(stored, None, weights, 0.5), host.score(stored, None, weights, 0.5)),
        (mlx.scores(stored, None, queries, 0.0), host.scores(stored, None, queries, 0.0)),
        (mlx.cosines(stored, inverse, queries), host.cosines(stored, inverse, queries)),
        (mlx.gram_block(stored), host.gram_block(stored)),
        (mlx.float32([1e-46, 2.5]), host.float32([1e-46, 2.5])),
    ]
    for got, want in pairs:
        assert got.tobytes() == want.tobytes()
    codes, scales = np.empty((6, 5), np.int8), np.empty(6, F32)
    mlx.store(stored, codes, scales, 0)
    want_codes, want_scales = _reference.quantize_rows(stored, "int8")
    assert codes.tobytes() + scales.tobytes() == want_codes.tobytes() + want_scales.tobytes()


@pytest.mark.usefixtures("mlx")
@pytest.mark.parametrize(
    "recipe", [fse.Average(dim=5), fse.SIF(components=1, fold=False)], ids=["average", "sif"]
)
def test_encoders_on_mlx_give_the_cpu_bits(recipe):
    """fit (its Gram matrix from rows of the backend) and encode, with FastText and int8."""
    wv, sentences = word_vectors("int8", True), texts(400)
    counts = np.arange(1, len(wv) + 1) * 10
    options = {} if isinstance(recipe, fse.Average) else {"counts": counts}
    cpu = recipe.fit(wv, sentences, backend="cpu", **options)
    on_mlx = recipe.fit(wv, sentences, backend="mlx", **options)
    assert on_mlx.fingerprint == cpu.fingerprint
    got = on_mlx.encode(sentences, backend="mlx", workers=3)
    assert got.tobytes() == cpu.encode(sentences, backend="cpu", workers=3).tobytes()


def test_a_batch_larger_than_a_device_group_runs_in_parts(mlx, monkeypatch):
    """Runs of sentences that fit a group on the device, a longer sentence on the host."""
    monkeypatch.setattr(mlx.device, "group_tokens", 3)
    _nudged(monkeypatch)
    got, prepared = _rows(mlx, _pool())  # sentences of 3, 0, 2 and 4 tokens
    want, _ = _rows(CpuBackend(), _pool())
    assert got[3].tobytes() == want[3].tobytes()
    assert got[[0, 2]].tobytes() == np.nextafter(want[[0, 2]], F32(np.inf)).tobytes()
    assert fallbacks(prepared)["large"] == 1


def test_the_window_counts_the_rows_on_the_device(mlx):
    from fse._runtime import EncodeSpec, Pipeline

    spec = EncodeSpec(word_vectors(dim=40), np.ones(200, F32))
    on_mlx, on_cpu = Pipeline(spec, "mlx"), Pipeline(spec, "cpu")
    assert mlx.capabilities().device_rows
    assert on_mlx._row_bytes("float32") == on_cpu._row_bytes("float32") + 4 * 40 + 4 * 2
