"""The witnesses of review 3 of PR #118 for the mlx backend: a flush that turns roundings
must not change which error or success the cpu backend would give. Each case has normal,
finite inputs (the prepare check passes) and is computed on the GPU.

- a zero factor of a word whose stored norm is infinite (R1): the bound stays finite;
- a row the cpu rounds above the float16 range while the device stays at 65504 (R2): the
  float16 store raises the cpu's error;
- a sum the cpu rounds to infinity while the device's stays finite (R4): pooling raises
  the cpu's error;
- a row whose squared norm is below 2**-100 on the device (R8): before normalize the
  host's row, which the suite demands (a mutant without the check fails it).
"""

import numpy as np
import pytest

from fse._backends import CpuBackend, PoolInput
from fse._backends._mlx import _bounds, _kernels
from fse._contract import Batch, Normalize
from support.bound import assert_within_bound
from support.cases import Case
from support.conformance import F32, Contract, mlx_module, pool_on_device

F = F32(1 + 2**-12)


@pytest.fixture
def mlx():
    return Contract("mlx").backend  # skips where it cannot run


def _one_sentence(weights, values):
    """One sentence of one-dimensional words, token i word i."""
    rows = np.array(values, F32).reshape(len(values), -1)
    n = len(values)
    batch = Batch(0, np.array([0, n], np.int64), np.arange(n, dtype=np.uint32))
    return PoolInput(rows, np.array(weights, F32)), batch


def _run(backend, pool, batch, steps=()):
    prepared = backend.prepare(pool, steps)
    out = np.empty((batch.n_sentences, prepared.output_dim), F32)
    backend.rows(batch, prepared, out)
    return out, prepared


def test_a_zero_factor_of_an_infinite_norm_keeps_the_bound_finite(mlx):
    big = np.finfo(F32).max
    rows = np.array([[2.0**63, 2.0**-64], [big, big]], F32)
    case = Case("zero factor", "float32", rows, np.array([2.0**-63, 0], F32), ((0, 1),))
    device = pool_on_device(mlx, case)[0]
    out, fell_back, launched = pool_on_device(mlx, case, (Normalize(),))
    assert launched
    assert not fell_back
    assert_within_bound(out, case, (Normalize(),), case.name, device=device)


R2 = ([2.0**-63, F, 1, 4097, 1, 1], [2.0**-64, F, -(1 + 2.0**-11), 4097, -16785408, 393024])


def test_a_row_the_cpu_rounds_beyond_float16_raises_in_the_store(mlx):
    pool, batch = _one_sentence(*R2)
    cpu, _ = _run(CpuBackend(), pool, batch)
    out, prepared = _run(mlx, pool, batch)
    assert cpu[0, 0] > 65504 >= out[0, 0]  # the witness: the cpu's row is beyond the range
    codes = np.empty(out.shape, np.float16)
    with pytest.raises(ValueError, match="float16") as on_cpu:
        CpuBackend().store(cpu, codes, None, 0)
    with pytest.raises(ValueError, match="float16") as on_mlx:
        mlx.store(out, codes, None, 0)
    assert str(on_mlx.value) == str(on_cpu.value)
    assert mlx_module("fse._backends._mlx._backend").fallbacks(prepared)["limit"] == 1


R4 = ([2.0**-63, F, 1, 31], [2.0**-64, F, -(1 + 2.0**-11 + 2.0**-23), 1082401 * 2.0**103])


def test_a_sum_the_cpu_rounds_to_infinity_raises_the_cpu_error(mlx, monkeypatch):
    pool, batch = _one_sentence(*R4)
    with pytest.raises(ValueError, match="row 0 contains NaN or infinite") as on_cpu:
        _run(CpuBackend(), pool, batch)
    seen = []
    real = _kernels.view
    monkeypatch.setattr(_kernels, "view", lambda launched: seen.append(1) or real(launched))
    with pytest.raises(ValueError, match="row 0 contains NaN or infinite") as on_mlx:
        _run(mlx, pool, batch)
    assert seen  # on the GPU, whose accumulator stays finite: found by the bound
    assert str(on_mlx.value) == str(on_cpu.value)


def test_a_device_row_with_squares_below_2_100_comes_from_the_host(mlx, monkeypatch):
    """With a device one ulp off the cpu: the row from the host passes the suite; a
    backend without the check (LOW 0) gives the device's row, which the suite refuses."""
    case = Case("tiny row", "float32", np.full((1, 1), 2.0**-55, F32), np.ones(1, F32), ((0,),))
    real = _kernels.view

    def nudged(launched):
        rows, flags, bounds = real(launched)
        return np.nextafter(rows, F32(np.inf)), flags, bounds

    monkeypatch.setattr(_kernels, "view", nudged)
    device = pool_on_device(mlx, case)[0]
    out, fell_back, _ = pool_on_device(mlx, case, (Normalize(),))
    assert fell_back
    assert_within_bound(out, case, (Normalize(),), case.name, device=device)
    monkeypatch.setattr(_bounds, "LOW", 0.0)
    out, _, _ = pool_on_device(mlx, case, (Normalize(),))
    with pytest.raises(AssertionError, match="must be the cpu's"):
        assert_within_bound(out, case, (Normalize(),), case.name, device=device)
