"""fse._backends._mlx._stages: the steps one by one on the cpu host give the bits of all
steps at once, with the checks between them: a row that may overflow in a linear step or
could be zero or tiny before normalize is zeroed and returned to be taken from the host.
Plain numpy on the cpu backend, so these run on every machine."""

import numpy as np

from fse._backends import CpuBackend, PoolInput
from fse._backends._mlx import _stages
from fse._contract import Normalize, Project, RemoveComponents

F32 = np.float32


def _basis(dim, k, seed):
    q = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, k)))[0]
    return np.ascontiguousarray(q.T, dtype=F32)


STEPS = (
    RemoveComponents(_basis(8, 2, 1), np.array([1.0, 0.5], F32)),
    Project(_basis(8, 4, 2)),
    Normalize(),
)


def _all_at_once(rows, steps):
    cpu = CpuBackend()
    prepared = cpu.prepare(PoolInput(np.zeros((1, rows.shape[1]), F32), np.ones(1, F32)), steps)
    out = np.empty((len(rows), prepared.output_dim), F32)
    cpu.steps(rows.copy(), prepared, 0, out)
    return out


def test_the_stages_are_the_steps():
    stages = _stages.stages(CpuBackend(), STEPS, 8)
    assert [s.dim for s in stages] == [8, 8, 4]
    assert [s.linear is None for s in stages] == [False, False, True]
    assert stages[1].linear.dim == 4


def test_the_stages_give_the_bits_of_all_steps_at_once():
    rows = np.random.default_rng(3).standard_normal((20, 8)).astype(F32)
    stages = _stages.stages(CpuBackend(), STEPS, 8)
    out = np.empty((20, 4), F32)
    error = np.full(20, 1e-6)
    bad = _stages.run(
        CpuBackend(), stages, rows.copy(), out, 0, error, np.ones(20, bool), np.zeros(20, bool)
    )
    assert not bad.any()
    assert out.tobytes() == _all_at_once(rows, STEPS).tobytes()


def test_the_checks_between_the_stages():
    """Row 0 vanishes in the projection (before normalize: zero, from the host), row 1
    may overflow in the projection, row 2 is empty (an exact zero row, kept), row 3 was
    flagged by the kernel."""
    rows = np.array([[1, 0], [2e30, 2e30], [0, 0], [1, 1], [1, 2]], F32)
    steps = (Project(np.array([[0.0, 1.0]], F32)), Normalize())
    stages = _stages.stages(CpuBackend(), steps, 2)
    out = np.empty((5, 1), F32)
    nonempty = np.array([True, True, False, True, True])
    flagged = np.array([False, False, False, True, False])
    bad = _stages.run(
        CpuBackend(), stages, rows.copy(), out, 0, np.full(5, 1e-6), nonempty, flagged
    )
    assert bad.tolist() == [True, True, False, True, False]
    assert out[[2, 4]].tolist() == [[0.0], [1.0]]
