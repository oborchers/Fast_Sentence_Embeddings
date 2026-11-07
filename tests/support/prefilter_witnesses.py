"""The fixed witnesses of the meta-tests of ``fse._reference_prefilter``.

Each builder is deterministic and returns stores and queries whose exact top row a weakened rule
loses; ``tests/test_reference_prefilter_meta.py`` states which rule each one tests. The geometry
of the first one: ``u = v + e`` is the query, ``v = s * qcodes`` its int8 copy and ``e`` the error.
A row along ``e`` has ``c - a`` of about ``+eps``, a row along ``-e`` about ``-eps``, so a pair
with ``a_B - a_A`` just below ``2 eps`` has ``c_A > c_B`` and still loses A to every bound below
``eps`` (and to the threshold ``T`` without ``- E``).
"""

from __future__ import annotations

import numpy as np

from fse import _reference_prefilter as prefilter
from support.prefilter import int8_store, prepared

F32 = np.float32
DIM = 256
# t / eps of the row along e + t * v: chosen so that a_B - a_A is 1.92 eps (checked by the test).
TILT = 2.34


def int8_rows(vectors):
    """Float rows scaled to the int8 range by their largest value (an adversarial store row)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    return np.round(127 * vectors / np.abs(vectors).max(axis=-1, keepdims=True)).astype(np.int8)


class Geometry:
    """One random query and the directions ``e``, ``v`` and ``u`` around it."""

    def __init__(self, seed=1, dim=DIM):
        self.rng = np.random.default_rng(seed)
        self.query = self.rng.standard_normal((1, dim)).astype(F32)
        units, self.qcodes, self.qscales, self.slack, _ = prepared(self.query)
        self.unit = units[0].astype(np.float64)
        v = self.qscales[0].astype(np.float64) * self.qcodes[0].astype(np.float64)
        error = self.unit - v
        self.eps = float(np.linalg.norm(error))
        self.e = error / self.eps
        self.v = v / np.linalg.norm(v)


def pair_store(seed=1):
    """A store of 14 rows, the pair A (position 5) and B (position 9) among fillers, k = 1.

    A is the exact top row. The fillers lie against the query, and two rows are zero.
    """
    g = Geometry(seed)
    a = int8_rows(g.e + TILT * g.eps * g.v)
    b = int8_rows(-g.e)
    fill = [int8_rows(-g.v + 0.05 * g.rng.standard_normal(DIM)) for _ in range(10)]
    zero = np.zeros(DIM, np.int8)
    rows = [*fill[:5], a, *fill[5:8], b, *fill[8:], zero, zero]
    stored, inverse = int8_store(np.stack(rows))
    return stored, inverse, g.query, g


def tie_store():
    """The exact tie of the inclusive comparison: two rows, two dimensions, k = 1, with every
    number a power of two or a sum of two, so ``E = eps_hat`` is exact (the caller sets the
    constants to 1 and 0). A (position 0) has ``c = a + E`` and B ``c = a - E``; both score
    ``+0.0``, so A wins by position, and ``fl(a_A + E) == fl(a_B - E)``: the threshold itself."""
    units = np.array([[0.0, 1.0]], F32)
    qcodes = np.array([[-1, 16]], np.int8)
    qscales = np.array([0.0625], F32)
    stored, inverse = int8_store(np.array([[1, 0], [-1, 0]]))
    return stored, inverse, units, qcodes, qscales


def unsafe_store(seed=1):
    """30 random rows around the top row (position 10) and a row U (position 21) whose
    inverse norm is 1000 times too large, whose direction is orthogonal to the query and
    along ``-e``: its score is small, its estimate about 7. Returns the store and the query."""
    g = Geometry(seed)
    y = -(g.e - (g.e @ (g.unit / np.linalg.norm(g.unit))) * g.unit / np.linalg.norm(g.unit))
    random_rows = [int8_rows(g.rng.standard_normal(DIM)) for _ in range(30)]
    rows = [*random_rows[:10], int8_rows(g.unit), *random_rows[10:20], int8_rows(y)]
    rows += random_rows[20:]
    stored, inverse = int8_store(np.stack(rows))
    inverse = inverse.copy()
    inverse[21] = F32(1000) * inverse[21]
    return stored, inverse, g.query, 21


def small_norm_query():
    """A query ``[2^-74, 2^-75, 0, ...]``: its squares underflow, ``normalize@1`` accepts it, and
    the unit vector has ``||u||^2 = 1.25`` (outside the proven domain)."""
    query = np.zeros((1, DIM), F32)
    query[0, 0], query[0, 1] = 2.0**-74, 2.0**-75
    return query


def random_store(rows, seed, dim=DIM):
    """Gaussian rows scaled to int8, for the tests of the whole search."""
    rng = np.random.default_rng(seed)
    return int8_store(int8_rows(rng.standard_normal((rows, dim))))


def tight_bounds(monkeypatch):
    """Constants of an exact bound (``E = eps_hat``), valid only where all arithmetic is exact."""
    monkeypatch.setattr(prefilter, "SLACK", 1.0)
    monkeypatch.setattr(prefilter, "ABSOLUTE", 0.0)
