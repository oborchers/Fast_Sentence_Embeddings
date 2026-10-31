#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The numpy reference of linear heads: ``linear@1``, ``exp@1`` and the links.

docs/architecture/numerical-contract.md, "Numerical contract (`transform_version` 2)", defines every rule here; this module computes
them bit for bit in plain numpy, as ``fse._reference`` does for the encoder, and is the
oracle of every CPU backend. A head ``z = A x + b`` (k rows of A, k biases) either folds
into the table (``fold``: every word and ngram row becomes ``score(row, a_j, 0)``, the
folded rows are pooled and ``b`` is added once after pooling) or reads the pooled row,
raw or normalized (``logits`` with A: ``score(row, a_j, b_j)``). The links work on
finished logits and never add the bias.

``exp@1`` is defined by its operations, so numpy and C give the same bits: ``k = rint(x *
log2e)``, ``r = fma(-k, ln2_hi, x)``, ``r = fma(-k, ln2_lo, r)``, a polynomial of degree 7
(the Taylor coefficients 1/i! rounded to float32) by Horner with fused multiply-adds,
then ``(p * 2**(k + 64)) * 2**-64``: the first product is exact, the second rounds once,
also into the subnormal range. Below -104 the result is +0.0. Callers pass only x <= 0
(``z - max`` for softmax, ``-|z|`` for sigmoid), so nothing overflows.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from fse._contract import Link
from fse._reference import dot16, fma32, scores

_F32 = np.float32


def _hex(text: str) -> np.float32:
    return _F32(float.fromhex(text))


LOG2E = _hex("0x1.715476p+0")
LN2_HI = _hex("0x1.62e430p-1")
LN2_LO = _hex("-0x1.05c610p-29")
# The coefficients of the polynomial, constant term first.
EXP_POLY = tuple(
    _hex(c)
    for c in (
        "0x1p+0",
        "0x1p+0",
        "0x1p-1",
        "0x1.555556p-3",
        "0x1.555556p-5",
        "0x1.111112p-7",
        "0x1.6c16c2p-10",
        "0x1.a01a02p-13",
    )
)
# Below this the result is +0.0 (exp(-104) is less than half the smallest subnormal).
EXP_LOW = _F32(-104.0)


def _pow2(exponent: ndarray) -> ndarray:
    """2**e as float32 for integer e from -126 to 127, built from its bits (exact)."""
    return ((exponent.astype(np.int32) + 127) << 23).astype(np.int32).view(_F32)


def exp(x) -> ndarray:
    """``exp@1`` of every value of ``x`` (float32, at most 0), elementwise."""
    x = np.asarray(x, dtype=_F32)
    with np.errstate(over="ignore", invalid="ignore"):
        low = ~(x >= EXP_LOW)
        k = np.where(low, _F32(0), np.rint(x * LOG2E)).astype(_F32)
        r = fma32(-k, LN2_HI, x)
        r = fma32(-k, LN2_LO, r)
        p = np.full(r.shape, EXP_POLY[-1], dtype=_F32)
        for c in EXP_POLY[-2::-1]:
            p = fma32(p, r, c)
        scaled = (p * _pow2(k + 64)).astype(_F32)
        out = (scaled * _pow2(np.full(k.shape, -64))).astype(_F32)
    return np.where(low, _F32(0), out).astype(_F32)


def sigmoid(z) -> ndarray:
    """``sigmoid@1``: the probability of the positive class, elementwise. With ``e =
    exp@1(-|z|)``: ``1 / (1 + e)`` for z >= 0 (-0.0 included), else ``e / (1 + e)``."""
    z = np.asarray(z, dtype=_F32)
    e = exp(-np.abs(z))
    d = (_F32(1) + e).astype(_F32)
    return np.where(z >= 0, _F32(1) / d, e / d).astype(_F32)


def _index_sum(values: ndarray) -> ndarray:
    """The sum of every row of ``values`` in index order, from +0.0."""
    total = np.zeros(len(values), dtype=_F32)
    for j in range(values.shape[1]):
        total = (total + values[:, j]).astype(_F32)
    return total


def softmax(z) -> ndarray:
    """``softmax@1`` of every row: ``e_j = exp@1(z_j - max)``, divided by their sum in
    index order."""
    z = np.asarray(z, dtype=_F32)
    with np.errstate(over="ignore", invalid="ignore"):
        e = exp((z - z.max(axis=1, keepdims=True)).astype(_F32))
    return (e / _index_sum(e)[:, None]).astype(_F32)


def ovr(z) -> ndarray:
    """``ovr@1`` of every row: ``sigmoid@1`` of every logit, divided by their sum in index
    order; a row whose sum is 0 becomes uniform (``1 / k``), as in scikit-learn."""
    s = sigmoid(z)
    total = _index_sum(s)
    zero = total == 0
    uniform = _F32(1) / _F32(s.shape[1])
    divided = (s / np.where(zero, _F32(1), total)[:, None]).astype(_F32)
    return np.where(zero[:, None], uniform, divided).astype(_F32)


def proba(logits, link: Link) -> ndarray:
    """The probabilities of finished logits (n, k): (n, 2) ``[1 - p, p]`` of ``sigmoid@1``
    for k = 1, (n, 2) ``softmax@1`` of ``[-z, z]`` for one logit with softmax (scikit-learn's
    binary multinomial model), else (n, k) of ``softmax@1`` or ``ovr@1``."""
    z = np.asarray(logits, dtype=_F32)
    if link is Link.SIGMOID:
        if z.shape[1] != 1:
            raise ValueError("sigmoid@1 takes one logit per row")
        p = sigmoid(z[:, 0])
        return np.stack([(_F32(1) - p).astype(_F32), p], axis=1)
    if link is Link.SOFTMAX and z.shape[1] == 1:
        return softmax(np.concatenate([-z, z], axis=1))
    if z.shape[1] < 2:
        raise ValueError(f"{link.id} takes at least two logits per row")
    return softmax(z) if link is Link.SOFTMAX else ovr(z)


def labels(logits) -> ndarray:
    """The index of the label of every row (int64): for k = 1, 1 if ``z > 0`` else 0
    (scikit-learn's rule, so -0.0 and 0.0 give 0); for k >= 2 the first largest logit."""
    z = np.asarray(logits, dtype=_F32)
    if z.shape[1] == 1:
        return (z[:, 0] > 0).astype(np.int64)
    return np.argmax(z, axis=1).astype(np.int64)


def fold(stored: ndarray, scales: ndarray | None, weight: ndarray) -> ndarray:
    """The folded table of stored word (or ngram) rows: ``score(row, scale, a_j, 0)`` for
    every row and head row j, a (rows, k) float32 array. Raises if a value overflows."""
    return scores(stored, scales, weight, 0.0)


def logits(rows, weight: ndarray | None, bias, first_row: int = 0) -> ndarray:
    """The logits of pooled rows. With ``weight`` (k, d): ``dot16(row, a_j) + b_j``, the
    rule of ``score`` on float32 rows; without it ``rows`` are pooled rows of the folded
    table (k columns) and ``b`` is added once. A row whose logits are not finite raises,
    named from ``first_row``."""
    rows = np.asarray(rows, dtype=_F32)
    bias = np.asarray(bias, dtype=_F32)
    with np.errstate(over="ignore", invalid="ignore"):
        if weight is None:
            out = (rows + bias[None, :]).astype(_F32)
        else:
            weight = np.asarray(weight, dtype=_F32)
            out = np.empty((len(rows), len(weight)), dtype=_F32)
            for j, a in enumerate(weight):
                out[:, j] = (dot16(rows, a) + bias[j]).astype(_F32)
    check_logits(out, first_row)
    return out


def check_logits(z: ndarray, first_row: int = 0) -> None:
    """Raise a ValueError naming the first row (from ``first_row``) whose logits are not
    finite."""
    bad = np.flatnonzero(~np.isfinite(z).all(axis=1))
    if len(bad):
        raise ValueError(f"the logits of row {first_row + int(bad[0])} are not finite")
