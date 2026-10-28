#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The numpy backend: the reference behind the backend contract.

It computes with ``fse._reference`` and is slow (a Python loop per token); it is the
second implementation of the contract and the oracle of the runtime tests. Like the
reference it runs in the floating-point environment of the calling thread.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from fse import _gram, _reference
from fse._contract import Aggregation, Batch, Op, Step

from ._base import (
    Capabilities,
    PoolInput,
    Prepared,
    check_out,
    check_pool_input,
    check_prepared,
    dims,
)

_CAPABILITIES = Capabilities(
    word_dtypes=frozenset({"float32", "float16", "int8"}),
    aggregations=frozenset({Aggregation.WEIGHTED_MEAN}),
    steps=frozenset(Op),
    fasttext=True,
    concurrent=True,
    max_dim=1 << 30,
)


class NumpyBackend:
    name = "numpy"

    def capabilities(self) -> Capabilities:
        return _CAPABILITIES

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared:
        check_pool_input(pool)
        _CAPABILITIES.check(self.name, pool, steps)
        checked, pooled, out = dims(pool, steps)
        return Prepared(self.name, pool, checked, pooled, out, None)

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        check_prepared(prepared, self.name)
        check_out(out, batch, prepared)
        p = prepared.pool
        rows = _reference.pool_weighted_mean(
            batch, p.vectors, p.weights, p.scales, p.ngram_vectors, p.ngram_scales, p.oov_weight
        )
        if prepared.steps:
            rows = _reference.postprocess(rows, prepared.steps, batch.first_row)
        out[...] = rows

    def gram_block(self, rows: ndarray) -> ndarray:
        return _gram.gram_block_product(_float32_rows(rows))

    def score(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float
    ) -> ndarray:
        return _reference.score(stored, scales, weights, bias)


def _float32_rows(rows: object) -> ndarray:
    if not isinstance(rows, ndarray) or rows.dtype != np.float32 or rows.ndim != 2:
        raise ValueError("rows must be a float32 matrix")
    return rows
