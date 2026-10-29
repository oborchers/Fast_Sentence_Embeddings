#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The CPU backend on the native kernel (portable, AVX2 or NEON).

The second of the two modules that import fse._native: fse._text uses it for stages 1
and 2 (text to batches), this one for the kernels of stages 2 to 4. Every native entry
point sets the floating-point environment of its thread itself, so worker threads need
none. Several threads may call ``rows`` and ``store`` at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from fse import _gram, _native
from fse._contract import Aggregation, Batch, Op, Step

from ._base import (
    Capabilities,
    PoolInput,
    Prepared,
    check_out,
    check_pool_input,
    check_prepared,
    check_score_input,
    check_scores_input,
    check_search_input,
    check_store_input,
    check_stored_input,
    dims,
)

_CAPABILITIES = Capabilities(
    word_dtypes=frozenset({"float32", "float16", "int8"}),
    aggregations=frozenset({Aggregation.WEIGHTED_MEAN}),
    steps=frozenset(Op),
    fasttext=True,
    concurrent=True,
    max_dim=_native.MAX_DIM,
)


@dataclass(frozen=True, eq=False)
class _State:
    kernel: _native.WeightedMean
    isa: str


class CpuBackend:
    name = "cpu"

    def __init__(self, isa: str | None = None):
        if isa is not None and isa not in _native.available_isas():
            raise ValueError(
                f"isa must be one of the kernels of this CPU, {_native.available_isas()}, not {isa!r}"
            )
        self.isa = _native.default_isa() if isa is None else isa

    def capabilities(self) -> Capabilities:
        return _CAPABILITIES

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared:
        check_pool_input(pool)
        checked, pooled, out = dims(pool, steps)  # checks and materializes the steps once
        _CAPABILITIES.check(self.name, pool, checked)
        kernel = _native.WeightedMean(
            pool.vectors,
            pool.weights,
            pool.scales,
            pool.ngram_vectors,
            pool.ngram_scales,
            pool.oov_weight,
            isa=self.isa,
        )
        return Prepared(self.name, pool, checked, pooled, out, _State(kernel, self.isa))

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        check_prepared(prepared, self.name)
        check_out(out, batch, prepared)
        state = prepared.state
        assert isinstance(state, _State)
        if not prepared.steps:
            state.kernel.pool(batch, out)
            return
        if prepared.pooled_dim == prepared.output_dim:  # no project: the steps in place
            state.kernel.pool(batch, out)
            _native.postprocess_pooled(out, prepared.steps, batch.first_row, isa=state.isa)
            return
        rows = state.kernel.pool(batch)  # a new array, finite as pooled
        out[...] = _native.postprocess_pooled(rows, prepared.steps, batch.first_row, isa=state.isa)

    def gram_block(self, rows: ndarray) -> ndarray:
        if not isinstance(rows, ndarray) or rows.dtype != np.float32 or rows.ndim != 2:
            raise ValueError("rows must be a float32 matrix")
        return _gram.gram_block_product(rows)

    def score(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float
    ) -> ndarray:
        check_score_input(stored, scales, weights, bias)
        return _native.score(stored, scales, weights, bias, isa=self.isa)

    def float32(self, values: object) -> ndarray:
        return _native.float32(values)

    def inverse_norms(self, stored: ndarray, first_row: int = 0) -> ndarray:
        check_stored_input(stored, first_row)
        return _native.inverse_norms(stored, int(first_row))

    def cosines(self, stored: ndarray, inverse: ndarray, queries: ndarray) -> ndarray:
        check_search_input(stored, inverse, queries)
        return _native.cosines(stored, inverse, queries, isa=self.isa)

    def scores(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float | ndarray
    ) -> ndarray:
        check_scores_input(stored, scales, weights, bias)
        return _native.scores(stored, scales, weights, bias, isa=self.isa)

    def store(self, rows: ndarray, codes: ndarray, scales: ndarray | None, first_row: int) -> None:
        check_store_input(rows, codes, scales, first_row)
        _native.store_rows(rows, codes, scales, int(first_row), isa=self.isa)
