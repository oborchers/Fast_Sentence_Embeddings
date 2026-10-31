#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The CPU backend on the native kernel (portable, AVX2 or NEON).

The second of the two modules that import fse._native: fse._text uses it for stages 1
and 2 (text to batches), this one for the kernels of stages 2 to 4. Every native entry
point sets the floating-point environment of its thread itself, so worker threads need
none. Several threads may call ``rows`` and ``store`` at once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from fse import _native
from fse._contract import Aggregation, Batch, Link, Op, Step

from ._base import (
    Capabilities,
    PoolInput,
    Prepared,
    check_logits_input,
    check_out,
    check_outputs_input,
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
    heads=True,
    max_dim=_native.MAX_DIM,
    text_rows=True,
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
        checked, out = dims(pool, steps)  # checks and materializes the steps once
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
        return Prepared(self.name, pool, checked, out, _State(kernel, self.isa))

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        check_prepared(prepared, self.name)
        check_out(out, batch, prepared)
        state = prepared.state
        assert isinstance(state, _State)
        state.kernel.pool(batch, out)
        if prepared.steps:  # normalize@1, in place
            _native.postprocess_pooled(out, prepared.steps, batch.first_row, isa=state.isa)

    def text_rows(
        self,
        text: object,
        span: tuple[int, int, int, int],
        after_encode: Callable[[], None] | None,
        lookup: _native.TextLookup,
        workspace: _native.Workspace,
        prepared: Prepared,
        out: ndarray,
    ) -> None:
        """Stages 1 to 4 of a range of text into ``out`` in one native call without the
        GIL in between (``_native.pool_text``): the rows, bits and errors of ``rows`` on
        the Batch that ``lookup.encode`` makes of it. ``span`` is (first, last, rows,
        first_row) of the range."""
        check_prepared(prepared, self.name)
        state = prepared.state
        assert isinstance(state, _State)
        first, last, n_rows, first_row = span
        _native.pool_text(
            state.kernel,
            lookup,
            workspace,
            text,
            first,
            last,
            n_rows,
            first_row,
            out,
            bool(prepared.steps),
            after_encode,
            isa=state.isa,
        )

    def steps(self, rows: ndarray, prepared: Prepared, first_row: int) -> None:
        """Stage 4 alone: the step of ``prepared`` (normalize, if any) in place on rows a
        device has pooled (finite, as pooling gives them; rows ``first_row`` on). The
        device backends run it on the host with this; the bits are those of ``rows``."""
        check_prepared(prepared, self.name)
        state = prepared.state
        assert isinstance(state, _State)
        if (
            not isinstance(rows, ndarray)
            or rows.dtype != np.float32
            or rows.ndim != 2
            or rows.shape[1] != prepared.output_dim
            or not rows.flags.c_contiguous
            or not rows.flags.writeable
        ):
            raise ValueError(
                "rows must be a writable C-contiguous float32 array of "
                f"{prepared.output_dim} values per row"
            )
        _native.postprocess_pooled(rows, prepared.steps, first_row, isa=state.isa)

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

    def logits(
        self,
        rows: ndarray,
        weights: ndarray | None,
        bias: ndarray,
        out: ndarray,
        first_row: int = 0,
    ) -> None:
        check_logits_input(rows, weights, bias, out, first_row)
        _native.head_logits(rows, weights, bias, out, int(first_row), isa=self.isa)

    def outputs(
        self, logits: ndarray, kind: str, link: Link | None, out: ndarray, first_row: int = 0
    ) -> None:
        check_outputs_input(logits, kind, link, out)
        _native.head_outputs(logits, kind, link, out, int(first_row), isa=self.isa)
