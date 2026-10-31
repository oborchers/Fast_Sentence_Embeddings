#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The numpy backend: the reference behind the backend contract.

It computes with ``fse._reference`` and is slow (a Python loop per token); it is the
second implementation of the contract and the oracle of the runtime tests. Like the
reference it runs in the floating-point environment of the calling thread.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from fse import _reference, _reference_head
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
    heads=True,
    max_dim=1 << 30,
)


class NumpyBackend:
    name = "numpy"

    def capabilities(self) -> Capabilities:
        return _CAPABILITIES

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared:
        check_pool_input(pool)
        checked, out = dims(pool, steps)  # checks and materializes the steps once
        _CAPABILITIES.check(self.name, pool, checked)
        return Prepared(self.name, pool, checked, out, None)

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

    def score(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float
    ) -> ndarray:
        check_score_input(stored, scales, weights, bias)
        return _reference.score(stored, scales, weights, bias)

    def float32(self, values: object) -> ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.ascontiguousarray(values, dtype=np.float32)

    def inverse_norms(self, stored: ndarray, first_row: int = 0) -> ndarray:
        check_stored_input(stored, first_row)
        return _reference.inverse_norms(stored, int(first_row))

    def cosines(self, stored: ndarray, inverse: ndarray, queries: ndarray) -> ndarray:
        check_search_input(stored, inverse, queries)
        return _reference.cosines(stored, inverse, queries)

    def scores(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float | ndarray
    ) -> ndarray:
        check_scores_input(stored, scales, weights, bias)
        return _reference.scores(stored, scales, weights, bias)

    def store(self, rows: ndarray, codes: ndarray, scales: ndarray | None, first_row: int) -> None:
        check_store_input(rows, codes, scales, first_row)
        _check_storable(rows, codes.dtype, int(first_row))
        got, got_scales = _reference.quantize_rows(
            rows, "int8" if scales is not None else "float16"
        )
        codes[...] = got
        if scales is not None:
            scales[...] = got_scales

    def logits(
        self,
        rows: ndarray,
        weights: ndarray | None,
        bias: ndarray,
        out: ndarray,
        first_row: int = 0,
    ) -> None:
        check_logits_input(rows, weights, bias, out, first_row)
        out[...] = _reference_head.logits(rows, weights, bias, int(first_row))

    def outputs(
        self, logits: ndarray, kind: str, link: Link | None, out: ndarray, first_row: int = 0
    ) -> None:
        check_outputs_input(logits, kind, link, out)
        _reference_head.check_logits(logits, int(first_row))
        if kind == "labels":
            out[...] = _reference_head.labels(logits)
        else:
            assert link is not None
            out[...] = _reference_head.proba(logits, link)


def _check_storable(rows: ndarray, dtype: np.dtype, first_row: int):
    """The errors of the cpu backend (``fse._codes``), naming the first bad row."""
    bad = np.flatnonzero(~np.isfinite(rows).all(axis=1))
    if len(bad):
        raise ValueError(f"row {first_row + int(bad[0])} contains NaN or infinite values")
    if dtype == np.float16:
        top = float(np.finfo(np.float16).max)
        large = np.flatnonzero((np.abs(rows) > top).any(axis=1))
        if len(large):
            raise ValueError(
                f"row {first_row + int(large[0])} has values beyond the float16 range of +-{top:g}"
            )
