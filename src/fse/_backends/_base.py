#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The contract of a backend (docs/architecture/backends.md, "Backends").

A backend prepares the arrays of an encoder once (``prepare``), then turns one batch
into finished rows (``rows``: stages 2 to 4, the aggregation and then the postprocessing
step on the rows of that batch) and computes scores. Arguments and
outputs are host numpy arrays; what ``prepare`` returns holds the backend's own state,
and every result has the bits of ``fse._reference``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy import ndarray

from fse._contract import Aggregation, Batch, Link, Op, Step, check_steps

from ._base_checks import (
    DTYPES,
    check_link,
    float32_matrix,
    float32_vector,
    int64_vector,
    real,
    rows_of,
)


class UnsupportedByBackendError(ValueError):
    """The backend cannot run what it was asked to; names the backend and the capability."""


@dataclass(frozen=True, eq=False)
class PoolInput:
    """The arrays of one aggregation over word rows, as an encoder holds them.

    ``vectors`` (and ``ngram_vectors``) are the stored rows as float32, float16 or int8
    codes with ``scales`` (``ngram_scales``) for int8, ``weights`` one float32 weight per
    word, ``oov_weight`` the weight of unknown FastText words. ``check_pool_input`` checks
    them; the backends keep references, so the owner must not change them.
    """

    vectors: ndarray
    weights: ndarray
    scales: ndarray | None = None
    ngram_vectors: ndarray | None = None
    ngram_scales: ndarray | None = None
    oov_weight: float | None = None
    aggregation: Aggregation = Aggregation.WEIGHTED_MEAN

    @property
    def dtype(self) -> str:
        return DTYPES[self.vectors.dtype]

    @property
    def fasttext(self) -> bool:
        return self.ngram_vectors is not None

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


@dataclass(frozen=True)
class Capabilities:
    """What a backend can run.
    ``text_rows``: it runs stages 1 to 4 of a range of text in one call (``text_rows``),
    with the rows and errors of ``rows`` on the batch of that range."""

    word_dtypes: frozenset[str]
    aggregations: frozenset[Aggregation]
    steps: frozenset[Op]
    fasttext: bool
    max_dim: int
    text_rows: bool = False

    def check(self, backend: str, pool: PoolInput, steps: Sequence[Step]):
        """Raise UnsupportedByBackendError for the first thing ``backend`` cannot run."""
        missing = None
        if pool.dtype not in self.word_dtypes:
            missing = f"word vectors of {pool.dtype}"
        elif pool.aggregation not in self.aggregations:
            missing = f"the aggregation {pool.aggregation.id}"
        elif pool.fasttext and not self.fasttext:
            missing = "FastText ngrams"
        elif pool.dim > self.max_dim:
            missing = f"rows of {pool.dim} values (at most {self.max_dim})"
        else:
            missing = next((f"the step {s.op.id}" for s in steps if s.op not in self.steps), None)
        if missing is not None:
            raise UnsupportedByBackendError(f"the {backend} backend cannot run {missing}")


@dataclass(frozen=True, eq=False)
class Prepared:
    """What a backend prepared once for one encoder; ``state`` is the backend's own."""

    backend: str
    pool: PoolInput
    steps: tuple[Step, ...]
    output_dim: int
    state: object


class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared: ...

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        """Stages 2 to 4 of ``batch`` into ``out``, ``(n_sentences, output_dim)`` float32."""
        ...

    def score(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float
    ) -> ndarray: ...

    def float32(self, values: object) -> ndarray:
        """``values`` as a C-contiguous float32 array, rounded to nearest as the contract
        converts inputs (``fse._reference`` runs in the default environment)."""
        ...

    def affine64(
        self, weight: ndarray, bias: ndarray, factor: ndarray, shift: ndarray
    ) -> tuple[ndarray, ndarray]:
        """``factor[:, None] * weight`` and ``factor * bias + shift`` in float64 (two
        roundings for the bias), the arithmetic with which a calibration is folded into a
        head on import; the cpu backend runs it in the floating-point environment of the
        kernels."""
        ...

    def quotient64(self, weight: ndarray, bias: ndarray, divisor: float) -> tuple[ndarray, ndarray]:
        """``weight / divisor`` and ``bias / divisor`` in float64 (the temperature of a
        softmax folded into a head on import), in the environment of ``affine64``."""
        ...

    def inverse_norms(self, stored: ndarray, first_row: int = 0) -> ndarray:
        """The inverse norm of every stored row, as ``fse._reference.inverse_norms``;
        errors name the rows from ``first_row``."""
        ...

    def cosines(self, stored: ndarray, inverse: ndarray, queries: ndarray) -> ndarray:
        """The cosine of every stored row with every query, a (rows, queries) array, as
        ``fse._reference.cosines``."""
        ...

    def scores(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float | ndarray
    ) -> ndarray:
        """The linear score of every stored row with every weight vector, a (rows,
        vectors) array, as ``fse._reference.scores``; ``bias`` one number or one per
        vector."""
        ...

    def store(self, rows: ndarray, codes: ndarray, scales: ndarray | None, first_row: int) -> None:
        """Stage 5 of finished float32 ``rows`` (rows ``first_row`` on) into float16 or
        int8 ``codes``, with one scale per row in ``scales`` for int8 (``quantize_rows@1``)."""
        ...

    def logits(
        self,
        rows: ndarray,
        weights: ndarray | None,
        bias: ndarray,
        out: ndarray,
        first_row: int = 0,
    ) -> None:
        """The logits of ``linear@1`` into ``out`` (n, k), as ``fse._reference_head.logits``:
        with ``weights`` (k, d) ``dot16(row, a_j) + b_j`` of the float32 pooled ``rows``,
        without them ``rows`` (the pooled rows of a folded table, (n, k), which may be
        ``out``) plus ``bias``, the bias exactly once. A row whose logits are not finite
        raises, named from ``first_row``."""
        ...

    def outputs(
        self, logits: ndarray, kind: str, link: Link | None, out: ndarray, first_row: int = 0
    ) -> None:
        """The probabilities (``kind`` "proba", by ``link``: float32 (n, 2) for sigmoid, (n,
        k) else) or label indices ("labels", int64 (n,)) of finished ``logits`` into
        ``out``, as ``fse._reference_head.proba`` and ``labels``; no bias is added."""
        ...

    def decisions(self, probabilities: ndarray, thresholds: ndarray, out: ndarray) -> None:
        """Strict multilabel cutoffs into bool (n, k), preserving subnormal probabilities.

        The CPU backend ignores and restores the caller's floating-point environment.
        """
        ...


def check_pool_input(pool: PoolInput) -> PoolInput:
    """Check the arrays of ``pool`` against each other, as the native kernel does."""
    if not isinstance(pool, PoolInput):
        raise TypeError(f"pool must be a PoolInput, not {type(pool).__name__}")
    vectors = rows_of(pool.vectors, "vectors")
    if vectors.shape[1] < 1:
        raise ValueError("vectors need at least one dimension")
    int8 = vectors.dtype == np.int8
    float32_vector(pool.weights, "weights", len(vectors))
    if pool.oov_weight is not None:
        real(pool.oov_weight, "oov_weight")
    if int8 != (pool.scales is not None):
        raise ValueError("scales are given exactly for int8 vectors")
    if int8:
        float32_vector(pool.scales, "scales", len(vectors))
    _check_ngram_input(pool, vectors)
    return pool


def _check_ngram_input(pool: PoolInput, vectors: ndarray):
    """The FastText rows of ``pool`` against its word rows."""
    if pool.ngram_vectors is None:
        if pool.ngram_scales is not None:
            raise ValueError("ngram_scales without ngram_vectors")
        return
    ngrams = rows_of(pool.ngram_vectors, "ngram_vectors")
    if ngrams.dtype != vectors.dtype or ngrams.shape[1] != vectors.shape[1]:
        raise ValueError("ngram_vectors must have the dtype and dimension of vectors")
    int8 = vectors.dtype == np.int8
    if int8 != (pool.ngram_scales is not None):
        raise ValueError("ngram_scales are given exactly for int8 ngram vectors")
    if int8:
        float32_vector(pool.ngram_scales, "ngram_scales", len(ngrams))


def dims(pool: PoolInput, steps: Sequence[Step]) -> tuple[tuple[Step, ...], int]:
    """The checked steps and the dimension of the output, that of the aggregated rows (the
    only step, normalize, keeps it)."""
    out = pool.aggregation.output_dim(pool.dim)
    return check_steps(steps, out), out


def check_out(out: object, batch: Batch, prepared: Prepared) -> ndarray:
    """``out`` must be a writable C-contiguous float32 array of the rows of ``batch`` that
    shares no memory with the batch or the prepared arrays (checked before anything is
    written: a backend may pool into ``out`` and normalize there)."""
    shape = (batch.n_sentences, prepared.output_dim)
    if (
        not isinstance(out, ndarray)
        or out.dtype != np.float32
        or out.shape != shape
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError(f"out must be a writable C-contiguous float32 array of shape {shape}")
    pool = prepared.pool
    for kept in (pool.vectors, pool.weights, pool.scales, pool.ngram_vectors, pool.ngram_scales):
        if kept is not None and np.may_share_memory(out, kept):
            raise ValueError("out must not share memory with the vectors or weights")
    for kept in (batch.offsets, batch.tokens, batch.ngram_offsets, batch.ngrams):
        if kept is not None and np.may_share_memory(out, kept):
            raise ValueError("out must not share memory with the batch")
    return out


def check_logits_input(
    rows: object, weights: object, bias: object, out: object, first_row: object
) -> None:
    """The arguments of ``logits``, checked the same way by every backend: C-contiguous
    float32 rows, float32 weights (k, d) or None, a float32 bias of k values and a
    writable float32 ``out`` (n, k) that shares no memory with the weights or bias, and
    with the rows only if it is them (in place, without weights)."""
    if not isinstance(bias, ndarray) or bias.dtype != np.float32 or bias.ndim != 1 or not len(bias):
        raise ValueError("bias must be a float32 vector of at least one value")
    k = len(bias)
    float32_matrix(rows, "rows", None, k if weights is None else None)
    assert isinstance(rows, ndarray)
    if weights is not None:
        float32_matrix(weights, "weights", k, rows.shape[1])
    float32_matrix(out, "out", len(rows), k, writable=True)
    for kept in (weights, bias):
        if kept is not None and np.may_share_memory(out, kept):
            raise ValueError("out must not share memory with the weights or the bias")
    if np.may_share_memory(out, rows) and (weights is not None or out is not rows):
        raise ValueError("out may be the rows themselves (without weights), else apart from them")
    if isinstance(first_row, bool) or not isinstance(first_row, int | np.integer) or first_row < 0:
        raise ValueError(f"first_row must be a non-negative int, not {first_row!r}")


def check_outputs_input(logits: object, kind: object, link: object, out: object) -> None:
    """The arguments of ``outputs``: finished float32 logits (n, k), ``kind`` "proba"
    with a link that fits k (sigmoid for k = 1, softmax or ovr for k >= 2) into float32
    (n, 2 or k), or "labels" into int64 (n,), ``out`` apart from the logits."""
    float32_matrix(logits, "logits", None, None)
    assert isinstance(logits, ndarray)
    n, k = logits.shape
    if kind == "labels":
        int64_vector(out, "out", n)
    elif kind == "proba":
        check_link(link, k)
        float32_matrix(
            out, "out", n, 2 if k == 1 and link is not Link.MULTILABEL else k, writable=True
        )
    else:
        raise ValueError(f"kind must be 'proba' or 'labels', not {kind!r}")
    if np.may_share_memory(out, logits):
        raise ValueError("out must not share memory with the logits")


def check_decisions_input(probabilities: object, thresholds: object, out: object) -> None:
    """C-contiguous F32 probabilities and cutoffs, with a separate writable bool result."""
    float32_matrix(probabilities, "probabilities", None, None)
    assert isinstance(probabilities, ndarray)
    float32_vector(thresholds, "thresholds", probabilities.shape[1])
    assert isinstance(thresholds, ndarray)
    if (
        probabilities.shape[1] < 1
        or not isinstance(out, ndarray)
        or out.dtype != np.bool_
        or out.shape != probabilities.shape
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError("out must be a writable C-contiguous bool matrix matching probabilities")
    if any(np.may_share_memory(out, kept) for kept in (probabilities, thresholds)):
        raise ValueError("out must not share memory with probabilities or thresholds")
    for name, values in (("probabilities", probabilities), ("thresholds", thresholds)):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} must be finite and in [0, 1]")


def check_prepared(prepared: object, backend: str) -> Prepared:
    if not isinstance(prepared, Prepared) or prepared.backend != backend:
        raise TypeError(f"prepared must be prepared by the {backend} backend")
    return prepared
