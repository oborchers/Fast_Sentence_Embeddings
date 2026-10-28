#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The device-neutral contract of a backend (ARCHITECTURE.md, "Backends").

A backend prepares the arrays of an encoder once (``prepare``), then turns one batch
into finished rows (``rows``: stages 2 to 4, the aggregation and then the postprocessing
steps on the rows of that batch) and computes Gram blocks and scores. Arguments and
outputs are host numpy arrays; what ``prepare`` returns holds the backend's own state,
which may live on a device. Every result has the bits of ``fse._reference``. ``store``
runs on the host in PR 5; a device backend may fuse it with ``rows`` later (PR 9).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy import ndarray

from fse._contract import Aggregation, Batch, Op, Step, check_steps, output_dim

_DTYPES = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float16): "float16",
    np.dtype(np.int8): "int8",
}


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
        return _DTYPES[self.vectors.dtype]

    @property
    def fasttext(self) -> bool:
        return self.ngram_vectors is not None

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


@dataclass(frozen=True)
class Capabilities:
    """What a backend can run. ``concurrent``: several threads may call ``rows`` at once."""

    word_dtypes: frozenset[str]
    aggregations: frozenset[Aggregation]
    steps: frozenset[Op]
    fasttext: bool
    concurrent: bool
    max_dim: int

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
    pooled_dim: int
    output_dim: int
    state: object


class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared: ...

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        """Stages 2 to 4 of ``batch`` into ``out``, ``(n_sentences, output_dim)`` float32."""
        ...

    def gram_block(self, rows: ndarray) -> ndarray:
        """``rows.T @ rows`` of one block of float32 rows, as ``fse._gram`` sums them."""
        ...

    def score(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float
    ) -> ndarray: ...

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


def check_pool_input(pool: PoolInput) -> PoolInput:
    """Check the arrays of ``pool`` against each other, as the native kernel does."""
    if not isinstance(pool, PoolInput):
        raise TypeError(f"pool must be a PoolInput, not {type(pool).__name__}")
    vectors = _rows(pool.vectors, "vectors")
    if vectors.shape[1] < 1:
        raise ValueError("vectors need at least one dimension")
    int8 = vectors.dtype == np.int8
    _float32_vector(pool.weights, "weights", len(vectors))
    if pool.oov_weight is not None:
        _real(pool.oov_weight, "oov_weight")
    if int8 != (pool.scales is not None):
        raise ValueError("scales are given exactly for int8 vectors")
    if int8:
        _float32_vector(pool.scales, "scales", len(vectors))
    _check_ngram_input(pool, vectors)
    return pool


def _check_ngram_input(pool: PoolInput, vectors: ndarray):
    """The FastText rows of ``pool`` against its word rows."""
    if pool.ngram_vectors is None:
        if pool.ngram_scales is not None:
            raise ValueError("ngram_scales without ngram_vectors")
        return
    ngrams = _rows(pool.ngram_vectors, "ngram_vectors")
    if ngrams.dtype != vectors.dtype or ngrams.shape[1] != vectors.shape[1]:
        raise ValueError("ngram_vectors must have the dtype and dimension of vectors")
    int8 = vectors.dtype == np.int8
    if int8 != (pool.ngram_scales is not None):
        raise ValueError("ngram_scales are given exactly for int8 ngram vectors")
    if int8:
        _float32_vector(pool.ngram_scales, "ngram_scales", len(ngrams))


def dims(pool: PoolInput, steps: Sequence[Step]) -> tuple[tuple[Step, ...], int, int]:
    """The checked steps, the dimension of the aggregated rows and that of the output."""
    pooled = pool.aggregation.output_dim(pool.dim)
    checked = check_steps(steps, pooled)
    return checked, pooled, output_dim(checked, pooled)


def check_out(out: object, batch: Batch, prepared: Prepared) -> ndarray:
    """``out`` must be a writable C-contiguous float32 array of the rows of ``batch`` that
    shares no memory with the batch or the prepared arrays."""
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


def check_score_input(stored: object, scales: object, weights: object, bias: object) -> None:
    """The arguments of ``score``, checked the same way by every backend: stored rows of a
    storage dtype, scales exactly for int8, one weight per dimension (any float dtype;
    each backend converts it to float32 as the reference does)."""
    stored = _rows(stored, "stored")
    if np.asarray(weights).dtype.kind not in "fiu":
        raise TypeError("weights must be numbers (float or int), not strings, objects or bools")
    if np.shape(weights) != (stored.shape[1],):
        raise ValueError(f"weights must hold {stored.shape[1]} values")
    int8 = stored.dtype == np.int8
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 rows")
    if int8:
        _float32_vector(scales, "scales", len(stored))
    _real(bias, "bias")


def check_stored_input(stored: object, first_row: object = 0) -> None:
    """The arguments of ``inverse_norms``: a C-contiguous matrix of stored rows and the
    row its first row is."""
    _rows(stored, "stored")
    if isinstance(first_row, bool) or not isinstance(first_row, int | np.integer) or first_row < 0:
        raise ValueError(f"first_row must be a non-negative int, not {first_row!r}")


def check_search_input(stored: object, inverse: object, queries: object) -> None:
    """The arguments of ``cosines``, checked the same way by every backend: stored rows,
    one float32 inverse norm per row, a matrix of queries of the stored dimension."""
    stored = _rows(stored, "stored")
    _float32_vector(inverse, "inverse", len(stored))
    _vectors(queries, "queries", stored.shape[1])


def check_scores_input(stored: object, scales: object, weights: object, bias: object) -> None:
    """The arguments of ``scores``: as for ``score``, with a matrix of weight vectors and
    one bias for all or one per vector."""
    stored = _rows(stored, "stored")
    weights = _vectors(weights, "weights", stored.shape[1])
    int8 = stored.dtype == np.int8
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 rows")
    if int8:
        _float32_vector(scales, "scales", len(stored))
    if np.ndim(bias) == 0:
        _real(bias, "bias")
    elif np.asarray(bias).dtype.kind not in "fiu" or np.shape(bias) != (len(weights),):
        raise ValueError(f"bias must be one number or {len(weights)} numbers, one per vector")


def _vectors(value: object, name: str, dim: int) -> ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be numbers (float or int), not strings, objects or bools")
    if value.ndim != 2 or value.shape[1] != dim:
        raise ValueError(
            f"{name} must be a matrix of vectors of {dim} values, not shape {value.shape}"
        )
    return value


def check_store_input(rows: object, codes: object, scales: object, first_row: object) -> None:
    """The arguments of ``store``, checked the same way by every backend: C-contiguous
    float32 rows, writable C-contiguous float16 or int8 codes of their shape, a writable
    float32 scale per row exactly for int8, none sharing memory with another."""
    if not isinstance(rows, ndarray) or rows.dtype != np.float32 or rows.ndim != 2:
        raise ValueError("rows must be a float32 matrix")
    if not rows.flags.c_contiguous:
        raise ValueError("rows must be C-contiguous")
    if (
        not isinstance(codes, ndarray)
        or codes.dtype not in (np.float16, np.int8)
        or codes.shape != rows.shape
        or not codes.flags.c_contiguous
        or not codes.flags.writeable
    ):
        raise ValueError(
            f"codes must be a writable C-contiguous float16 or int8 array of shape {rows.shape}"
        )
    _check_store_scales(scales, codes.dtype == np.int8, len(rows))
    arrays = [a for a in (rows, codes, scales) if a is not None]
    if any(np.may_share_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :]):
        raise ValueError("rows, codes and scales must not share memory")
    if isinstance(first_row, bool) or not isinstance(first_row, int | np.integer) or first_row < 0:
        raise ValueError(f"first_row must be an int of at least 0, not {first_row!r}")


def _check_store_scales(scales: object, int8: bool, length: int):
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 codes")
    if int8:
        _float32_vector(scales, "scales", length)
        assert isinstance(scales, ndarray)
        if not scales.flags.writeable:
            raise ValueError("scales must be writable")


def check_prepared(prepared: object, backend: str) -> Prepared:
    if not isinstance(prepared, Prepared) or prepared.backend != backend:
        raise TypeError(f"prepared must be prepared by the {backend} backend")
    return prepared


def _rows(value: object, name: str) -> ndarray:
    if not isinstance(value, ndarray) or value.ndim != 2 or value.dtype not in _DTYPES:
        raise ValueError(f"{name} must be a matrix of float32, float16 or int8 rows")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _float32_vector(value: object, name: str, length: int):
    if not isinstance(value, ndarray) or value.dtype != np.float32 or value.shape != (length,):
        raise ValueError(f"{name} must be a float32 vector of {length} values")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _real(value: object, name: str):
    """A real number, Python or numpy; arrays, strings and bools are refused."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
