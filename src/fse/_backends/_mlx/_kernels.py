#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The pooling kernel of the mlx backend: its arrays, its launch and its canary.

``subnormal_input`` tells whether an encoder's inputs are or may give subnormal factors
or row values (then the host runs it), ``table`` uploads the word rows of an encoder
once, ``task`` turns a batch into the host arrays of one launch (or None when the CPU
must report the batch's error), ``launch`` queues the kernel of one task and ``read``
copies its rows out and gives the flags of its sentences. Every function that calls MLX
runs in the device thread (``_device``); the others are plain numpy and run in the
worker.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import OOV_TOKEN, Batch

from .._base import PoolInput
from . import _sources

# Up to 256 threads (8 SIMD groups) of a threadgroup along a row, the rest in sentences.
THREADGROUP = 256
SIMD = 32
_LOCK = threading.Lock()
_KERNELS: dict[str, Any] = {}


def kernel(mx: ModuleType, vec: int) -> Any:
    """The pooling kernel with ``vec`` values per thread (4 or 1), built once per process
    (MLX compiles each instance of its templates once)."""
    with _LOCK:
        built = _KERNELS.get(f"pool{vec}")
        if built is None:
            built = _KERNELS[f"pool{vec}"] = mx.fast.metal_kernel(
                name=f"fse_pool{vec}",
                input_names=list(_sources.POOL_INPUTS),
                output_names=list(_sources.POOL_OUTPUTS),
                source=_sources.pool(vec),
                header=_sources.HEADER,
                compile_options={"math_mode": "safe"},
            )
        return built


CHECK_VALUES = 1 << 20  # values a check looks at at once, to bound its temporary memory


def _subnormal(values: Any) -> bool:
    """Whether a float32 value is subnormal, on its bit patterns (no comparison of
    floats, which a thread that treats subnormal values as zero would get wrong)."""
    bits = np.ascontiguousarray(values, np.float32).reshape(-1).view(np.uint32)
    for begin in range(0, len(bits), CHECK_VALUES):
        part = bits[begin : begin + CHECK_VALUES] & np.uint32(0x7FFFFFFF)
        if ((part != 0) & (part < 0x00800000)).any():
            return True
    return False


SMALLEST_NORMAL = 2.0**-126
MAX_BUCKETS = 1 << 24  # more buckets of one unknown word go to the host (LONG)
# K, the tokens and buckets of a sentence on the device, is below 2**24: the term
# gamma_K of the bound is undefined from there (the host computes such a sentence).
MAX_CONTRIBUTIONS = (1 << 24) - 1


def _small_products(a: ndarray, b: ndarray) -> bool:
    """Whether a nonzero float32 product ``a * b`` (elementwise) may be subnormal: its
    exact value, in float64 (where it cannot be subnormal, so no floating-point
    environment changes it), is below 2**-126."""
    products = np.abs(np.asarray(a, np.float64) * np.asarray(b, np.float64))
    return bool(((products != 0) & (products < SMALLEST_NORMAL)).any())


def subnormal_input(pool: PoolInput, oov_weight: ndarray | None) -> bool:
    """Whether a stage-2 factor or row value of ``pool`` is or may be subnormal, as the cpu
    backend computes it: a subnormal weight, scale, oov weight or float32 row value; an
    int8 factor ``weight * scale``; or an ngram factor ``(1 / m) * oov_weight`` (times
    each ngram scale for int8) for any m up to 2**24: ``1 / m`` rounds to at least 2**-24
    and rounding is monotonic, so no factor is smaller than the exact products with
    2**-24. float16 values widen to normal float32 values."""
    values = [pool.weights, *(a for a in (pool.scales, pool.ngram_scales) if a is not None)]
    rows = (pool.vectors, pool.ngram_vectors)
    values += [a for a in rows if a is not None and a.dtype == np.float32]
    if oov_weight is not None:
        values.append(oov_weight)
    if any(_subnormal(a) for a in values):
        return True
    if pool.scales is not None and _small_products(pool.weights, pool.scales):
        return True
    if oov_weight is None or pool.ngram_vectors is None:
        return False
    shared = abs(float(oov_weight[0])) / MAX_BUCKETS  # the smallest (1 / m) * oov_weight
    if shared != 0 and shared < SMALLEST_NORMAL:
        return True
    return pool.ngram_scales is not None and _small_products(
        np.full(len(pool.ngram_scales), shared), pool.ngram_scales
    )


@dataclass(frozen=True, eq=False)
class Table:
    """The arrays of one PoolInput on the device, evaluated before they are shared."""

    arrays: tuple  # vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight
    dim: int
    n_words: int
    n_buckets: int
    int8: bool
    fasttext: bool
    oov_weight: bool  # whether the encoder has one
    norms: tuple = ()  # the norms of the word and of the ngram rows, rounded up

    @property
    def vec(self) -> int:
        """The values per thread of the pooling kernel: 4 (aligned rows) or 1."""
        return 4 if self.dim % 4 == 0 else 1


def table(mx: ModuleType, pool: PoolInput, oov_weight: ndarray | None) -> Table:
    """Upload the rows of ``pool``; ``oov_weight`` is its float32 value (as the CPU
    converts it) or None. Arrays the kernel does not read are one value each."""
    int8, fasttext = pool.scales is not None, pool.ngram_vectors is not None
    dummy = np.zeros(1, np.float32)
    arrays = tuple(
        mx.array(a)
        for a in (
            pool.vectors,
            pool.weights,
            pool.scales if int8 else dummy,
            pool.ngram_vectors if fasttext else np.zeros((1, 4), pool.vectors.dtype),
            pool.ngram_scales if fasttext and int8 else dummy,
            dummy if oov_weight is None else oov_weight.reshape(1),
        )
    )
    ngram_norms = dummy if pool.ngram_vectors is None else _row_norms(pool.ngram_vectors)
    norms = (mx.array(_row_norms(pool.vectors)), mx.array(ngram_norms))
    mx.eval(*arrays, *norms)
    buckets = len(pool.ngram_vectors) if pool.ngram_vectors is not None else 0
    return Table(
        arrays,
        pool.dim,
        len(pool.vectors),
        buckets,
        int8,
        fasttext,
        oov_weight is not None,
        norms,
    )


def _row_norms(rows: ndarray) -> ndarray:
    """The Euclidean norm of every stored row (int8 codes as they are, their scale is in
    the factor), in float64 from the exactly widened values, rounded up to float32."""
    out = np.empty(len(rows), np.float64)
    step = max(1, CHECK_VALUES // max(rows.shape[1], 1))
    for begin in range(0, len(rows), step):
        part = rows[begin : begin + step].astype(np.float64)
        out[begin : begin + step] = np.sqrt(np.einsum("ij,ij->i", part, part)) * (1 + 2.0**-50)
    with np.errstate(over="ignore"):  # a norm beyond float32: infinite, a row to the host
        norms = out.astype(np.float32)
        low = np.isfinite(norms) & (norms.astype(np.float64) < out)
        norms[low] = np.nextafter(norms[low], np.float32(np.inf))
    return norms


@dataclass(frozen=True, eq=False)
class Task:
    """The host arrays of one launch: the tokens, offsets, the index of the first unknown
    word of every sentence, and the ngram offsets and buckets (one value each without),
    all uint32 (a task holds fewer than 2**32 tokens and buckets)."""

    n_sentences: int
    first_row: int
    arrays: tuple
    unknown: bool

    @property
    def tokens(self) -> int:
        """The tokens and ngram buckets the kernel reads, the size of its work."""
        return len(self.arrays[0]) + (len(self.arrays[4]) if self.unknown else 0)


_NO_OFFSETS = np.zeros(1, np.uint32)
_NO_TOKENS = np.zeros(1, np.uint32)
LIMIT = (1 << 32) - 1  # the most tokens and buckets of a task, counted by uint32 offsets


def size(batch: Batch) -> int:
    """The tokens and ngram buckets of ``batch``, the size of its task."""
    return len(batch.tokens) + (len(batch.ngrams) if batch.ngrams is not None else 0)


def sizes(batch: Batch) -> ndarray:
    """The tokens and ngram buckets of every sentence of ``batch``."""
    counts = np.diff(batch.offsets)
    if batch.ngrams is None or not len(batch.ngrams):
        return counts
    unknown = np.concatenate([[0], np.cumsum(batch.tokens == OOV_TOKEN, dtype=np.int64)])
    first = unknown[batch.offsets]  # the index of the first unknown word of each sentence
    return counts + np.diff(np.asarray(batch.ngram_offsets)[first])


def split(batch: Batch, budget: int, limit: int) -> list[tuple[int, int, Batch | None]]:
    """``batch`` in runs of whole sentences of at most ``budget`` tokens and buckets, as
    ``(begin, end, sub-batch)``: a sentence of more than ``budget`` alone, and one of more
    than ``limit`` alone with None (the host's)."""
    each = sizes(batch)
    parts, begin, total = [], 0, 0
    for s, n in enumerate(each.tolist()):
        if n > budget:
            parts += [(begin, s), (s, s + 1)] if s > begin else [(s, s + 1)]
            begin, total = s + 1, 0
            continue
        if total + n > budget:
            parts.append((begin, s))
            begin, total = s, 0
        total += n
    if begin < len(each):
        parts.append((begin, len(each)))
    return [(b, e, None if each[b:e].max() > limit else _slice(batch, b, e)) for b, e in parts]


def _slice(batch: Batch, begin: int, end: int) -> Batch:
    """Sentences ``begin`` to ``end`` of ``batch`` as a batch of their own."""
    lo, hi = int(batch.offsets[begin]), int(batch.offsets[end])
    offsets = batch.offsets[begin : end + 1] - lo
    tokens = batch.tokens[lo:hi]
    first_row = batch.first_row + begin
    if batch.ngrams is None:
        return Batch(first_row, offsets, tokens)
    assert batch.ngram_offsets is not None
    k0 = int(np.count_nonzero(batch.tokens[:lo] == OOV_TOKEN))
    k1 = k0 + int(np.count_nonzero(tokens == OOV_TOKEN))
    ngram_offsets = batch.ngram_offsets[k0 : k1 + 1]
    b0, b1 = int(ngram_offsets[0]), int(ngram_offsets[-1])
    return Batch(first_row, offsets, tokens, ngram_offsets - b0, batch.ngrams[b0:b1])


def _offsets(offsets: ndarray) -> ndarray:
    return offsets.astype(np.uint32)


def task(batch: Batch, table: Table) -> Task | None:
    """The arrays of ``batch`` (of at most LIMIT tokens and buckets, ``size``) for the
    kernel, or None if the CPU reports an error for it: a token or bucket beyond the rows,
    or an unknown word the encoder cannot add. An index out of range would read other
    memory on the GPU without any error."""
    if size(batch) > LIMIT:
        raise ValueError(f"a task holds at most {LIMIT} tokens and buckets")
    tokens = batch.tokens
    unknown = tokens == OOV_TOKEN
    any_unknown = bool(unknown.any())
    known = tokens[~unknown] if any_unknown else tokens
    if len(known) and int(known.max()) >= table.n_words:
        return None
    if not any_unknown:
        arrays = (tokens, _offsets(batch.offsets), _NO_OFFSETS, _NO_OFFSETS, _NO_TOKENS)
        return Task(batch.n_sentences, batch.first_row, arrays, False)
    ngrams, ngram_offsets = batch.ngrams, batch.ngram_offsets
    if not (table.fasttext and table.oov_weight) or ngrams is None or ngram_offsets is None:
        return None
    if len(ngrams) and int(ngrams.max()) >= table.n_buckets:
        return None
    before = np.concatenate([[0], np.cumsum(unknown, dtype=np.int64)])[batch.offsets[:-1]]
    flat = ngrams if len(ngrams) else _NO_TOKENS
    arrays = (
        tokens,
        _offsets(batch.offsets),
        before.astype(np.uint32),
        _offsets(ngram_offsets),
        flat,
    )
    return Task(batch.n_sentences, batch.first_row, arrays, True)


def shape(dim: int, vec: int, rows: int) -> tuple[tuple, tuple, int]:
    """The grid and threadgroup of the pooling kernel for ``rows`` sentences, and the flag
    words per sentence: the threads of a row padded to SIMD groups, at most THREADGROUP of
    them per group (several groups along a long row), the rest of the group in sentences."""
    padded = -(-(dim // vec) // SIMD) * SIMD
    x = min(THREADGROUP, padded)
    width = -(-padded // x) * x
    y = max(1, min(THREADGROUP // x, rows))
    return (width, rows, 1), (x, y, 1), width // SIMD


def normalizes(table: Table) -> bool:
    """Whether the kernel can normalize the rows of ``table``: a row within one
    threadgroup."""
    return -(-(table.dim // table.vec) // SIMD) * SIMD <= THREADGROUP


def launch(mx: ModuleType, stream: Any, table: Table, task: Task, normalize: bool = False):
    """The pooling kernel of ``task``, queued on ``stream``: its rows (normalized with
    ``normalize``, which needs ``normalizes(table)``), flags, and per row the sums of
    ``|f| ||x||`` and (with ``normalize``) of squares (not evaluated)."""
    assert not normalize or normalizes(table)
    grid, group, words = shape(table.dim, table.vec, task.n_sentences)
    inputs = [*table.arrays, *(mx.array(a) for a in task.arrays), *table.norms]
    n = task.n_sentences
    return tuple(
        kernel(mx, table.vec)(
            inputs=inputs,
            template=[
                ("T", table.arrays[0].dtype),
                ("DIM", table.dim),
                ("FLAG_WORDS", words),
                ("INT8", table.int8),
                ("FASTTEXT", table.fasttext and task.unknown),
                ("NORMALIZE", normalize),
            ],
            grid=grid,
            threadgroup=group,
            output_shapes=[(n, table.dim), (n, words), (n, 2)],
            output_dtypes=[mx.float32, mx.uint32, mx.float32],
            stream=stream,
        )
    )


def view(launched: tuple) -> tuple[ndarray, ndarray, ndarray]:
    """A view of the evaluated rows of a launch (no copy: it keeps the MLX array alive, so
    the device thread lends it until the caller has copied it), the flags of every
    sentence (the OR of its words' flags) and a copy of its bounds."""
    rows, flags, bounds = launched
    # A copy: with one flag word, reduce would give a view of the MLX array back.
    combined = np.array(functools.reduce(np.bitwise_or, np.asarray(flags).T), copy=True)
    return np.asarray(rows), combined, np.array(bounds)


def read(launched: tuple, target: ndarray) -> ndarray:
    """Copy the evaluated rows of a launch into ``target``; the flags of its sentences."""
    rows, flags, _ = view(launched)
    np.copyto(target, rows)
    return flags


ALWAYS = _sources.NONFINITE | _sources.LONG


def flagged(flags: ndarray) -> ndarray:
    """The sentences the host computes: a value that is not finite, a count that would
    round."""
    return (flags & ALWAYS) != 0


def reason(flags: int) -> str:
    """Why rows go to the host (``flags``, the OR of theirs): ``long``, ``nonfinite`` or
    ``limit`` (no flag: a row near a limit of a step, found by ``_bounds``)."""
    if flags & _sources.LONG:
        return "long"
    return "nonfinite" if flags & _sources.NONFINITE else "limit"
