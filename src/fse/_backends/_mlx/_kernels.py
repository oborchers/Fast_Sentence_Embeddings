#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The kernels of the mlx backend: their arrays and their launch.

``table`` uploads the word rows of an encoder once and makes their factors and checks on
the GPU (``_sources.PREPARE``), ``task`` turns a batch into the host arrays of one launch
(or None when the CPU must report the batch's error), ``launch`` queues the pooling
kernel of one task and ``read`` copies its rows out and gives the OR of its flags. Every
function that calls MLX runs in the device thread (``_device``); ``task`` and ``reason``
are plain numpy and run in the worker. The canary is in ``_canary``.
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

from .._base import PoolInput, UnsupportedByBackendError
from . import _sources

# Up to 256 threads (8 SIMD groups) of a threadgroup along a row, the rest in sentences.
THREADGROUP = 256
SIMD = 32
PREPARE_ROWS = 8  # rows of a threadgroup of the prepare kernel, one SIMD group each
_LOCK = threading.Lock()
_KERNELS: dict[str, Any] = {}


def kernel(mx: ModuleType, name: str) -> Any:
    """The kernel ``name`` (``prepare``, ``pool4`` or ``pool1``), built once per process
    (MLX compiles each instance of its templates once)."""
    with _LOCK:
        built = _KERNELS.get(name)
        if built is None:
            inputs: tuple[str, ...]
            outputs: tuple[str, ...]
            if name == "prepare":
                source = _sources.PREPARE
                inputs, outputs = _sources.PREPARE_INPUTS, _sources.PREPARE_OUTPUTS
            else:
                source = _sources.pool(int(name[-1]))
                inputs, outputs = _sources.POOL_INPUTS, _sources.POOL_OUTPUTS
            built = _KERNELS[name] = mx.fast.metal_kernel(
                name=f"fse_{name}",
                input_names=list(inputs),
                output_names=list(outputs),
                source=source,
                header=_sources.HEADER,
                compile_options={"math_mode": "safe"},
            )
        return built


@dataclass(frozen=True, eq=False)
class Table:
    """The arrays of one PoolInput on the device, evaluated before they are shared, and
    the words whose factor the prepare kernel flagged (for the reason of a fallback)."""

    arrays: tuple  # vectors, factors, ngram_vectors, ngram_scales, ngram_low, oov_weight
    dim: int
    n_words: int
    n_buckets: int
    int8: bool
    fasttext: bool
    oov_weight: bool  # whether the encoder has one
    flagged: ndarray | None = None  # bool per word

    @property
    def vec(self) -> int:
        """The values per thread of the pooling kernel: 4 (aligned rows) or 1."""
        return 4 if self.dim % 4 == 0 else 1


def _prepare(mx: ModuleType, stream: Any, rows: Any, factors: tuple | None) -> Any:
    """With ``factors`` (the weights and the int8 scales or None), the factor bits of
    every word; without, the smallest exponent field of every ngram row. Made on the GPU,
    not evaluated."""
    n, dim = rows.shape
    dummy = mx.array(np.zeros(1, np.float32))
    weights, scales = factors or (dummy, None)
    words, int8 = factors is not None, scales is not None
    (out,) = kernel(mx, "prepare")(
        inputs=[rows, weights, dummy if scales is None else scales],
        template=[("DIM", dim), ("WORDS", words), ("INT8", int8)],
        grid=(SIMD, n, 1),
        threadgroup=(SIMD, PREPARE_ROWS, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.uint32],
        stream=stream,
    )
    return out


def table(mx: ModuleType, stream: Any, pool: PoolInput, oov_weight: ndarray | None) -> Table:
    """Upload the rows of ``pool`` and make its factors and ngram checks on the GPU;
    ``oov_weight`` is its float32 value (as the CPU converts it) or None. Arrays the
    kernel does not read are one value each."""
    int8, fasttext = pool.scales is not None, pool.ngram_vectors is not None
    dummy = mx.array(np.zeros(1, np.float32))
    vectors = mx.array(pool.vectors)
    scales = mx.array(pool.scales) if int8 else None
    factors = _prepare(mx, stream, vectors, (mx.array(pool.weights), scales))
    if fasttext:
        ngram_vectors = mx.array(pool.ngram_vectors)
        ngram_scales = mx.array(pool.ngram_scales) if int8 else dummy
        ngram_low = _prepare(mx, stream, ngram_vectors, None)
    else:
        ngram_vectors = mx.array(np.zeros((1, 4), pool.vectors.dtype))
        ngram_scales, ngram_low = dummy, mx.array(np.zeros(1, np.uint32))
    oov = dummy if oov_weight is None else mx.array(oov_weight.reshape(1))
    arrays = (vectors, factors, ngram_vectors, ngram_scales, ngram_low, oov)
    mx.eval(*arrays)
    # A NaN factor of a finite weight (and scale) is one the prepare kernel flagged.
    flagged = np.isnan(np.asarray(factors).view(np.float32)) & np.isfinite(pool.weights)
    if int8:
        flagged &= np.isfinite(pool.scales)
    buckets = len(pool.ngram_vectors) if pool.ngram_vectors is not None else 0
    return Table(
        arrays,
        pool.dim,
        len(pool.vectors),
        buckets,
        int8,
        fasttext,
        oov_weight is not None,
        flagged,
    )


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


def split(batch: Batch, limit: int) -> list[tuple[int, int, Batch | None]]:
    """``batch`` in runs of whole sentences of at most ``limit`` tokens and buckets, as
    ``(begin, end, sub-batch)``; a sentence beyond ``limit`` alone, with None."""
    each = sizes(batch)
    parts, begin, total = [], 0, 0
    for s, n in enumerate(each.tolist()):
        if n > limit:
            parts += [(begin, s), (s, s + 1)] if s > begin else [(s, s + 1)]
            begin, total = s + 1, 0
            continue
        if total + n > limit:
            parts.append((begin, s))
            begin, total = s, 0
        total += n
    if begin < len(each):
        parts.append((begin, len(each)))
    return [(b, e, None if each[b:e].sum() > limit else _slice(batch, b, e)) for b, e in parts]


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


def launch(mx: ModuleType, stream: Any, table: Table, task: Task) -> tuple:
    """The pooling kernel of ``task``, queued on ``stream``: its rows and flags (not
    evaluated)."""
    grid, group, words = shape(table.dim, table.vec, task.n_sentences)
    inputs = [*table.arrays, *(mx.array(a) for a in task.arrays)]
    return tuple(
        kernel(mx, f"pool{table.vec}")(
            inputs=inputs,
            template=[
                ("T", table.arrays[0].dtype),
                ("DIM", table.dim),
                ("FLAG_WORDS", words),
                ("INT8", table.int8),
                ("FASTTEXT", table.fasttext and task.unknown),
            ],
            grid=grid,
            threadgroup=group,
            output_shapes=[(task.n_sentences, table.dim), (task.n_sentences, words)],
            output_dtypes=[mx.float32, mx.uint32],
            stream=stream,
        )
    )


def read(launched: tuple, target: ndarray) -> ndarray:
    """Copy the evaluated rows of a launch into ``target``; the flags of every sentence:
    the OR of its words' flags, with TINY only if every word has it."""
    rows, flags = launched
    np.copyto(target, np.asarray(rows))
    words = np.asarray(flags).T
    tiny = np.uint32(_sources.TINY)
    return functools.reduce(np.bitwise_or, words & ~tiny) | functools.reduce(
        np.bitwise_and, words & tiny
    )


ALWAYS = _sources.SUBNORMAL | _sources.NONFINITE | _sources.LONG


def flagged(flags: ndarray, task: Task, normalize: bool) -> ndarray:
    """The sentences of ``task`` the host computes: a flag of the kernel, and before
    ``normalize`` a value that may overflow its squares or a row of contributions whose
    squares may be subnormal (or that is zero)."""
    bad = (flags & ALWAYS) != 0
    if normalize:
        nonempty = np.diff(task.arrays[1]) > 0
        bad |= ((flags & _sources.HUGE) != 0) | (((flags & _sources.TINY) != 0) & nonempty)
    return bad


def reason(flags: int, task: Task, table: Table) -> str:
    """Why rows of a task go to the host (``flags``, the OR of theirs): ``underflow`` (a
    value that is or may be subnormal, also a word the prepare kernel flagged), ``long``,
    ``nonfinite`` or ``limit`` (a value near a limit of ``normalize``)."""
    if flags & _sources.SUBNORMAL:
        return "underflow"
    tokens = task.arrays[0]
    known = tokens[tokens != OOV_TOKEN]
    if flags & _sources.NONFINITE and table.flagged is not None and table.flagged[known].any():
        return "underflow"
    if flags & _sources.LONG:
        return "long"
    return "nonfinite" if flags & _sources.NONFINITE else "limit"


# The canary: cases whose bits and flags the GPU must give, checked against fixed patterns
# (not against a reference computed in the floating-point environment of the caller).
# Bit patterns, not arithmetic: numpy would compute them in the floating-point environment
# of the importing thread (rounding up, or flushing 2**-149 to zero). Word 0 is
# f = 1 + 2**-12, word 1 the rounded -(f * f), word 2 is 1 + 2**-23, word 3 the smallest
# subnormal, words 4 and 5 are 1.25 * 2**-126 and -2**-126, word 6 the largest float32,
# word 7 is -(1 + 2**-23).
CANARY_ROWS = np.array(
    [0x3F800800, 0xBF801000, 0x3F800001, 0x1, 0x00A00000, 0x80800000, 0x7F7FFFFF, 0xBF800001],
    np.uint32,
).view(np.float32)
CANARY_WEIGHTS = np.array([0x3F800800, *[0x3F800000] * 7], np.uint32).view(np.float32)
# Sentence 0 keeps the rounding error of f * f only if the fma is fused; sentence 1 is
# (1 / 3) * acc, not acc / 3 (0x3F800001); sentence 2 has a subnormal value, sentence 3
# cancels to a subnormal sum, sentence 4 overflows (all flagged); sentence 5 cancels
# exactly to +0.0 (not flagged, but tiny for normalize).
CANARY_OFFSETS = np.array([0, 2, 5, 6, 8, 10, 12], np.int64)
CANARY_TOKENS = np.array([1, 0, 2, 2, 2, 3, 4, 5, 6, 6, 2, 7], np.uint32)
CANARY_BITS = (0x33000000, 0x3F800002, 0x00000000)
CANARY_FLAGGED = (False, False, True, True, True, False)
# The smallest subnormal float16, 2**-24, which must widen to the float32 2**-24 exactly.
CANARY_HALF = (0x0001, 0x33800000)


def _canary_rows(mx: ModuleType, stream: Any, rows: ndarray, weights: ndarray, batch: Batch):
    """The rows and the flag of every sentence of ``batch`` from the kernel."""
    t = table(mx, stream, PoolInput(rows, weights), None)
    probe = task(batch, t)
    assert probe is not None
    launched = launch(mx, stream, t, probe)
    mx.eval(*launched)
    rows = np.empty((probe.n_sentences, t.dim), np.float32)
    flags = read(launched, rows)
    return rows, flags


def canary(mx: ModuleType, stream: Any) -> None:
    """Raise UnsupportedByBackendError unless this GPU fuses, rounds 1 / n first, widens
    float16 exactly and flags the cases the kernel must flag, for one and for four values
    per thread; once per process (``_device``)."""
    batch = Batch(0, CANARY_OFFSETS, CANARY_TOKENS)
    for dim in (1, 4):
        rows = np.ascontiguousarray(np.repeat(CANARY_ROWS[:, None], dim, axis=1))
        got, flags = _canary_rows(mx, stream, rows, CANARY_WEIGHTS, batch)
        flagged = (flags & ALWAYS) != 0
        bits = got.view(np.uint32)
        exact = bits[[0, 1, 5]]
        if not (exact == np.array(CANARY_BITS, np.uint32)[:, None]).all():
            raise UnsupportedByBackendError(
                "the mlx backend cannot run on this GPU: its fma or the rounding of 1 / n "
                f"differs from the numerical contract (bits {[hex(b) for b in exact[:, 0]]})"
            )
        tiny = (flags & _sources.TINY) != 0
        if tuple(flagged.tolist()) != CANARY_FLAGGED or not tiny[5] or tiny[:2].any():
            raise UnsupportedByBackendError(
                "the mlx backend cannot run on this GPU: its subnormal check does not work "
                f"(flags {flagged.tolist()})"
            )
    half = np.full((1, 4), CANARY_HALF[0], np.uint16).view(np.float16)
    one = Batch(0, np.array([0, 1], np.int64), np.zeros(1, np.uint32))
    got, flags = _canary_rows(mx, stream, half, np.ones(1, np.float32), one)
    if flags.any() or (got.view(np.uint32) != CANARY_HALF[1]).any():
        raise UnsupportedByBackendError(
            "the mlx backend cannot run on this GPU: it does not widen float16 exactly"
        )
