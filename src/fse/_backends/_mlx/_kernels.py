#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The pooling kernel of the mlx backend: its arrays, its launch and its canary.

``table`` uploads the word rows of an encoder once, ``task`` turns a batch into the host
arrays of one launch (or None when the CPU must report the batch's error), ``launch``
queues the kernel of one task and ``read`` copies its rows out and gives the OR of its
flags. Every function that calls MLX runs in the device thread (``_device``); ``task``
is plain numpy and runs in the worker.
"""

from __future__ import annotations

import functools
import operator
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import OOV_TOKEN, Batch

from .._base import PoolInput, UnsupportedByBackendError
from ._sources import HEADER, POOL, POOL_INPUTS, POOL_OUTPUTS

# One thread per value; a threadgroup of up to 256 threads (8 SIMD groups) along a row.
THREADGROUP = 256
SIMD = 32
_LOCK = threading.Lock()
_KERNELS: dict[str, Any] = {}


def kernel(mx: ModuleType, name: str = "fse_pool_weighted_mean") -> Any:
    """The pooling kernel, built once per process (MLX compiles each instance once)."""
    with _LOCK:
        built = _KERNELS.get(name)
        if built is None:
            built = _KERNELS[name] = mx.fast.metal_kernel(
                name=name,
                input_names=list(POOL_INPUTS),
                output_names=list(POOL_OUTPUTS),
                source=POOL,
                header=HEADER,
                compile_options={"math_mode": "safe"},
            )
        return built


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
            pool.ngram_vectors if fasttext else np.zeros(1, pool.vectors.dtype),
            pool.ngram_scales if fasttext and int8 else dummy,
            dummy if oov_weight is None else oov_weight.reshape(1),
        )
    )
    mx.eval(*arrays)
    buckets = len(pool.ngram_vectors) if pool.ngram_vectors is not None else 0
    return Table(
        arrays, pool.dim, len(pool.vectors), buckets, int8, fasttext, oov_weight is not None
    )


@dataclass(frozen=True, eq=False)
class Task:
    """The host arrays of one launch: the tokens, offsets, the index of the first unknown
    word of every sentence, and the ngram offsets and buckets (one value each without)."""

    n_sentences: int
    first_row: int
    arrays: tuple
    unknown: bool

    @property
    def tokens(self) -> int:
        """The tokens and ngram buckets the kernel reads, the size of its work."""
        return len(self.arrays[0]) + (len(self.arrays[4]) if self.unknown else 0)


_NO_OFFSETS = np.zeros(1, np.int64)
_NO_TOKENS = np.zeros(1, np.uint32)


def task(batch: Batch, table: Table) -> Task | None:
    """The arrays of ``batch`` for the kernel, or None if the CPU reports an error for it:
    a token or bucket beyond the rows, or an unknown word the encoder cannot add. An index
    out of range would read other memory on the GPU without any error."""
    tokens = batch.tokens
    unknown = tokens == OOV_TOKEN
    any_unknown = bool(unknown.any())
    known = tokens[~unknown] if any_unknown else tokens
    if len(known) and int(known.max()) >= table.n_words:
        return None
    if not any_unknown:
        arrays = (tokens, batch.offsets, _NO_OFFSETS, _NO_OFFSETS, _NO_TOKENS)
        return Task(batch.n_sentences, batch.first_row, arrays, False)
    ngrams, ngram_offsets = batch.ngrams, batch.ngram_offsets
    if not (table.fasttext and table.oov_weight) or ngrams is None or ngram_offsets is None:
        return None
    if len(ngrams) and int(ngrams.max()) >= table.n_buckets:
        return None
    before = np.concatenate([[0], np.cumsum(unknown, dtype=np.int64)])[batch.offsets[:-1]]
    flat = ngrams if len(ngrams) else _NO_TOKENS
    arrays = (tokens, batch.offsets, np.ascontiguousarray(before), ngram_offsets, flat)
    return Task(batch.n_sentences, batch.first_row, arrays, True)


def launch(mx: ModuleType, stream: Any, table: Table, task: Task, checks: bool = True) -> tuple:
    """The kernel of ``task``, queued on ``stream``: its rows and flags (not evaluated).
    ``checks=False`` drops the subnormal flags, for the tests that show they matter."""
    padded = -(-table.dim // SIMD) * SIMD
    words = padded // SIMD
    inputs = [*table.arrays, *(mx.array(a) for a in task.arrays)]
    return tuple(
        kernel(mx)(
            inputs=inputs,
            template=[
                ("DIM", table.dim),
                ("WORDS", words),
                ("INT8", table.int8),
                ("FASTTEXT", table.fasttext and task.unknown),
                ("CHECKS", checks),
            ],
            grid=(padded, task.n_sentences, 1),
            threadgroup=(min(THREADGROUP, padded), 1, 1),
            output_shapes=[(task.n_sentences, table.dim), (task.n_sentences, words)],
            output_dtypes=[mx.float32, mx.uint32],
            stream=stream,
        )
    )


def read(launched: tuple, target: ndarray) -> int:
    """Copy the evaluated rows of a launch into ``target``; the OR of its flags."""
    rows, flags = launched
    np.copyto(target, np.asarray(rows))
    return functools.reduce(operator.or_, np.unique(np.asarray(flags)).tolist(), 0)


# The canary: cases whose bits the GPU must give, checked against fixed patterns (not
# against a reference computed in the floating-point environment of the caller). Word 0 is
# f = 1 + 2**-12, word 1 the rounded -(f * f), word 2 is 1 + 2**-23, word 3 the smallest
# subnormal.
# Bit patterns, not arithmetic: numpy would compute them in the floating-point environment
# of the importing thread (rounding up, or flushing 2**-149 to zero).
CANARY_ROWS = np.array([[0x3F800800], [0xBF801000], [0x3F800001], [0x00000001]], np.uint32).view(
    np.float32
)
CANARY_WEIGHTS = np.array([0x3F800800, 0x3F800000, 0x3F800000, 0x3F800000], np.uint32).view(
    np.float32
)
# Sentence 0 keeps the rounding error of f * f only if the fma is fused; sentence 1 is
# (1 / 3) * acc, not acc / 3 (0x3F800001); sentence 2 must set the subnormal flag.
CANARY_OFFSETS = np.array([0, 2, 5, 6], np.int64)
CANARY_TOKENS = np.array([1, 0, 2, 2, 2, 3], np.uint32)
CANARY_BITS = (0x33000000, 0x3F800002)


def canary(mx: ModuleType, stream: Any) -> None:
    """Raise UnsupportedByBackendError unless this GPU fuses, rounds 1 / n first and
    flags subnormal values as the kernel expects; once per process (``_device``)."""
    t = table(mx, PoolInput(CANARY_ROWS, CANARY_WEIGHTS), None)
    probe = task(Batch(0, CANARY_OFFSETS, CANARY_TOKENS), t)
    assert probe is not None
    launched = launch(mx, stream, t, probe)
    mx.eval(*launched)
    rows = np.empty((3, 1), np.float32)
    read(launched, rows)
    flags = np.asarray(launched[1]).ravel()
    got = tuple(int(b) for b in rows[:2].view(np.uint32).ravel())
    if got != CANARY_BITS:
        raise UnsupportedByBackendError(
            "the mlx backend cannot run on this GPU: its fma or the rounding of 1 / n "
            f"differs from the numerical contract (bits {[hex(b) for b in got]})"
        )
    if flags[0] or flags[1] or not flags[2] & 1:
        raise UnsupportedByBackendError(
            "the mlx backend cannot run on this GPU: its subnormal check does not work "
            f"(flags {flags.tolist()})"
        )
