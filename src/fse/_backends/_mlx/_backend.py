#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The mlx backend: stage 2 (the weighted mean) on the GPU, the rest on the host.

``rows`` pools a batch in the pooling kernel (``_kernels``) through the device thread
(``_device``), which evaluates the calls of all waiting workers together. Everything
else runs on the injected host backend, the cpu backend, with the same bits by
construction: the steps on the pooled rows (``host.steps``), stage 5, the inverse norms,
the scores and the Gram blocks. The host is injected (``get_backend``) so that this
package imports neither the cpu backend nor ``fse._native``.

The GPU flushes subnormal values to zero. A task whose kernel flags such a value (or a
sentence of 2**24 tokens, a value that is not finite) is
computed again on the host, which gives the bits and the errors of the cpu backend; the
first such task of an encoder logs a warning, and the backend counts them by reason.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy import ndarray

from fse._contract import Aggregation, Batch, Op, Project, Step

from .._base import (
    Capabilities,
    PoolInput,
    Prepared,
    check_out,
    check_pool_input,
    check_prepared,
    dims,
)
from . import _kernels, _load, _sources
from ._device import Device

logger = logging.getLogger(__name__)

DEVICE = Device(_load.core)  # the one device thread of the process, for every encoder

MAX_DIM = 1 << 16
_CAPABILITIES = Capabilities(
    word_dtypes=frozenset({"float32", "float16", "int8"}),
    aggregations=frozenset({Aggregation.WEIGHTED_MEAN}),
    steps=frozenset(Op),
    fasttext=True,
    concurrent=True,
    max_dim=MAX_DIM,
    device_rows=True,
)
_REASONS = (
    (_sources.SUBNORMAL, "underflow"),
    (_sources.LONG, "long"),
    (_sources.NONFINITE, "nonfinite"),
)
_WHY = {
    "underflow": "a possible underflow: a subnormal value, or a product or sum that may be "
    "one, which the GPU would flush to zero",
    "long": "a sentence or unknown word of 2**24 tokens or more",
    "nonfinite": "a value that is not finite",
    "large": "more tokens and ngram buckets than one device group takes",
}


class Host(Protocol):
    """What the mlx backend needs of its host backend (the cpu backend)."""

    name: str

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared: ...

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None: ...

    def steps(self, pooled: ndarray, prepared: Prepared, first_row: int, out: ndarray) -> None: ...

    def float32(self, values: object) -> ndarray: ...

    def gram_block(self, rows: ndarray) -> ndarray: ...

    def score(self, stored: ndarray, scales: Any, weights: ndarray, bias: float) -> ndarray: ...

    def inverse_norms(self, stored: ndarray, first_row: int = 0) -> ndarray: ...

    def cosines(self, stored: ndarray, inverse: ndarray, queries: ndarray) -> ndarray: ...

    def scores(self, stored: ndarray, scales: Any, weights: ndarray, bias: Any) -> ndarray: ...

    def store(self, rows: ndarray, codes: ndarray, scales: Any, first_row: int) -> None: ...


@dataclass(eq=False)
class Fallbacks:
    """The tasks of one encoder computed on the host, by reason: the first flag of the
    kernel (``underflow``, a value that is or may be subnormal; ``long``; ``nonfinite``),
    or ``large``, a task the device does not take."""

    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_WHY, 0))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def count(self, flags: int, reason: str | None = None) -> str:
        reason = reason or next(name for bit, name in _REASONS if flags & bit)
        with self.lock:
            first = not any(self.counts.values())
            self.counts[reason] += 1
        if first:
            logger.warning(
                "the mlx backend computes a task on the CPU (%s); it does so for every such "
                "task of this encoder, which is slower but gives the same bits",
                _WHY[reason],
            )
        return reason


@dataclass(frozen=True, eq=False)
class _State:
    host: Prepared
    table: _kernels.Table
    fallbacks: Fallbacks


class MlxBackend:
    """The weighted mean on the GPU of Apple silicon through MLX; ``host`` runs the rest."""

    name = "mlx"

    def __init__(self, host: Host, *, device: Device = DEVICE):
        _load.core()  # raises if this machine cannot run it
        self.host = host
        self.device = device
        self._checks = True  # False only in the tests that show the flags matter

    def capabilities(self) -> Capabilities:
        return _CAPABILITIES

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared:
        check_pool_input(pool)
        checked, pooled, out = dims(pool, steps)  # checks and materializes the steps once
        _CAPABILITIES.check(self.name, pool, checked)
        host = self.host.prepare(pool, checked)
        self.device.ready()  # the canary, once per process
        oov = None if pool.oov_weight is None else self.host.float32([pool.oov_weight])
        table = self.device.call(lambda mx, _stream: _kernels.table(mx, pool, oov))
        return Prepared(self.name, pool, checked, pooled, out, _State(host, table, Fallbacks()))

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        check_prepared(prepared, self.name)
        check_out(out, batch, prepared)
        state = prepared.state
        assert isinstance(state, _State)
        if batch.n_sentences == 0:
            return
        task = _kernels.task(batch, state.table)
        if task is None:  # an error the cpu backend reports, in its words
            self.host.rows(batch, state.host, out)
            return
        if task.tokens > self.device.group_tokens:  # a single upload and kernel too large
            state.fallbacks.count(0, "large")
            self.host.rows(batch, state.host, out)
            return
        project = any(isinstance(step, Project) for step in prepared.steps)
        pooled = np.empty((task.n_sentences, prepared.pooled_dim), np.float32) if project else out
        checks = self._checks
        flags = self.device.run(
            lambda mx, stream: _kernels.launch(mx, stream, state.table, task, checks),
            lambda launched: _kernels.read(launched, pooled),
            task.tokens,
        )
        if flags:
            state.fallbacks.count(flags)
            self.host.rows(batch, state.host, out)
        elif prepared.steps:
            self.host.steps(pooled, state.host, batch.first_row, out)

    def gram_block(self, rows: ndarray) -> ndarray:
        return self.host.gram_block(rows)

    def score(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float
    ) -> ndarray:
        return self.host.score(stored, scales, weights, bias)

    def float32(self, values: object) -> ndarray:
        return self.host.float32(values)

    def inverse_norms(self, stored: ndarray, first_row: int = 0) -> ndarray:
        return self.host.inverse_norms(stored, first_row)

    def cosines(self, stored: ndarray, inverse: ndarray, queries: ndarray) -> ndarray:
        return self.host.cosines(stored, inverse, queries)

    def scores(
        self, stored: ndarray, scales: ndarray | None, weights: ndarray, bias: float | ndarray
    ) -> ndarray:
        return self.host.scores(stored, scales, weights, bias)

    def store(self, rows: ndarray, codes: ndarray, scales: ndarray | None, first_row: int) -> None:
        self.host.store(rows, codes, scales, first_row)


def fallbacks(prepared: Prepared) -> dict[str, int]:
    """The tasks of ``prepared`` computed again on the host so far, by reason."""
    check_prepared(prepared, MlxBackend.name)
    state = prepared.state
    assert isinstance(state, _State)
    with state.fallbacks.lock:
        return dict(state.fallbacks.counts)
