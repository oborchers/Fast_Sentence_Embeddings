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

The rows keep the tolerance contract of device backends (ARCHITECTURE.md, "MLX"). The GPU
flushes subnormal values to zero: an encoder whose factors or row values are or may be
subnormal runs on the host (``subnormal_input``), and the flushes of smaller products and
partial sums are within the contract's absolute term. The rows the kernel flags (a value
that is not finite, a sentence of 2**24 tokens, and before normalize a row near a limit
of its sum of squares) come from the host, which computes their task again and so raises
the cpu backend's error, if any; every other row keeps the device's result. The first
fallback of an encoder logs a warning, and the backend counts them by reason.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import operator
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy import ndarray

from fse._contract import Aggregation, Batch, Normalize, Op, Project, Step

from .._base import (
    Capabilities,
    PoolInput,
    Prepared,
    check_out,
    check_pool_input,
    check_prepared,
    dims,
)
from . import _bounds, _kernels, _load, _stages
from ._device import Device

logger = logging.getLogger(__name__)

DEVICE = Device(_load.core)  # the one device thread of the process, for every encoder

DEVICE_DIM = 1 << 16  # wider rows run on the host (the contract's d <= 65,536)
_CAPABILITIES = Capabilities(
    word_dtypes=frozenset({"float32", "float16", "int8"}),
    aggregations=frozenset({Aggregation.WEIGHTED_MEAN}),
    steps=frozenset(Op),
    fasttext=True,
    concurrent=True,
    max_dim=DEVICE_DIM,  # the host's limit in capabilities()
    device_rows=True,
)
_WHY = {
    "dim": f"rows of more than {DEVICE_DIM} values",
    "subnormal_input": "a factor or row value of the encoder that is or may be subnormal, "
    "which the GPU would flush to zero",
    "long": "a sentence or unknown word of 2**24 tokens or more",
    "nonfinite": "a value that is not finite",
    "large": "a sentence of more tokens and ngram buckets than one device group takes",
    "limit": "a row near a limit of a step (zero or tiny before normalize, or values that "
    "may overflow), whose bound cannot exclude another result",
}


class Host(Protocol):
    """What the mlx backend needs of its host backend (the cpu backend)."""

    name: str

    def capabilities(self) -> Capabilities: ...

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

    def count(self, reason: str) -> str:
        with self.lock:
            first = not any(self.counts.values())
            self.counts[reason] += 1
        if first:
            logger.warning(
                "the mlx backend computes a task on the CPU (%s); it does so for every such "
                "task of this encoder, which is slower but gives the cpu's rows",
                _WHY[reason],
            )
        return reason


@dataclass(frozen=True, eq=False)
class _State:
    host: Prepared
    table: _kernels.Table | None  # None: the encoder runs on the host, for ``reason``
    fallbacks: Fallbacks
    reason: str = ""
    stages: tuple = ()  # the steps one by one on the host (``_stages``)


class MlxBackend:
    """The weighted mean on the GPU of Apple silicon through MLX; ``host`` runs the rest."""

    name = "mlx"

    def __init__(self, host: Host, *, device: Device = DEVICE):
        _load.core()  # raises if this machine cannot run it
        self.host = host
        self.device = device

    def capabilities(self) -> Capabilities:
        """The cpu host's dimensions: wider rows than the device takes run on the host."""
        return dataclasses.replace(_CAPABILITIES, max_dim=self.host.capabilities().max_dim)

    def prepare(self, pool: PoolInput, steps: Sequence[Step]) -> Prepared:
        check_pool_input(pool)
        checked, pooled, out = dims(pool, steps)  # checks and materializes the steps once
        self.capabilities().check(self.name, pool, checked)
        host = self.host.prepare(pool, checked)
        self.device.ready()  # the canary, once per process
        oov = None if pool.oov_weight is None else self.host.float32([pool.oov_weight])
        table, reason = None, ""
        if pool.dim > DEVICE_DIM:
            reason = "dim"
        elif _kernels.subnormal_input(pool, oov):
            reason = "subnormal_input"
        else:
            table = self.device.call(lambda mx, _stream: _kernels.table(mx, pool, oov))
        state = _State(host, table, Fallbacks(), reason, _stages.stages(self.host, checked, pooled))
        return Prepared(self.name, pool, checked, pooled, out, state)

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        check_prepared(prepared, self.name)
        check_out(out, batch, prepared)
        state = prepared.state
        assert isinstance(state, _State)
        if batch.n_sentences == 0:
            return
        table = state.table
        if table is None:
            state.fallbacks.count(state.reason)
            self.host.rows(batch, state.host, out)
            return
        size = _kernels.size(batch)
        if size > _kernels.MAX_CONTRIBUTIONS or (
            size > self.device.group_tokens and batch.n_sentences > 1
        ):  # more than one group, or perhaps a sentence the device must not take
            self._large(batch, prepared, out)
            return
        task = _kernels.task(batch, table)
        if task is None:  # an error the cpu backend reports, in its words
            self.host.rows(batch, state.host, out)
            return
        project = any(isinstance(step, Project) for step in prepared.steps)
        pooled = np.empty((task.n_sentences, prepared.pooled_dim), np.float32) if project else out
        steps = state.host.steps
        # normalize alone runs in the kernel, with the cpu's bits where nothing flushes
        on_device = len(steps) == 1 and isinstance(steps[0], Normalize)
        on_device = on_device and _kernels.normalizes(table)
        flags, bounds = self.device.run(
            lambda mx, stream: _kernels.launch(mx, stream, table, task, bool(steps), on_device),
            _kernels.view,
            task.tokens,
            lambda viewed: _use(viewed, pooled),  # in this worker, from the lent rows
        )
        bad = _kernels.flagged(flags)
        if steps:
            bad = self._steps(batch, state, task, (pooled, out), bounds, bad, on_device)
        if bad.any():
            why = _kernels.reason(functools.reduce(operator.or_, flags[bad].tolist()))
            state.fallbacks.count(why)
            host = np.empty_like(out)  # the whole task, for the cpu's error if there is one
            self.host.rows(batch, state.host, host)
            out[bad] = host[bad]

    def _steps(
        self,
        batch: Batch,
        state: _State,
        task: _kernels.Task,
        rows: tuple[ndarray, ndarray],
        bounds: ndarray,
        bad: ndarray,
        on_device: bool,
    ) -> ndarray:
        """The steps on the device's rows (``pooled``, ``out``), already run with
        ``on_device``: the rows the host must give, also those near a limit of a step."""
        n = np.diff(task.arrays[1])
        error = _bounds.pooling(bounds[:, 0], n, _kernels.sizes(batch), state.host.pooled_dim)
        if on_device:
            low, high = _bounds.device_norms(bounds[:, 1], state.host.pooled_dim)
            return bad | (_bounds.unnormalizable(low, high, error) & (n > 0))
        pooled, out = rows
        return _stages.run(self.host, state.stages, pooled, out, batch.first_row, error, n > 0, bad)

    def _large(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        """A batch beyond one device group (or that may hold a sentence of K >= 2**24) in
        runs of sentences of at most a group, after the host has computed it whole (which
        raises the cpu's error, if any): a sentence of more than a group runs alone, one of
        K >= 2**24 keeps the host's row (``large``); a sentence's path depends on its own
        K only, never on the group."""
        state = prepared.state
        assert isinstance(state, _State)
        self.host.rows(batch, state.host, out)
        limit = _kernels.MAX_CONTRIBUTIONS
        parts = _kernels.split(batch, min(self.device.group_tokens, limit), limit)
        for begin, end, part in parts:
            if part is None:
                state.fallbacks.count("large")
            else:
                self.rows(part, prepared, out[begin:end])

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


def _use(viewed: tuple, pooled: ndarray) -> tuple[ndarray, ndarray]:
    """Copy the device's rows into ``pooled``: their flags and bounds."""
    rows, flags, bounds = viewed
    np.copyto(pooled, rows)
    return flags, bounds


def fallbacks(prepared: Prepared) -> dict[str, int]:
    """The tasks of ``prepared`` computed again on the host so far, by reason."""
    check_prepared(prepared, MlxBackend.name)
    state = prepared.state
    assert isinstance(state, _State)
    with state.fallbacks.lock:
        return dict(state.fallbacks.counts)
