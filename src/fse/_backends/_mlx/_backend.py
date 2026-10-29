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
from . import _kernels, _load
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
    "limit": "a row before normalize whose sum of squares may overflow or be subnormal",
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
        state = _State(host, table, Fallbacks(), reason)
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
        normalize = any(isinstance(step, Normalize) for step in steps)
        # normalize alone runs in the kernel, with the cpu's bits where nothing flushes
        on_device = normalize and len(steps) == 1 and _kernels.normalizes(table)
        flags, bad = self.device.run(
            lambda mx, stream: _kernels.launch(mx, stream, table, task, on_device),
            _kernels.view,
            task.tokens,
            lambda viewed: self._use(viewed, task, pooled, normalize),  # from the lent rows
        )
        if not bad.any():
            if steps and not on_device:
                self.host.steps(pooled, state.host, batch.first_row, out)
            return
        why = _kernels.reason(functools.reduce(operator.or_, flags[bad].tolist()))
        state.fallbacks.count(why)
        self._replace(batch, state, pooled if not on_device else None, out, bad)

    @staticmethod
    def _use(viewed: tuple, task: _kernels.Task, pooled: ndarray, normalize: bool):
        """Copy the device's rows into ``pooled``: the flags and the flagged rows."""
        rows, flags = viewed
        np.copyto(pooled, rows)
        return flags, _kernels.flagged(flags, task, normalize)

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

    def _replace(
        self, batch: Batch, state: _State, pooled: ndarray | None, out: ndarray, bad: ndarray
    ):
        """The rows ``bad`` from the host, the others from the device (after the steps on
        ``pooled``, or already stepped without). The host computes the whole task first,
        so that its error, if any, is the cpu's for the task."""
        host = np.empty_like(out)
        self.host.rows(batch, state.host, host)
        if state.host.steps and pooled is not None:
            pooled[bad] = 0  # the steps keep a zero row and raise for none
            self.host.steps(pooled, state.host, batch.first_row, out)
        out[bad] = host[bad]

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
