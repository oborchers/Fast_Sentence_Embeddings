#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The mlx backend: stage 2 (the weighted mean) and ``normalize`` on the GPU, the rest on
the host.

``rows`` pools a batch in the pooling kernel (``_kernels``) through the device thread
(``_device``), which evaluates the calls of all waiting workers together; with
``normalize`` as the only step the kernel also normalizes (rows within one threadgroup)
or the host does on the device's rows. Everything else runs on the injected host backend,
the cpu backend, with the bits of the cpu operation on the same input: stage 5, the
inverse norms, the scores and the Gram blocks. The host is injected (``get_backend``) so
that this package imports neither the cpu backend nor ``fse._native``.

The rows keep the tolerance contract of device backends (ARCHITECTURE.md, "MLX"). An
encoder runs on the host for this backend when its steps are more than ``normalize``
(``steps``; fse 2.0 folds linear steps into the word vectors), when its rows are wider
than 65,536 values (``dim``) or when its factors or row values are or may be subnormal
(``subnormal_input``). The GPU flushes the subnormal products and partial sums that
remain; the bound ``(tau + 2 gamma_K) m + delta`` covers them, the term ``2 gamma_K``
also the roundings a flush turns. From the kernel's ``sum |f| ||x||`` the backend bounds
every row (``_bounds``) and takes from the host the rows that could give another result
or error than the cpu's: a value that is not finite, a sentence of 2**24 tokens, sums
that could overflow on the cpu, before normalize a row that could be zero, tiny or huge,
and before float16 storage a row that could reach its range. The host computes their task
again and so raises the cpu backend's error, if any; every other row keeps the device's
result. The first fallback of an encoder logs a warning, and the backend counts them by
reason.
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

from fse._contract import Aggregation, Batch, Normalize, Op, Step

from .._base import (
    Capabilities,
    PoolInput,
    Prepared,
    check_out,
    check_pool_input,
    check_prepared,
    dims,
)
from . import _bounds, _kernels, _load
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
    "steps": "steps other than normalize alone, which fse 2.0 folds into the word vectors",
    "dim": f"rows of more than {DEVICE_DIM} values",
    "subnormal_input": "a factor or row value of the encoder that is or may be subnormal, "
    "which the GPU would flush to zero",
    "long": "a sentence or unknown word of 2**24 tokens or more",
    "nonfinite": "a value that is not finite",
    "large": "a sentence of more tokens and ngram buckets than one device group takes",
    "limit": "a row near a limit (sums that may overflow, a row zero or tiny before "
    "normalize, a value near the float16 range), whose bound cannot exclude another result",
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
    """The tasks of one encoder computed (in part) on the host, by reason."""

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


@dataclass(frozen=True, eq=False)
class _Pooled:
    """The rows of the last ``rows`` call of a worker, for its ``store``: the bound of
    every row (0 for the host's)."""

    rows: ndarray
    batch: Batch
    state: _State
    error: ndarray


def _host_reason(pool: PoolInput, steps: Sequence[Step], oov: ndarray | None) -> str:
    """Why an encoder runs on the host for this backend, or ""."""
    if any(not isinstance(step, Normalize) for step in steps):
        return "steps"
    if pool.dim > DEVICE_DIM:
        return "dim"
    return "subnormal_input" if _kernels.subnormal_input(pool, oov) else ""


class MlxBackend:
    """The weighted mean on the GPU of Apple silicon through MLX; ``host`` runs the rest."""

    name = "mlx"

    def __init__(self, host: Host, *, device: Device = DEVICE):
        _load.core()  # raises if this machine cannot run it
        self.host = host
        self.device = device
        self._last = threading.local()  # the _Pooled of each worker's last rows

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
        table, reason = None, _host_reason(pool, checked, oov)
        if not reason:
            table = self.device.call(lambda mx, _stream: _kernels.table(mx, pool, oov))
        state = _State(host, table, Fallbacks(), reason)
        return Prepared(self.name, pool, checked, pooled, out, state)

    def rows(self, batch: Batch, prepared: Prepared, out: ndarray) -> None:
        check_prepared(prepared, self.name)
        check_out(out, batch, prepared)
        state = prepared.state
        assert isinstance(state, _State)
        error = self._rows(batch, state, out)
        self._last.pooled = _Pooled(out, batch, state, error)

    def _rows(self, batch: Batch, state: _State, out: ndarray) -> ndarray:
        """The rows of ``batch`` into ``out``; the bound of every row."""
        if batch.n_sentences == 0:
            return np.zeros(0)
        table = state.table
        if table is None:
            state.fallbacks.count(state.reason)
            self.host.rows(batch, state.host, out)
            return np.zeros(batch.n_sentences)
        size = _kernels.size(batch)
        if size > _kernels.MAX_CONTRIBUTIONS or (
            size > self.device.group_tokens and batch.n_sentences > 1
        ):  # more than one group, or perhaps a sentence the device must not take
            return self._large(batch, state, out)
        task = _kernels.task(batch, table)
        if task is None:  # an error the cpu backend reports, in its words
            self.host.rows(batch, state.host, out)
            return np.zeros(batch.n_sentences)
        return self._device_rows(batch, state, table, task, out)

    def _device_rows(
        self, batch: Batch, state: _State, table: _kernels.Table, task: _kernels.Task, out
    ) -> ndarray:
        """The rows of ``task`` on the device, the flagged ones from the host."""
        normalize = bool(state.host.steps)  # normalize alone: the others run on the host
        in_kernel = normalize and _kernels.normalizes(table)
        flags, bounds = self.device.run(
            lambda mx, stream: _kernels.launch(mx, stream, table, task, in_kernel),
            _kernels.view,
            task.tokens,
            lambda viewed: _use(viewed, out),  # in this worker, from the lent rows
        )
        n, k = np.diff(task.arrays[1]), _kernels.sizes(batch)
        error = _bounds.pooling(bounds[:, 0], n, k, table.dim)
        bad = _kernels.flagged(flags) | _bounds.overflows(bounds[:, 0], k) & (n > 0)
        if normalize:
            if in_kernel:
                low, high = _bounds.device_norms(bounds[:, 1], table.dim)
            else:
                low = high = _bounds.norms(out)
            bad |= _bounds.unnormalizable(low, high, error) & (n > 0)
            if not in_kernel:
                out[bad] = 0  # normalize keeps a zero row and raises for none
                self.host.steps(out, state.host, batch.first_row, out)
            error = np.where(n > 0, _bounds.after_normalize(error, low, table.dim), 0.0)
        if bad.any():  # the kernel's flags, or "limit" for the bounds alone
            state.fallbacks.count(
                _kernels.reason(functools.reduce(operator.or_, flags[bad].tolist()))
            )
            self._replace(batch, state, out, bad)
            error[bad] = 0
        return error

    def _replace(self, batch: Batch, state: _State, out: ndarray, bad: ndarray) -> None:
        """The rows ``bad`` from the host, which computes the whole batch first, so that
        its error, if any, is the cpu's for the batch."""
        host = np.empty_like(out)
        self.host.rows(batch, state.host, host)
        out[bad] = host[bad]

    def _large(self, batch: Batch, state: _State, out: ndarray) -> ndarray:
        """A batch beyond one device group (or that may hold a sentence of K >= 2**24) in
        runs of sentences of at most a group, after the host has computed it whole (which
        raises the cpu's error, if any): a sentence of more than a group runs alone, one of
        K >= 2**24 keeps the host's row (``large``); a sentence's path depends on its own
        K only, never on the group."""
        self.host.rows(batch, state.host, out)
        error = np.zeros(batch.n_sentences)
        limit = _kernels.MAX_CONTRIBUTIONS
        for begin, end, part in _kernels.split(batch, min(self.device.group_tokens, limit), limit):
            if part is None:
                state.fallbacks.count("large")
            else:
                error[begin:end] = self._rows(part, state, out[begin:end])
        return error

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
        """Stage 5 on the host. Before float16 storage of the rows this worker's last
        ``rows`` gave, a row that could reach the float16 range on the cpu takes the
        host's row, so that the store raises the cpu's error if there is one."""
        pooled = getattr(self._last, "pooled", None)
        if codes.dtype == np.float16 and pooled is not None and pooled.rows is rows:
            near = _bounds.beyond_float16(rows, pooled.error)
            if near.any():
                pooled.state.fallbacks.count("limit")
                self._replace(pooled.batch, pooled.state, rows, near)
        self.host.store(rows, codes, scales, first_row)


def _use(viewed: tuple, out: ndarray) -> tuple[ndarray, ndarray]:
    """Copy the device's rows into ``out``: their flags and bounds."""
    rows, flags, bounds = viewed
    np.copyto(out, rows)
    return flags, bounds


def fallbacks(prepared: Prepared) -> dict[str, int]:
    """The tasks of ``prepared`` computed (in part) on the host so far, by reason."""
    check_prepared(prepared, MlxBackend.name)
    state = prepared.state
    assert isinstance(state, _State)
    with state.fallbacks.lock:
        return dict(state.fallbacks.counts)
