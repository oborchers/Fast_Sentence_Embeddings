#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The pipeline of an encoder on one backend: sources to finished rows.

Built once per encoder and backend: the backend is prepared and the lookup made (on the
encoder's one ``Vocabulary``, if given) before any input is read, so an unsupported
combination fails at once. ``infer`` then plans the tasks of a source and runs them with the executor; every worker thread encodes its task
(stages 1 and 2) with its own workspace and lets the backend turn the batch into the
finished rows of its task (stages 2 to 4) and, for a float16 or int8 sink, store them
(stage 5), so nothing of the rows is computed in the calling thread. A Pipeline does not
change, so several threads may call ``infer`` at once (a server: ``workers=1`` each).
"""

from __future__ import annotations

import dataclasses
import threading
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._backends import PoolInput, get_backend
from fse._contract import Batch, Step
from fse._cpus import usable_cpus
from fse._text import (
    MAX_CHUNK_ROWS,
    Vocabulary,
    check_chunk_bytes,
    make_lookup,
    new_workspace,
)

from ._errors import earliest, first_rows
from ._executor import run
from ._sinks import (
    ArrayRows,
    GrowingRows,
    Rows,
    Sink,
    Stored,
    sink_dtype,
    split_target,
    wants_inverse_norms,
)
from ._spec import EncodeSpec
from ._tasks import Plan, Task, plan

# The finished rows of one task take at most this many bytes, whatever the bytes of its
# sentences (an empty sentence takes none). Smaller tasks cost throughput: a task has a
# fixed cost of about 0.2 ms for its checks (4 MiB lost 7 % on an M1 Pro, 16 MiB 1 %).
TASK_ROW_BYTES = 1 << 24
# The rows of all tasks in flight or waiting to be committed take at most this many bytes:
# the window is 2 * workers tasks, so with many workers a task holds fewer rows. A task
# that normalizes counts 12 more bytes per value (STEP_ROW_BYTES), although normalize runs
# in place: without them its tasks were up to twice as large and the peak RSS of a stream
# at 10 workers grew by 65 to 140 MiB for at most 10 % throughput (M1 Pro, PR 123), as the
# rest of a task (its batch, tokens and buffers) grows with its rows too.
WINDOW_BYTES = 1 << 29
STEP_ROW_BYTES = 12


class Pipeline:
    def __init__(
        self,
        spec: EncodeSpec,
        backend: str = "cpu",
        *,
        isa: str | None = None,
        vocabulary: Vocabulary | None = None,
        pool: PoolInput | None = None,
        steps: tuple[Step, ...] | None = None,
    ):
        """``pool`` and ``steps`` replace the encoder's table and steps (a head reads the
        rows of a folded table, or the full mean with its own step); the tokenizer and the
        vocabulary stay the encoder's."""
        if not isinstance(spec, EncodeSpec):
            raise TypeError(f"spec must be an EncodeSpec, not {type(spec).__name__}")
        self.spec = spec
        self.backend = get_backend(backend, isa=isa)
        self.prepared = self.backend.prepare(
            spec.pool_input() if pool is None else pool, spec.steps if steps is None else steps
        )
        self.lookup = make_lookup(
            spec.word_vectors, spec.tokenizer, max_ngrams=spec.max_ngrams, vocabulary=vocabulary
        )
        self.vocabulary = vocabulary  # shared with the pipelines of a head on this encoder
        self.output_dim = self.prepared.output_dim
        self.task_rows = max(1, min(MAX_CHUNK_ROWS, TASK_ROW_BYTES // (4 * self.output_dim)))
        # One call at a time on a backend that is not concurrent; encoding stays parallel.
        concurrent = self.backend.capabilities().concurrent
        self._device_rows = self.backend.capabilities().device_rows
        self._serial: AbstractContextManager[Any] = (
            nullcontext() if concurrent else threading.Lock()
        )

    def _row_bytes(self, dtype: str, inverse: bool = False) -> int:
        """The bytes one row of a task takes at most: its target until its commit (with
        its inverse norm if the sink wants them), while it runs the float32 rows before
        storage (a float16 or int8 target), STEP_ROW_BYTES per value with normalize, and on
        a device backend its pooled rows, flags and bounds."""
        out = self.output_dim
        target = (
            np.dtype(_codes.STORAGE_TYPES[dtype]).itemsize * out
            + (4 if dtype == "int8" else 0)
            + (8 if inverse else 0)
        )  # the vector in the target and the backend's result
        scratch = 0 if dtype == "float32" else 4 * out
        steps = STEP_ROW_BYTES * out if self.prepared.steps else 0
        # A device backend's pooled rows, flags (one word per 32 values) and two bounds.
        device = 4 * out + 4 * -(-out // 32) + 8 if self._device_rows else 0
        return target + scratch + steps + device

    def infer(
        self,
        source: object,
        *,
        workers: int | None = None,
        sink: Sink | None = None,
        chunk_bytes: int | None = None,
        dtype: str = "float32",
        first_row: int = 0,
    ) -> Rows | None:
        """The rows of every sentence of ``source``: returned (as ``dtype``: float32 rows,
        float16 codes, or int8 codes and scales), or committed to ``sink`` in its dtype,
        numbered from ``first_row`` there (the rows of a run before this source).

        Every argument is checked before the source is read.
        """
        workers = check_workers(workers) or usable_cpus()
        base = _non_negative(first_row, "first_row")
        inverse = False
        if sink is not None:
            check_sink_rows(sink, self.output_dim)
            dtype = sink_dtype(sink)
            inverse = wants_inverse_norms(sink)
        else:
            dtype = _codes.check_dtype(dtype)
            if base:
                raise ValueError("first_row numbers the rows of a sink; without one it is 0")
        tasks, window, stop = self.plan_tasks(
            source, workers, chunk_bytes, self._row_bytes(dtype, inverse)
        )
        own: ArrayRows | GrowingRows | None = None
        if sink is None:
            rows = tasks.rows
            own = (
                GrowingRows(self.output_dim, dtype)
                if rows is None
                else ArrayRows(rows, self.output_dim, dtype)
            )
            sink = own
        local = threading.local()

        def reserve(task: Task) -> tuple[Task, Stored]:  # in the calling thread, in order
            return task, sink.target(base + task.first_row, task.n_rows)

        def work(item: tuple[Task, Stored]) -> Stored:
            self._work(local, item[0], split_target(item[1], dtype, inverse), base)
            return item[1]

        def commit(item: tuple[Task, Stored], rows: Stored):
            sink.commit(base + item[0].first_row, rows)

        run(
            tasks.tasks,
            work,
            commit,
            workers=workers,
            inline=tasks.inline,
            window=window,
            reserve=reserve,
            stop=stop,
        )
        return None if own is None else own.result()

    def plan_tasks(
        self, source: object, workers: int, chunk_bytes: int | None, row_bytes: int
    ) -> tuple[Plan, int, threading.Event]:
        """The tasks of ``source`` for rows of ``row_bytes`` each, the window, and the event
        the executor sets once a task failed. The rows of the window stay within
        WINDOW_BYTES: with many workers a task holds fewer rows, and with rows wider than
        the budget allows the window is smaller (at least one task of one row, however
        wide)."""
        window = max(1, min(2 * workers, WINDOW_BYTES // row_bytes))
        task_rows = max(1, min(self.task_rows, WINDOW_BYTES // (window * row_bytes)))
        stop = threading.Event()
        tasks = plan(
            source,
            workers=workers,
            chunk_bytes=check_chunk_bytes(chunk_bytes),
            chunk_rows=task_rows,
            tokenizer=self.spec.tokenizer.name,
            stop=stop,
        )
        return tasks, window, stop

    def _work(self, local: threading.local, task: Task, target: tuple, base: int):
        """Stages 1 to 5 of one task in a worker thread: its rows into the codes of
        ``target`` (codes, scales, inverse norms), numbered from ``base`` in errors; an
        error is that of its earliest failing row (``earliest``)."""

        def rows(n: int) -> None:
            part = task.prefix(n)
            self.finish(
                local, self.batch(local, part, base), first_rows(target, n), base + task.first_row
            )

        earliest(rows, task.n_rows)

    def batch(self, local: threading.local, task: Task, base: int) -> Batch:
        """Stages 1 and 2 of one task with the workspace of its worker thread; its rows are
        numbered from ``base`` (errors name the rows of the run)."""
        workspace = getattr(local, "workspace", None)
        if workspace is None:
            workspace = local.workspace = new_workspace()
        batch = task.batch(self.lookup, workspace)
        if base:
            batch = dataclasses.replace(batch, first_row=base + batch.first_row)
        return batch

    def finish(self, local: threading.local, batch: Batch, target: tuple, first: int) -> ndarray:
        """Stages 2 to 5 of ``batch`` (rows ``first`` on) into ``target`` (codes, scales,
        inverse norms); returns the float32 rows, the codes themselves or the scratch of
        the worker they were stored from."""
        codes, scales, norms = target
        with self._serial:
            if codes.dtype == np.float32:
                rows = codes
                self.backend.rows(batch, self.prepared, codes)
            else:
                rows = _scratch(local, len(codes), self.output_dim)
                self.backend.rows(batch, self.prepared, rows)
                self.backend.store(rows, codes, scales, first)
            if norms is not None:
                norms[...] = self.backend.inverse_norms(codes, first)
        return rows


def _scratch(local: threading.local, n_rows: int, dim: int) -> ndarray:
    """The float32 rows of one task before they are stored, reused by its worker thread."""
    scratch = getattr(local, "scratch", None)
    if scratch is None or len(scratch) < n_rows or scratch.shape[1] != dim:
        scratch = local.scratch = np.empty((n_rows, dim), dtype=np.float32)
    return scratch[:n_rows]


def check_workers(workers: object) -> int | None:
    """``workers`` of a run: None (the usable CPUs) or a positive int."""
    return None if workers is None else _positive(workers, "workers")


def _non_negative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, not {value!r}")
    return int(value)


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)


def check_sink_rows(sink: object, dim: int):
    """``sink`` has ``target`` and ``commit`` and takes rows of ``dim`` values of a storage
    dtype, with or without inverse norms."""
    if not callable(getattr(sink, "target", None)) or not callable(getattr(sink, "commit", None)):
        raise TypeError(f"the sink must have target and commit methods, not {type(sink).__name__}")
    if getattr(sink, "dim", None) != dim:
        raise ValueError(f"the sink must take rows of {dim} values")
    sink_dtype(sink)
    wants_inverse_norms(sink)
