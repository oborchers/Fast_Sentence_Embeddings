#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Linear heads as a group in the runtime: the heads ``z_j = A_j x + b_j`` of one encoder
scored in one pass over the texts (docs/architecture/runtime.md, "Runtime", Groups of
heads; the rule ``linear@1``).

The worker of a task tokenizes it once and pools what the heads read: the shared folded
table for the raw heads (one pooling of K_F columns), the full rows for the heads on
normalized means (one pooling, one ``normalize@1``) and, with the vectors, the raw rows
that give them. Then every head turns its rows into its logits and outputs with the
functions of its own path, in keyword order, and the vectors are stored last. A lone head
is a group of one: the same code, the same bits.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import Normalize
from fse._cpus import usable_cpus

from ._errors import earliest, first_rows
from ._executor import run
from ._head_common import buffer, write_outputs
from ._heads_errors import FULL, NORMED, TABLE, PoolingError, attribute
from ._heads_plan import (
    FOLDED,
    NO_VECTORS,
    NORMALIZED,
    OWN_ROWS,
    STANDALONE,
    HeadsPlan,
)
from ._heads_targets import Targets
from ._pipeline import check_workers
from ._sinks import Sink, split_target
from ._tasks import Task

NORMALIZE = (Normalize(),)
F32 = np.float32


class Rows:
    """The rows the heads of a task read: the folded (or standalone) rows, the normalized
    rows, the raw full rows and, with the vectors, the vector rows."""

    __slots__ = ("folded", "normed", "raw", "vectors")

    def __init__(self, folded=None, normed=None, raw=None, vectors=None):
        self.folded, self.normed, self.raw, self.vectors = folded, normed, raw, vectors


class HeadsPipeline:
    """The heads of a ``HeadsPlan`` on one backend."""

    def __init__(self, plan: HeadsPlan):
        self.plan = plan

    @property
    def vector_dim(self) -> int:
        return self.plan.own.output_dim

    def infer(
        self,
        source: object,
        *,
        outputs: tuple[str, ...],
        workers: int | None = None,
        sinks: list[Sink | None] | None = None,
        vector_sink: Sink | None = None,
        vector_dtype: str = "float32",
        inverse: bool = False,
        chunk_bytes: int | None = None,
        first_row: int = 0,
    ) -> tuple[list[ndarray | None], Any]:
        """The ``outputs`` ("logits", "proba" or "labels" per head) of every sentence of
        ``source`` (and, if the plan has the vectors, the vectors, as ``vector_dtype``,
        with the inverse norms a ``vector_sink`` wants if ``inverse``), returned or
        committed to the sinks, numbered from ``first_row`` there; errors name the rows
        from ``first_row``. The caller checked every argument but ``workers`` before the
        source is read."""
        plan = self.plan
        workers = check_workers(workers) or usable_cpus()
        sinks = sinks or [None] * len(plan.members)
        row_bytes = plan.row_bytes(outputs, vector_dtype, inverse)
        tasks, window, stop = plan.primary.plan_tasks(
            source, workers, chunk_bytes, row_bytes, first_row
        )
        heads = [m.head for m in plan.members]
        vectors = (self.vector_dim, vector_dtype, vector_sink) if plan.with_vectors else None
        targets = Targets(heads, outputs, sinks, tasks.rows, base=first_row, vectors=vectors)
        local = threading.local()

        def work(item: tuple) -> tuple:
            task, (rows, vrows) = item

            def run_rows(n: int) -> None:  # an error is that of the earliest failing row
                try:
                    self._work(
                        local,
                        task.prefix(n),
                        [first_rows(t, n) for t in rows],
                        first_rows(vrows, n),
                        first_row,
                        outputs,
                        vector_dtype,
                        inverse,
                    )
                except PoolingError as error:
                    error.prefix = n
                    raise

            try:
                earliest(run_rows, task.n_rows)
            except PoolingError as error:
                raise attribute(plan, error, task, local, first_row) from None
            return rows, vrows

        run(
            tasks.tasks,
            work,
            lambda item, done: targets.commit(item[0], done),
            workers=workers,
            inline=tasks.inline,
            window=window,
            reserve=lambda task: (task, targets.reserve(task)),
            stop=stop,
        )
        return targets.results()

    # --- one task -------------------------------------------------------------------

    def _work(self, local, task: Task, rows, vrows, base, outputs, vdtype, inverse) -> None:
        """One task in its worker: the rows the heads read (and the vectors), then each
        head's logits and output into its target (and the vectors into ``vrows``)."""
        plan = self.plan
        batch = plan.primary.batch(local, task, base)
        first = base + task.first_row
        direct = self._direct(rows, outputs)
        vsplit = split_target(vrows, vdtype, inverse) if plan.with_vectors else None
        read = self._read(local, batch, task.n_rows, first, direct, vsplit)
        for i, output in enumerate(outputs):
            self._head(local, i, read, rows[i], output, first)
        if vsplit is not None:
            # The vectors are stored (stage 5) after the heads (4b): for one row the heads'
            # error comes first.
            plan.own.stored(read.vectors, vsplit, first)

    def _direct(self, rows: list, outputs: tuple[str, ...]) -> ndarray | None:
        """The target of the logits of a lone folded head: pooled in place, the bias added
        there."""
        plan = self.plan
        if len(rows) == 1 and outputs[0] == "logits" and plan.kinds[0] in (FOLDED, STANDALONE):
            return rows[0]
        return None

    def _read(self, local, batch, n: int, first: int, direct, vsplit) -> Rows:
        """Stages 2 to 4 of the batch: the rows the heads read, pooled once each."""
        plan = self.plan
        folded = None
        if plan.table is not None:
            folded = (
                direct
                if direct is not None
                else buffer(local, "head_table", n, plan.table.output_dim)
            )
            self._pool(TABLE, plan.table, batch, folded)
        if plan.vector_mode == NO_VECTORS:
            normed = None
            if plan.normed is not None:
                normed = buffer(local, "head_normed", n, plan.normed.output_dim)
                self._pool(NORMED, plan.normed, batch, normed)
            return Rows(folded, normed)
        return self._read_full(local, batch, n, first, vsplit, folded)

    def _read_full(self, local, batch, n: int, first: int, vsplit, folded) -> Rows:
        """The rows of the full table with the vectors: one pooling gives them, and a copy
        is normalized once for the heads on normalized means where the pooling is raw."""
        plan = self.plan
        dim = plan.own.output_dim
        if plan.vector_mode == OWN_ROWS:
            rows = self._guard(FULL, plan.own.pooled, local, batch, vsplit)
            if plan.encoder_normalizes:  # the vectors are the normalized rows
                return Rows(folded, rows, None, rows)
            normed = self._copy(local, rows, n, first) if plan.indices(NORMALIZED) else None
            return Rows(folded, normed, rows, rows)
        raw = buffer(local, "head_rows", n, dim)  # the encoder normalizes, a head reads raw rows
        self._pool(FULL, plan.full, batch, raw)
        codes = vsplit[0]
        vectors = codes if codes.dtype == F32 else buffer(local, "head_vectors", n, dim)
        vectors[...] = raw
        self._guard(
            FULL, plan.own.backend.postprocess_pooled, vectors, plan.own.prepared.steps, first
        )
        return Rows(folded, vectors, raw, vectors)

    def _copy(self, local, rows: ndarray, n: int, first: int) -> ndarray:
        """``rows`` normalized in a copy."""
        copy = buffer(local, "head_normed", n, rows.shape[1])
        copy[...] = rows
        self._guard(FULL, self.plan.own.backend.postprocess_pooled, copy, NORMALIZE, first)
        return copy

    def _pool(self, table: str, pipeline, batch, out: ndarray) -> None:
        self._guard(table, pipeline.backend.rows, batch, pipeline.prepared, out)

    def _guard(self, table: str, call, *args) -> Any:
        """``call(*args)``; in a group, a pooling error is raised as ``PoolingError`` of ``table``
        so that the heads it belongs to can be named."""
        try:
            return call(*args)
        except ValueError as error:
            if not self.plan.named:
                raise
            raise PoolingError(table, error) from None

    def _head(self, local, i: int, read: Rows, target: ndarray, output: str, first: int) -> None:
        """The logits of head ``i`` from the rows it reads, and its output into ``target``."""
        plan = self.plan
        member, kind = plan.members[i], plan.kinds[i]
        head, n = member.head, len(target)
        try:
            source, weight = self._source(local, i, kind, read, n)
            backend = plan.primary.backend
            logits = target if output == "logits" else buffer(local, f"head_logits{i}", n, head.k)
            backend.logits(source, weight, head.bias, logits, first)
            if output in ("proba", "labels"):
                write_outputs(backend, local, head, logits, target, output, first, f"{i}")
        except ValueError as error:
            if not plan.named:
                raise
            raise ValueError(f"head {member.name!r}: {error}") from None

    def _source(self, local, i: int, kind: str, read: Rows, n: int) -> tuple[ndarray, Any]:
        """The rows head ``i`` of ``kind`` reads and its weight (None: the rows are folded)."""
        head = self.plan.members[i].head
        if kind == STANDALONE or (kind == FOLDED and len(self.plan.indices(FOLDED)) == 1):
            return read.folded, None
        if kind == FOLDED:  # the columns of the head, in a buffer of their own
            start = self.plan.starts[i]
            columns = buffer(local, f"head_columns{i}", n, head.k)
            columns[...] = read.folded[:, start : start + head.k]
            return columns, None
        return (read.normed if kind == NORMALIZED else read.raw), head.weight
