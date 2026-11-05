#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Linear heads in the runtime: a head ``z = A x + b`` on the rows of an encoder, in the
worker of every task (docs/architecture/runtime.md, "Runtime"; the rule ``linear@1``).

A head reads the rows of one pipeline (its *reader*): for a head on raw means the
pipeline of the folded table (every word and ngram row times A, k columns, made once
when the head is prepared), whose pooled rows get the bias once; for a head on normalized
means, or when the vectors are wanted too, the pipeline of the full mean with the head's
step, whose rows are scored with A and b. With the vectors, one pooling gives both when
the head reads the encoder's own rows; otherwise the vectors come from the encoder's
pipeline, pooled a second time. The logits then become the outputs in the worker:
themselves, the probabilities of the link, or the label indices. The sentence vector of a
task never leaves its worker unless the vectors are asked for.
"""

from __future__ import annotations

import threading
from typing import Any

from numpy import ndarray

from fse import _codes
from fse._backends import get_backend
from fse._contract import Head, Link
from fse._cpus import usable_cpus

from ._errors import earliest, first_rows
from ._executor import run
from ._head_common import buffer, fold, own, own_vectors, result, write_outputs
from ._head_spec import HeadSpec
from ._pipeline import STEP_ROW_BYTES, Pipeline, check_sink_rows, check_workers
from ._sinks import Sink, sink_dtype, split_target, wants_inverse_norms
from ._spec import EncodeSpec
from ._tasks import Task

# What a head gives per row: its logits, the probabilities of its link, or label indices.
OUTPUTS = ("logits", "proba", "labels")


def make_head(
    spec: EncodeSpec,
    head: HeadSpec,
    backend: str,
    *,
    own: Pipeline,
    with_vectors: bool = False,
    workers: int | None = None,
    isa: str | None = None,
) -> HeadPipeline:
    """The head on ``backend`` for the encoder of ``spec``, whose own pipeline there is
    ``own`` (its rows are the vectors): folded where that is legal and no vectors are
    wanted, else on the full mean, sharing ``own`` where the head reads the encoder's rows.
    The fold runs now, over ``workers`` threads (the usable CPUs by default); ``isa``
    selects the kernel of the cpu backend for the pipelines it makes."""
    if own.spec is not spec:
        raise ValueError("own must be the pipeline of spec")
    vocabulary = own.vocabulary
    if head.weight is None:  # standalone: the table of spec is folded already
        if with_vectors:
            raise ValueError("a standalone head has no vectors: its table is folded")
        return HeadPipeline(own, head, folded=True)
    if with_vectors or head.normalized:
        if tuple(spec.steps) == head.steps:
            return HeadPipeline(own, head, folded=False, with_vectors=with_vectors)
        reader = Pipeline(spec, backend, isa=isa, vocabulary=vocabulary, steps=head.steps)
        return HeadPipeline(reader, head, folded=False, vectors=own, with_vectors=with_vectors)
    pool = fold(
        get_backend(backend, isa=isa), spec.pool_input(), head.weight, workers or usable_cpus()
    )
    reader = Pipeline(spec, backend, isa=isa, vocabulary=vocabulary, pool=pool, steps=())
    return HeadPipeline(reader, head, folded=True)


class HeadPipeline:
    """A head on one backend: the pipeline whose rows it reads (``reader``), whether those
    are the folded rows (k columns, the bias still to add) and, for the vectors, the
    pipeline of the encoder's own rows (None: the reader's rows are the vectors)."""

    def __init__(
        self,
        reader: Pipeline,
        head: HeadSpec,
        *,
        folded: bool,
        vectors: Pipeline | None = None,
        with_vectors: bool = False,
    ):
        want = head.k if folded else (head.weight.shape[1] if head.weight is not None else -1)
        if reader.output_dim != want:
            raise ValueError(f"the reader gives rows of {reader.output_dim} values, not {want}")
        if folded and with_vectors:
            raise ValueError("the vectors come from the full mean, not from a folded table")
        self.reader = reader
        self.head = head
        self.folded = folded
        self.with_vectors = with_vectors
        self.vectors = vectors if with_vectors else None
        if self.vectors is not None and self.vectors.lookup.tokenizer != reader.lookup.tokenizer:
            raise ValueError("the vectors and the head must share the tokenizer")

    @property
    def vector_dim(self) -> int:
        return (self.vectors or self.reader).output_dim

    def _row_bytes(self, output: str, vdtype: str, inverse: bool) -> int:
        """The bytes one row of a task takes at most: the rows the head reads (and their
        step), the logits, the output target and, with the vectors, their target and
        scratch as the encoder's pipeline counts them."""
        reader = self.reader.output_dim
        steps = STEP_ROW_BYTES * reader if self.head.normalized and not self.folded else 0
        multi = self.head.link is Link.MULTILABEL
        out = (self.head.k if multi else 8) if output == "labels" else 4 * self.head.width(output)
        total = 4 * reader + steps + 4 * self.head.k + out
        if output == "labels" and self.head.rule is Head.LINEAR2:
            # The probabilities the label is decided on: k for multilabel, else the width
            # of predict_proba (2 for one output).
            total += 4 * self.head.width("proba")
        if self.with_vectors:
            total += (self.vectors or self.reader)._row_bytes(vdtype, inverse)
        return total

    def infer(
        self,
        source: object,
        *,
        output: str,
        workers: int | None = None,
        sink: Sink | None = None,
        vector_sink: Sink | None = None,
        vector_dtype: str = "float32",
        chunk_bytes: int | None = None,
        first_row: int = 0,
    ) -> tuple[ndarray | None, Any]:
        """The ``output`` of every sentence of ``source`` (and, if the head was made with
        vectors, the vectors), returned or committed to ``sink`` (``vector_sink``), numbered
        from ``first_row`` there; errors name the rows from ``first_row``. Every argument
        is checked before the source is read."""
        workers = check_workers(workers) or usable_cpus()
        width = self._check_outputs(output, sink)
        vector_dtype, inverse = self._check_vectors(vector_sink, vector_dtype)
        obase = first_row if sink is not None else 0  # own targets count from 0
        vbase = first_row if vector_sink is not None else 0
        row_bytes = self._row_bytes(output, vector_dtype, inverse)
        tasks, window, stop = self.reader.plan_tasks(
            source, workers, chunk_bytes, row_bytes, first_row
        )
        outputs = (
            sink
            if sink is not None
            else own(tasks.rows, width, output, self.head.link is Link.MULTILABEL)
        )
        vectors: Any = vector_sink
        if self.with_vectors and vectors is None:
            vectors = own_vectors(tasks.rows, self.vector_dim, vector_dtype)
        local = threading.local()

        def reserve(task: Task) -> tuple:
            vtarget = (
                None if vectors is None else vectors.target(vbase + task.first_row, task.n_rows)
            )
            return task, outputs.target(obase + task.first_row, task.n_rows), vtarget

        def work(item: tuple) -> tuple:
            task, target, vtarget = item

            def rows(n: int) -> None:  # an error is that of the earliest failing row
                self._work(
                    local,
                    task.prefix(n),
                    first_rows(target, n),
                    first_rows(vtarget, n),
                    first_row,
                    output,
                    vector_dtype,
                    inverse,
                )

            earliest(rows, task.n_rows)
            return target, vtarget

        def commit(item: tuple, done: tuple) -> None:
            outputs.commit(obase + item[0].first_row, done[0])
            if vectors is not None:
                vectors.commit(vbase + item[0].first_row, done[1])

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
        scores = None if sink is not None else result(outputs, output)
        rows = None if vectors is None or vector_sink is not None else vectors.result()
        return scores, rows

    def _check_outputs(self, output: str, sink: Sink | None) -> int:
        """The values per row of ``output``, checked with ``sink`` (float32, no labels)."""
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, not {output!r}")
        width = self.head.width(output)
        if sink is not None:
            if output == "labels":
                raise ValueError("labels go to no sink; take them from the batches")
            check_sink_rows(sink, width)
            if sink_dtype(sink) != "float32" or wants_inverse_norms(sink):
                raise ValueError("a sink of head outputs takes float32 rows without inverse norms")
        return width

    def _check_vectors(self, vector_sink: Sink | None, dtype: str) -> tuple[str, bool]:
        """The storage dtype of the vectors and whether their target wants inverse norms."""
        if vector_sink is None:
            return _codes.check_dtype(dtype), False
        if not self.with_vectors:
            raise ValueError("a sink of vectors needs a head made with the vectors")
        check_sink_rows(vector_sink, self.vector_dim)
        return sink_dtype(vector_sink), wants_inverse_norms(vector_sink)

    def _work(self, local, task, target, vtarget, base, output, vdtype, inverse) -> None:
        """One task in its worker: the rows the head reads (and the vectors), the logits
        and the output, into ``target`` (and ``vtarget``)."""
        batch = self.reader.batch(local, task, base)
        first = base + task.first_row
        n = task.n_rows
        vectors = vrows = vsplit = None
        if output == "logits" and self.folded:
            rows = target  # pooled in place, the bias added there
        elif self.with_vectors:
            # The vectors are stored (stage 5) after the head (4b), so for one row the
            # head's error comes first.
            vectors, vsplit = self.vectors or self.reader, split_target(vtarget, vdtype, inverse)
            vrows = vectors.pooled(local, batch, vsplit)
            rows = (
                vrows
                if self.vectors is None
                else buffer(local, "head_rows", n, self.reader.output_dim)
            )
        else:
            rows = buffer(local, "head_rows", n, self.reader.output_dim)
        backend = self.reader.backend
        if not (self.with_vectors and self.vectors is None):
            backend.rows(batch, self.reader.prepared, rows)
        logits = target if output == "logits" else buffer(local, "head_logits", n, self.head.k)
        weight = None if self.folded else self.head.weight
        backend.logits(rows, weight, self.head.bias, logits, first)
        if output in ("proba", "labels"):
            write_outputs(backend, local, self.head, logits, target, output, first)
        if vectors is not None:
            assert vrows is not None
            assert vsplit is not None
            vectors.stored(vrows, vsplit, first)
