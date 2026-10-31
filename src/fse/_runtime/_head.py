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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _codes, _log
from fse._backends import PoolInput, get_backend
from fse._contract import Link, Normalize, Step
from fse._cpus import usable_cpus

from ._errors import earliest, first_rows
from ._executor import run
from ._pipeline import STEP_ROW_BYTES, Pipeline, check_sink_rows, check_workers
from ._sinks import ArrayRows, GrowingRows, Sink, sink_dtype, split_target, wants_inverse_norms
from ._spec import EncodeSpec
from ._tasks import Task

F32 = np.float32
# What a head gives per row: its logits, the probabilities of its link, or label indices.
OUTPUTS = ("logits", "proba", "labels")
# The rows of a table folded at once by one task of the fold.
_FOLD_ROWS = 1 << 14


@dataclass(frozen=True, eq=False)
class HeadSpec:
    """A linear head: ``weight`` (k, d) float32 and ``bias`` (k,), or for a standalone head
    (whose table is folded already) no weight; ``normalized``: it reads normalized rows;
    ``link``: the link of its probabilities, or None. Owns read-only copies."""

    weight: ndarray | None
    bias: ndarray
    normalized: bool
    link: Link | None

    def __post_init__(self):
        bias = _frozen(self.bias, "bias", 1)
        object.__setattr__(self, "bias", bias)
        if self.weight is not None:
            weight = _frozen(self.weight, "weight", 2)
            if len(weight) != len(bias):
                raise ValueError(f"weight has {len(weight)} rows, but bias {len(bias)} values")
            object.__setattr__(self, "weight", weight)
        elif self.normalized:
            raise ValueError("a standalone head reads raw means; it cannot be normalized")
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be a bool")
        if self.link is not None and not isinstance(self.link, Link):
            raise TypeError("link must be a Link or None")

    @property
    def k(self) -> int:
        return len(self.bias)

    @property
    def steps(self) -> tuple[Step, ...]:
        """The step of the rows the head reads unfolded."""
        return (Normalize(),) if self.normalized else ()

    def width(self, output: str) -> int:
        """The values per row of ``output``: k logits, 2 probabilities for k = 1 (else k),
        one label index."""
        if output == "labels":
            return 1
        if output == "proba":
            if self.link is None:
                raise ValueError("the head has no link, so no probabilities")
            return 2 if self.k == 1 else self.k
        return self.k


def _frozen(value: object, name: str, ndim: int) -> ndarray:
    if not isinstance(value, ndarray) or value.dtype != F32 or value.ndim != ndim or not value.size:
        raise ValueError(f"{name} must be a non-empty float32 array of {ndim} dimensions")
    copy = np.array(value, copy=True, order="C")
    if not np.isfinite(copy).all():
        raise ValueError(f"{name} must be finite")
    copy.setflags(write=False)
    return copy


def fold(backend: Any, pool: PoolInput, weight: ndarray, workers: int) -> PoolInput:
    """The pool of the folded table: every word and ngram row of ``pool`` scored with every
    row of ``weight`` (``score(row, scale, a_j, 0)``, k float32 columns, no scales), in
    blocks of rows over ``workers`` threads; weights and ``oov_weight`` are kept."""
    start = time.perf_counter()
    vectors = _fold_rows(backend, pool.vectors, pool.scales, weight, workers, "word")
    ngrams = None
    if pool.ngram_vectors is not None:
        ngrams = _fold_rows(
            backend, pool.ngram_vectors, pool.ngram_scales, weight, workers, "ngram"
        )
    rows = len(vectors) + (0 if ngrams is None else len(ngrams))
    _log.emit(
        "head.folded",
        "fse runtime: folded a head of %d outputs into %d rows of the table",
        len(weight),
        rows,
        rows=rows,
        outputs=len(weight),
        workers=workers,
        backend=backend.name,
        seconds=time.perf_counter() - start,
    )
    return PoolInput(vectors, pool.weights, None, ngrams, None, pool.oov_weight, pool.aggregation)


def _fold_rows(backend, stored, scales, weight, workers, kind) -> ndarray:
    out = np.empty((len(stored), len(weight)), dtype=F32)

    def block(begin: int) -> None:
        end = min(len(stored), begin + _FOLD_ROWS)
        part = None if scales is None else scales[begin:end]
        try:
            out[begin:end] = backend.scores(stored[begin:end], part, weight, 0.0)
        except ValueError as error:
            raise ValueError(
                f"the head does not fold into the table: {kind} rows {begin} to {end - 1}: {error}"
            ) from None

    starts = range(0, len(stored), _FOLD_ROWS)
    if workers == 1 or len(starts) < 2:
        for begin in starts:
            block(begin)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fse-fold") as pool:
            for result in [pool.submit(block, begin) for begin in starts]:
                result.result()  # the first failing block, in order
    out.setflags(write=False)
    return out


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
        out = 8 if output == "labels" else 4 * self.head.width(output)
        total = 4 * reader + steps + 4 * self.head.k + out
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
        outputs = sink if sink is not None else _own(tasks.rows, width, output)
        vectors: Any = vector_sink
        if self.with_vectors and vectors is None:
            vectors = _own_vectors(tasks.rows, self.vector_dim, vector_dtype)
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
        result = None if sink is not None else _result(outputs, output)
        rows = None if vectors is None or vector_sink is not None else vectors.result()
        return result, rows

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
                else _buffer(local, "head_rows", n, self.reader.output_dim)
            )
        else:
            rows = _buffer(local, "head_rows", n, self.reader.output_dim)
        backend = self.reader.backend
        if not (self.with_vectors and self.vectors is None):
            backend.rows(batch, self.reader.prepared, rows)
        logits = target if output == "logits" else _buffer(local, "head_logits", n, self.head.k)
        weight = None if self.folded else self.head.weight
        backend.logits(rows, weight, self.head.bias, logits, first)
        if output == "proba":
            backend.outputs(logits, "proba", self.head.link, target, first)
        elif output == "labels":
            backend.outputs(logits, "labels", None, target, first)
        if vectors is not None:
            assert vrows is not None
            assert vsplit is not None
            vectors.stored(vrows, vsplit, first)


def _buffer(local: threading.local, name: str, n_rows: int, dim: int) -> ndarray:
    """A float32 (n_rows, dim) buffer of the worker thread, reused across its tasks."""
    buffer = getattr(local, name, None)
    if buffer is None or len(buffer) < n_rows or buffer.shape[1] != dim:
        buffer = np.empty((n_rows, dim), dtype=F32)
        setattr(local, name, buffer)
    return buffer[:n_rows]


class _Labels:
    """The label indices (int64) of a run: one array of a known number of rows, or parts."""

    dim = 1

    def __init__(self, rows: int | None):
        self.next_row = 0
        self._out = None if rows is None else np.empty(rows, dtype=np.int64)
        self._parts: list[ndarray] = []

    def target(self, first_row: int, n_rows: int) -> ndarray:
        if self._out is None:
            return np.empty(n_rows, dtype=np.int64)
        return self._out[first_row : first_row + n_rows]

    def commit(self, first_row: int, rows: ndarray) -> None:
        if first_row != self.next_row:
            raise ValueError(f"rows from {first_row} committed, expected rows from {self.next_row}")
        self.next_row += len(rows)
        if self._out is None:
            self._parts.append(rows)

    def result(self) -> ndarray:
        if self._out is not None:
            return self._out
        return np.concatenate(self._parts) if self._parts else np.empty(0, dtype=np.int64)


def _own(rows: int | None, width: int, output: str) -> Any:
    if output == "labels":
        return _Labels(rows)
    return GrowingRows(width) if rows is None else ArrayRows(rows, width)


def _own_vectors(rows: int | None, dim: int, dtype: str) -> Any:
    return GrowingRows(dim, dtype) if rows is None else ArrayRows(rows, dim, dtype)


def _result(outputs: Any, output: str) -> ndarray:
    result = outputs.result()
    return result if output == "labels" else np.asarray(result)
