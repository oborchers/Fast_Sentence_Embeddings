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

import threading
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _codes
from fse._backends import get_backend
from fse._cpus import usable_cpus
from fse._text import (
    MAX_CHUNK_ROWS,
    Vocabulary,
    check_chunk_bytes,
    make_lookup,
    new_workspace,
)

from ._executor import run
from ._sinks import ArrayRows, GrowingRows, Sink, Stored, sink_dtype
from ._spec import EncodeSpec
from ._tasks import Task, plan

# The finished rows of one task take at most this many bytes, whatever the bytes of its
# sentences (an empty sentence takes none). Smaller tasks cost throughput: a task has a
# fixed cost of about 0.2 ms for its checks (4 MiB lost 7 % on an M1 Pro, 16 MiB 1 %).
TASK_ROW_BYTES = 1 << 24
# The rows of all tasks in flight or waiting to be committed take at most this many bytes:
# the window is 2 * workers tasks, so with many workers a task holds fewer rows. A task
# with steps also holds its pooled rows and one copy while it runs (see the cpu backend).
WINDOW_BYTES = 1 << 29


class Pipeline:
    def __init__(
        self,
        spec: EncodeSpec,
        backend: str = "cpu",
        *,
        isa: str | None = None,
        vocabulary: Vocabulary | None = None,
    ):
        if not isinstance(spec, EncodeSpec):
            raise TypeError(f"spec must be an EncodeSpec, not {type(spec).__name__}")
        self.spec = spec
        self.backend = get_backend(backend, isa=isa)
        self.prepared = self.backend.prepare(spec.pool_input(), spec.steps)
        self.lookup = make_lookup(
            spec.word_vectors, spec.tokenizer, max_ngrams=spec.max_ngrams, vocabulary=vocabulary
        )
        self.output_dim = self.prepared.output_dim
        self._widest = 4 * max(self.prepared.pooled_dim, self.output_dim)
        self.task_rows = max(1, min(MAX_CHUNK_ROWS, TASK_ROW_BYTES // self._widest))
        # One call at a time on a backend that is not concurrent; encoding stays parallel.
        concurrent = self.backend.capabilities().concurrent
        self._serial: AbstractContextManager[Any] = (
            nullcontext() if concurrent else threading.Lock()
        )

    def _row_bytes(self, dtype: str) -> int:
        """The bytes one row of a task takes: its float32 rows while it runs, and for a
        storage dtype also its stored target until its commit."""
        if dtype == "float32":
            return self._widest
        stored = np.dtype(_codes.STORAGE_TYPES[dtype]).itemsize * self.output_dim
        return self._widest + stored + (4 if dtype == "int8" else 0)

    def infer(
        self,
        source: object,
        *,
        workers: int | None = None,
        sink: Sink | None = None,
        chunk_bytes: int | None = None,
        dtype: str = "float32",
    ) -> Stored | None:
        """The rows of every sentence of ``source``: returned (as ``dtype``: float32 rows,
        float16 codes, or int8 codes and scales), or committed to ``sink`` in its dtype.

        Every argument is checked before the source is read.
        """
        workers = usable_cpus() if workers is None else _positive(workers, "workers")
        if sink is not None:
            _check_sink(sink, self.output_dim)
            dtype = sink_dtype(sink)
        else:
            dtype = _codes.check_dtype(dtype)
        # The rows of the window stay within WINDOW_BYTES: with many workers a task holds
        # fewer rows, and with rows wider than the budget allows the window is smaller (at
        # least one task of one row, however wide).
        row_bytes = self._row_bytes(dtype)
        window = max(1, min(2 * workers, WINDOW_BYTES // row_bytes))
        task_rows = max(1, min(self.task_rows, WINDOW_BYTES // (window * row_bytes)))
        stop = threading.Event()  # set by the executor once a task failed
        tasks = plan(
            source,
            workers=workers,
            chunk_bytes=check_chunk_bytes(chunk_bytes),
            chunk_rows=task_rows,
            tokenizer=self.spec.tokenizer,
            stop=stop,
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
            return task, sink.target(task.first_row, task.n_rows)

        def work(item: tuple[Task, Stored]) -> Stored:
            self._work(local, *item)
            return item[1]

        def commit(item: tuple[Task, Stored], rows: Stored):
            sink.commit(item[0].first_row, rows)

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

    def _work(self, local: threading.local, task: Task, target: Stored):
        """Stages 1 to 5 of one task in a worker thread: its rows into ``target``."""
        workspace = getattr(local, "workspace", None)
        if workspace is None:
            workspace = local.workspace = new_workspace()
        batch = task.batch(self.lookup, workspace)
        if isinstance(target, ndarray) and target.dtype == np.float32:
            with self._serial:
                self.backend.rows(batch, self.prepared, target)
            return
        codes, scales = target if isinstance(target, tuple) else (target, None)
        rows = _scratch(local, task.n_rows, self.output_dim)
        with self._serial:
            self.backend.rows(batch, self.prepared, rows)
            self.backend.store(rows, codes, scales, task.first_row)


def _scratch(local: threading.local, n_rows: int, dim: int) -> ndarray:
    """The float32 rows of one task before they are stored, reused by its worker thread."""
    scratch = getattr(local, "scratch", None)
    if scratch is None or len(scratch) < n_rows:
        scratch = local.scratch = np.empty((n_rows, dim), dtype=np.float32)
    return scratch[:n_rows]


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)


def _check_sink(sink: object, dim: int):
    if not callable(getattr(sink, "target", None)) or not callable(getattr(sink, "commit", None)):
        raise TypeError(f"the sink must have target and commit methods, not {type(sink).__name__}")
    if getattr(sink, "dim", None) != dim:
        raise ValueError(f"the sink must take rows of {dim} values")
    sink_dtype(sink)
