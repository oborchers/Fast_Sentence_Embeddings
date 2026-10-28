#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The pipeline of an encoder on one backend: sources to finished rows.

Built once per encoder and backend: the backend is prepared and the lookup made before
any input is read, so an unsupported combination fails at once. ``infer`` then plans the
tasks of a source and runs them with the executor; every worker thread encodes its task
(stages 1 and 2) with its own workspace and lets the backend turn the batch into the
finished rows of its task (stages 2 to 4). A Pipeline does not change, so several
threads may call ``infer`` at once (a server: ``workers=1`` each).
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
from numpy import ndarray

from fse._backends import get_backend
from fse._cpus import usable_cpus
from fse._text import MAX_CHUNK_ROWS, check_chunk_bytes, make_lookup, new_workspace

from ._executor import run
from ._sinks import ArrayRows, GrowingRows, Sink
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
    def __init__(self, spec: EncodeSpec, backend: str = "cpu", *, isa: str | None = None):
        if not isinstance(spec, EncodeSpec):
            raise TypeError(f"spec must be an EncodeSpec, not {type(spec).__name__}")
        self.spec = spec
        self.backend = get_backend(backend, isa=isa)
        self.prepared = self.backend.prepare(spec.pool_input(), spec.steps)
        self.lookup = make_lookup(spec.word_vectors, spec.tokenizer, max_ngrams=spec.max_ngrams)
        self.output_dim = self.prepared.output_dim
        self._widest = 4 * max(self.prepared.pooled_dim, self.output_dim)
        self.task_rows = max(1, min(MAX_CHUNK_ROWS, TASK_ROW_BYTES // self._widest))
        # One call at a time on a backend that is not concurrent; encoding stays parallel.
        concurrent = self.backend.capabilities().concurrent
        self._serial: AbstractContextManager[Any] = (
            nullcontext() if concurrent else threading.Lock()
        )

    def infer(
        self,
        source: object,
        *,
        workers: int | None = None,
        sink: Sink | None = None,
        chunk_bytes: int | None = None,
    ) -> ndarray | None:
        """The rows of every sentence of ``source``: returned, or committed to ``sink``.

        Every argument is checked before the source is read.
        """
        workers = usable_cpus() if workers is None else _positive(workers, "workers")
        if sink is not None:
            _check_sink(sink, self.output_dim)
        # The rows of the window stay within WINDOW_BYTES: with many workers a task holds
        # fewer rows, and with rows wider than the budget allows the window is smaller (at
        # least one task of one row, however wide).
        window = max(1, min(2 * workers, WINDOW_BYTES // self._widest))
        task_rows = max(1, min(self.task_rows, WINDOW_BYTES // (window * self._widest)))
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
            own = GrowingRows(self.output_dim) if rows is None else ArrayRows(rows, self.output_dim)
            sink = own
        local = threading.local()

        def reserve(task: Task) -> tuple[Task, ndarray]:  # in the calling thread, in order
            return task, sink.target(task.first_row, task.n_rows)

        def work(item: tuple[Task, ndarray]) -> ndarray:
            task, target = item
            workspace = getattr(local, "workspace", None)
            if workspace is None:
                workspace = local.workspace = new_workspace()
            batch = task.batch(self.lookup, workspace)
            with self._serial:
                self.backend.rows(batch, self.prepared, target)
            return target

        def commit(item: tuple[Task, ndarray], rows: ndarray):
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


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)


def _check_sink(sink: object, dim: int):
    if not callable(getattr(sink, "target", None)) or not callable(getattr(sink, "commit", None)):
        raise TypeError(f"the sink must have target and commit methods, not {type(sink).__name__}")
    if getattr(sink, "dim", None) != dim:
        raise ValueError(f"the sink must take rows of {dim} values")
