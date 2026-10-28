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
from fse._text import MAX_CHUNK_ROWS, make_lookup, new_workspace

from ._executor import run
from ._sinks import ArrayRows, GrowingRows, Sink
from ._spec import EncodeSpec
from ._tasks import Task, plan

# The finished rows of one task take at most this many bytes, whatever the bytes of its
# sentences (an empty sentence takes none), so a window of tasks stays small.
TASK_ROW_BYTES = 1 << 22


class Pipeline:
    def __init__(self, spec: EncodeSpec, backend: str = "cpu", *, isa: str | None = None):
        if not isinstance(spec, EncodeSpec):
            raise TypeError(f"spec must be an EncodeSpec, not {type(spec).__name__}")
        self.spec = spec
        self.backend = get_backend(backend, isa=isa)
        self.prepared = self.backend.prepare(spec.pool_input(), spec.steps)
        self.lookup = make_lookup(spec.word_vectors, spec.tokenizer, max_ngrams=spec.max_ngrams)
        self.output_dim = self.prepared.output_dim
        widest = 4 * max(self.prepared.pooled_dim, self.output_dim)
        self.task_rows = max(1, min(MAX_CHUNK_ROWS, TASK_ROW_BYTES // widest))
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
        """The rows of every sentence of ``source``: returned, or committed to ``sink``."""
        workers = usable_cpus() if workers is None else _positive(workers, "workers")
        tasks = plan(
            source,
            workers=workers,
            chunk_bytes=chunk_bytes,
            chunk_rows=self.task_rows,
            tokenizer=self.spec.tokenizer,
        )
        own: ArrayRows | GrowingRows | None = None
        if sink is None:
            rows = tasks.rows
            own = GrowingRows(self.output_dim) if rows is None else ArrayRows(rows, self.output_dim)
            sink = own
        elif getattr(sink, "dim", None) != self.output_dim:
            raise ValueError(f"the sink must take rows of {self.output_dim} values")
        local = threading.local()

        def reserved():
            for task in tasks.tasks:  # in the calling thread, in order
                yield task, sink.target(task.first_row, task.n_rows)

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

        run(reserved(), work, commit, workers=workers, inline=tasks.inline)
        return None if own is None else own.result()


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)
