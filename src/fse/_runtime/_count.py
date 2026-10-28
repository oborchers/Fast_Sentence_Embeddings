#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The sentences and tokens of a source, for the sentence length of uSIF.

A pass of stage 1 only, on the executor: every task encodes its chunk with the lookup of
a pipeline and reports the tokens its tokenizer gave, unknown words included (as 1.x
counted them); the counts are added in the calling thread.
"""

from __future__ import annotations

import threading

from fse._cpus import usable_cpus
from fse._text import MAX_CHUNK_ROWS, check_chunk_bytes, new_workspace

from ._executor import run
from ._pipeline import Pipeline, _positive
from ._tasks import Task, plan


def count_tokens(
    pipeline: Pipeline,
    source: object,
    *,
    workers: int | None = None,
    chunk_bytes: int | None = None,
) -> tuple[int, int]:
    """The number of sentences of ``source`` and of the tokens its tokenizer gives."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"pipeline must be a Pipeline, not {type(pipeline).__name__}")
    workers = usable_cpus() if workers is None else _positive(workers, "workers")
    stop = threading.Event()
    tasks = plan(
        source,
        workers=workers,
        chunk_bytes=check_chunk_bytes(chunk_bytes),
        chunk_rows=MAX_CHUNK_ROWS,
        tokenizer=pipeline.spec.tokenizer,
        stop=stop,
    )
    local = threading.local()
    totals = [0, 0]

    def work(task: Task) -> tuple[int, int]:
        workspace = getattr(local, "workspace", None)
        if workspace is None:
            workspace = local.workspace = new_workspace()
        task.batch(pipeline.lookup, workspace)
        return task.n_rows, workspace.tokens_seen

    def commit(_task: Task, counts: tuple[int, int]):
        totals[0] += counts[0]
        totals[1] += counts[1]

    run(tasks.tasks, work, commit, workers=workers, inline=tasks.inline, stop=stop)
    return totals[0], totals[1]
