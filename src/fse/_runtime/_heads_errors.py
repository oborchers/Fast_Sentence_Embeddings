#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Which head an error of a group belongs to (docs/architecture/runtime.md, "Runtime",
Groups of heads).

A lone head keeps its errors as they are. In a group the pooling is shared, so its errors
do not tell the head: a pooling error of a group is raised as ``PoolingError`` (a ValueError with
the table that failed and the rows that were being pooled), and ``attribute`` turns it into
the error of the heads it belongs to. This costs nothing until a task fails. The error of
the shared folded table is attributed by pooling the failing row again with the folded table
of each head alone (the pooling is defined per value, so exactly the heads whose columns
overflow fail); the other tables name the heads that read them.
"""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np

from fse._backends import PoolInput

from ._heads_plan import FOLDED, NORMALIZED, RAW, STANDALONE, HeadsPlan
from ._pipeline import Pipeline
from ._tasks import Task

# The tables a group pools.
TABLE, FULL, NORMED, COPY, POST = "table", "full", "normed", "copy", "post"


class PoolingError(ValueError):
    """A pooling error of a group: ``table`` names the table that failed, ``prefix`` is the
    number of rows of the task that were being pooled (set by the task that raised it)."""

    def __init__(self, table: str, error: ValueError):
        super().__init__(str(error))
        self.table = table
        self.prefix = 0


def attribute(
    plan: HeadsPlan, error: PoolingError, task: Task, local: threading.local, base: int
) -> ValueError:
    """The error of ``error`` with the heads it belongs to named."""
    if error.table == TABLE and plan.indices(STANDALONE):
        who = plan.who(STANDALONE)
    elif error.table == TABLE:
        who = _folded_heads(plan, error, task, local, base)
    elif error.table in (NORMED, COPY):
        who = plan.who(NORMALIZED)
    elif error.table == POST:
        who = plan.who(NORMALIZED, vectors=True)
    else:
        who = plan.who(NORMALIZED, RAW, vectors=True)
    return ValueError(f"{who}: {error}")


def _folded_heads(plan: HeadsPlan, error: PoolingError, task, local, base: int) -> str:
    """The heads whose own folded table fails on the rows that failed in the shared one."""
    spec = plan.spec
    batch = plan.primary.batch(local, task.prefix(error.prefix), base)
    failing = []
    for i in plan.indices(FOLDED):
        head = plan.members[i].head
        assert head.weight is not None
        try:
            pool = _columns(plan, i)
            alone = Pipeline(
                spec,
                plan.backend_name,
                isa=plan.isa,
                vocabulary=plan.own.vocabulary,
                pool=pool,
                steps=(),
            )
            alone.backend.rows(batch, alone.prepared, np.empty((batch.n_sentences, head.k), "f4"))
        except ValueError:
            failing.append(i)
    return plan.who(FOLDED) if not failing else _names(plan, failing)


def _columns(plan: HeadsPlan, index: int) -> PoolInput:
    """A contiguous copy of one head's already folded columns, with no new fold."""
    assert plan.table is not None
    pool = plan.table.prepared.pool
    first = plan.starts[index]
    columns = slice(first, first + plan.members[index].head.k)
    ngrams = pool.ngram_vectors
    return replace(
        pool,
        vectors=np.ascontiguousarray(pool.vectors[:, columns]),
        ngram_vectors=None if ngrams is None else np.ascontiguousarray(ngrams[:, columns]),
    )


def _names(plan: HeadsPlan, indices: list[int]) -> str:
    names = [f"'{plan.members[i].name}'" for i in indices]
    return ("head " if len(names) == 1 else "heads ") + ", ".join(names)
