#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Where the rows of a group of heads go: one target per head, and one for the vectors
(docs/architecture/runtime.md, "Runtime", Groups of heads).

The runtime reserves and commits in the calling thread, in task order, as for one sink. A
head with a ``sink`` (only a lone head, through the model's own methods) writes there and
numbers its rows from the first row of the run; every other target is an array of the run
that counts from 0.
"""

from __future__ import annotations

from typing import Any

from numpy import ndarray

from fse._contract import Link

from ._head_common import own, own_vectors, result
from ._tasks import Task


class Targets:
    """The targets of a run of ``rows`` rows (None: a stream) for ``heads`` and their
    ``outputs`` ("logits", "proba" or "labels"), optionally with ``sinks`` (None for an own
    array) and the vectors (``vector_sink`` or an own array of ``vector_dim`` values)."""

    def __init__(
        self,
        heads: list,
        outputs: tuple[str, ...],
        sinks: list,
        rows: int | None,
        *,
        base: int,
        vectors: tuple[int, str, Any] | None,
    ):
        self.outputs = outputs
        self.bases = [base if sink is not None else 0 for sink in sinks]
        self.sinks = sinks
        self.targets: list[Any] = [
            sink
            if sink is not None
            else own(rows, head.width(output), output, head.link is Link.MULTILABEL)
            for head, output, sink in zip(heads, outputs, sinks, strict=True)
        ]
        self.vector_base = 0
        self.vector_sunk = False  # the vectors went to a sink
        self.vectors: Any = None
        if vectors is not None:
            dim, dtype, sink = vectors
            self.vector_sunk = sink is not None
            self.vector_base = base if self.vector_sunk else 0
            self.vectors = sink if self.vector_sunk else own_vectors(rows, dim, dtype)

    def reserve(self, task: Task) -> tuple[list, Any]:
        """The rows of ``task`` in every target."""
        rows = [
            t.target(b + task.first_row, task.n_rows)
            for t, b in zip(self.targets, self.bases, strict=True)
        ]
        vectors = (
            None
            if self.vectors is None
            else self.vectors.target(self.vector_base + task.first_row, task.n_rows)
        )
        return rows, vectors

    def commit(self, task: Task, done: tuple[list, Any]) -> None:
        for target, base, rows in zip(self.targets, self.bases, done[0], strict=True):
            target.commit(base + task.first_row, rows)
        if self.vectors is not None:
            self.vectors.commit(self.vector_base + task.first_row, done[1])

    def results(self) -> tuple[list[ndarray | None], Any]:
        """The arrays of the run: None for a head that had a sink, and the vectors unless a
        sink took them."""
        arrays = [
            None if sink is not None else result(target, output)
            for target, output, sink in zip(self.targets, self.outputs, self.sinks, strict=True)
        ]
        vectors = (
            self.vectors.result() if self.vectors is not None and not self.vector_sunk else None
        )
        return arrays, vectors
