#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Where the rows of a group of heads go: one target per head, and one for the vectors
(docs/architecture/runtime.md, "Runtime", Groups of heads).

The runtime reserves and commits in the calling thread, in task order, as for one sink. A
head with a ``sink`` (only a lone head, through the model's own methods) writes there and
numbers its rows from the first row of the run; every other target is an array of the run
that counts from 0. A target may also have a ``tee``: once the rows of a task are finished,
the commit hands them on to it, numbered from the first row of the run (a score sink copies
them into its columns). A tee takes no part in the arrays the run returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import ndarray

from fse._contract import Link

from ._head_common import own, own_vectors, result
from ._tasks import Task


class Scratch:
    """The rows of an extra output: a buffer for each task, dropped at its commit (a tee has
    copied it on by then)."""

    def __init__(self, width: int):
        self.width = width

    def target(self, first_row: int, n_rows: int) -> ndarray:
        del first_row
        return np.empty((n_rows, self.width), np.float32)

    def commit(self, first_row: int, rows: ndarray) -> None:
        pass


class Slots:
    """The output slots of a run: one for the main output of each head, then one more for
    every head with an extra output (``extras``, per head: its output and the tee that gets
    its rows). ``of_member[i]`` lists the slots of head ``i``, the main one first."""

    def __init__(
        self,
        heads: list,
        outputs: tuple[str, ...],
        sinks: list,
        tees: list | None,
        extras: list | None,
    ):
        count = len(outputs)
        given_tees = list(tees or [None] * count)
        given_extras = list(extras or [None] * count)
        if not len(sinks) == len(given_tees) == len(given_extras) == len(heads) == count:
            raise ValueError("a run has one sink, tee and extra output for each head")
        self.heads_of = list(range(count))
        self.of_member: list[list[int]] = [[i] for i in range(count)]
        outs, self.sinks, self.tees = list(outputs), list(sinks), given_tees
        self.extra_kinds = tuple(None if e is None else e[0] for e in given_extras)
        for i, extra in enumerate(given_extras):
            if extra is None:
                continue
            self.of_member[i].append(len(outs))
            self.heads_of.append(i)
            outs.append(extra[0])
            self.sinks.append(Scratch(heads[i].width(extra[0])))
            self.tees.append(extra[1])
        self.outputs = tuple(outs)

    def heads(self, heads: list) -> list:
        """The head of every slot."""
        return [heads[i] for i in self.heads_of]


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
        tees: list | None = None,
    ):
        self.outputs = outputs
        self.base = base
        self.tees = tees or [None] * len(heads)
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
        for tee in self.tees:  # a tee opens its rows of the task now; it gets them at the commit
            if tee is not None:
                tee.target(self.base + task.first_row, task.n_rows)
        vectors = (
            None
            if self.vectors is None
            else self.vectors.target(self.vector_base + task.first_row, task.n_rows)
        )
        return rows, vectors

    def commit(self, task: Task, done: tuple[list, Any]) -> None:
        for target, base, rows, tee in zip(
            self.targets, self.bases, done[0], self.tees, strict=True
        ):
            target.commit(base + task.first_row, rows)
            if tee is not None:
                tee.commit(self.base + task.first_row, rows)
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
