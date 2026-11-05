#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The run of a group of heads into a score sink (docs/architecture/scores.md, "Writing").

The sink is a float32 sink of the columns of all heads side by side. The workers fill the
targets of the heads as ever, each in its own contiguous array; this module joins them to
the matrix of the sink. For each head, the stored output (the logits or the probabilities)
is either the output the batch returns, then its target gets a tee, or a second output the
worker derives from the same logits into a scratch buffer, which gets the tee. When the
rows of a task are finished, the commit in the calling thread copies each tee into the
columns of its head in the matrix the sink reserved for the task, and the last tee of the
task commits the matrix. The copy is in the calling thread, so the budgets of the
worker threads do not change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from numpy import ndarray

from fse._stream import StreamSource
from fse._text._ids import IdSource

# What the runtime computes for a stored output.
STORED_RUNTIME = {"decision": "logits", "values": "logits", "proba": "proba"}


def attach(sink: Any, heads: Mapping[str, Any], paths: Mapping[str, str], doc_ids: bool) -> None:
    """Check ``sink`` against ``heads`` and the document-ID mode, then attach the run with
    the ``paths`` of the heads. Raises before anything is read."""
    schema = getattr(sink, "score_schema", None)
    if schema is None or not callable(getattr(sink, "_attach", None)):
        raise TypeError(f"a score_sink must be a fse.ScoreSink, not {type(sink).__name__}")
    names = [head.name for head in schema.heads]
    if names != list(heads):
        raise ValueError(f"the sink was made for the heads {names}, not for {list(heads)}")
    for head, (name, model) in zip(schema.heads, heads.items(), strict=True):
        if head.model != model.fingerprint:
            raise ValueError(f"head {name!r} is not the model the sink was made for")
    if schema.encoder != next(iter(heads.values())).encoder_fingerprint:
        raise ValueError("the heads read another encoder than the sink was made for")
    if doc_ids != sink.has_doc_ids:
        raise ValueError(
            "doc_ids=True is needed exactly when the sink has doc_ids; pass (texts, doc_ids) "
            "batches to a sink with doc_ids"
        )
    sink._attach(paths)


class _Column:
    """The tee of one head: it takes the finished rows of a task at the commit."""

    def __init__(self, run: ScoreRun, index: int):
        self.run, self.index = run, index

    def target(self, first_row: int, n_rows: int) -> None:
        self.run._open_task(first_row, n_rows)

    def commit(self, first_row: int, rows: ndarray) -> None:
        self.run._copy(self.index, first_row, rows)


class ScoreRun:
    """The batches of a ``predict_batches`` call into one score sink.

    ``kinds`` is the runtime output the batch returns for each head ("logits", "proba" or
    "labels"). ``tees`` and ``extras`` are what ``HeadsPipeline.infer`` takes to write the
    stored output of each head; ``tee_bytes`` is what a row holds in the sink matrix and the
    document IDs beyond the outputs themselves."""

    def __init__(self, sink: Any, kinds: tuple[str, ...]):
        self.sink = sink
        schema = sink.score_schema
        self.segments = [slice(*head.columns) for head in schema.heads]
        self.doc_ids = bool(sink.has_doc_ids)
        self.tees: list[Any] = [None] * len(kinds)
        self.extras: list[Any] = [None] * len(kinds)
        for i, (head, kind) in enumerate(zip(schema.heads, kinds, strict=True)):
            stored, column = STORED_RUNTIME[head.stored], _Column(self, i)
            if stored == kind:
                self.tees[i] = column
            else:
                self.extras[i] = (stored, column)
        self.tee_bytes = 4 * sink.dim + (8 if self.doc_ids else 0)
        self._open: dict[int, list] = {}
        self._values: IdSource | None = None
        self._base = self._end = 0

    def infer(
        self, pipeline: Any, source: Any, kinds: tuple[str, ...], workers: Any, row: int, ids: Any
    ):
        """The batch ``source`` starting at ``row`` of the run (``(arrays, vectors)`` of
        the pipeline); ``ids`` are its document IDs, given exactly if the sink has them."""
        self._open.clear()  # nothing of an earlier batch that failed
        arguments = {
            "outputs": kinds,
            "workers": workers,
            "first_row": row,
            "tees": self.tees,
            "extras": self.extras,
            "tee_bytes": self.tee_bytes,
        }
        if ids is None:
            self._values = None
            return pipeline.infer(source, **arguments)
        with IdSource(ids) as values:
            if not isinstance(source, StreamSource) and len(source) != len(values):
                raise ValueError(f"doc_ids has {len(values)} rows for {len(source)} texts")
            self._values, self._base, self._end = values, row, row
            found = pipeline.infer(source, **arguments)
            if self._end != row + len(values):
                raise ValueError("doc_ids has more rows than the text stream")
        return found

    # --- the tees -------------------------------------------------------------------

    def _open_task(self, first_row: int, n_rows: int) -> None:
        """The matrix and the IDs of the task, reserved once for all its tees."""
        if first_row in self._open:
            return
        ids = None
        if self._values is not None:
            ids = self._values.block(first_row - self._base, first_row - self._base + n_rows)
        self._open[first_row] = [self.sink.target(first_row, n_rows), ids, len(self.segments)]
        self._end = first_row + n_rows

    def _copy(self, index: int, first_row: int, rows: ndarray) -> None:
        entry = self._open[first_row]
        matrix, ids, _ = entry
        matrix[:, self.segments[index]] = rows
        entry[2] -= 1
        if entry[2] == 0:  # the last tee of the task
            del self._open[first_row]
            if ids is None:
                self.sink.commit(first_row, matrix)
            else:
                self.sink.commit(first_row, matrix, doc_ids=ids)
