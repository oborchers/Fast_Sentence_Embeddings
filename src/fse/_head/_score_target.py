#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The run of heads, a group or a single one, into a score sink (docs/architecture/scores.md, "Writing").

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

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from numpy import ndarray

from fse._stream import StreamSource
from fse._text._ids import IdSource, IdWindow

# What the runtime computes for a stored output.
STORED_RUNTIME = {"decision": "logits", "values": "logits", "proba": "proba"}


def check_score_sink(sink: Any, heads: Mapping[str, Any], *, resumed: bool = False) -> None:
    """Check ``sink`` against ``heads``; a sink made with ``resume=True`` is refused unless
    ``resumed`` says the call resumes (``predict``). Raises before anything is read."""
    schema = getattr(sink, "_score_schema", None)
    if schema is None or not callable(getattr(sink, "_attach", None)):
        raise TypeError(f"a score_sink must be a fse.ScoreSink, not {type(sink).__name__}")
    if not resumed and getattr(sink, "resume", False) is True:
        raise ValueError(
            "a score_sink made with resume=True takes the rows of predict(texts, score_sink=sink) "
            "of a source that can be read again, or batches via "
            "predict_batches(..., positions=True, score_sink=sink); "
            "otherwise use a sink without resume for predict_batches"
        )
    names = [head.name for head in schema.heads]
    if names != list(heads):
        raise ValueError(f"the sink was made for the heads {names}, not for {list(heads)}")
    for head, (name, model) in zip(schema.heads, heads.items(), strict=True):
        if head.model != model.fingerprint:
            raise ValueError(f"head {name!r} is not the model the sink was made for")
    if schema.encoder != next(iter(heads.values())).encoder_fingerprint:
        raise ValueError("the heads read another encoder than the sink was made for")


@contextmanager
def _opened(ids: Any) -> Iterator[Any]:
    """``ids`` as a source of IDs: a window of a resumed run is open already."""
    if isinstance(ids, IdWindow):
        yield ids
        return
    with IdSource(ids) as values:
        yield values


class TaskIds:
    """The document IDs of the tasks of a batch, fetched once per task for all ``users``
    (the sinks that store them). The first reservation of a task fetches its block; every
    user gets the same read-only array and ``release`` is called once per user, at its
    commit. A run without IDs gives None."""

    def __init__(self, users: int):
        self.users = users
        self._values: IdSource | None = None
        self._open: dict[int, list] = {}
        self._base = self._end = 0

    @contextmanager
    def batch(self, source: Any, row: int, ids: Any) -> Iterator[None]:
        """The IDs ``ids`` of the batch ``source`` that starts at ``row`` of the run (None:
        the run has none), checked against the rows of the texts once the batch ran."""
        self._open.clear()  # nothing of an earlier batch that failed
        self._values = None
        if ids is None:
            yield
            return
        with _opened(ids) as values:
            if not isinstance(source, StreamSource) and len(source) != len(values):
                raise ValueError(f"doc_ids has {len(values)} rows for {len(source)} texts")
            self._values, self._base, self._end = values, row, row
            yield
            if self._end != row + len(values):
                raise ValueError("doc_ids has more rows than the text stream")
        self._values = None

    def block(self, first_row: int, n_rows: int) -> Any:
        """The IDs of the task (None without IDs); one fetch per task."""
        if self._values is None:
            return None
        entry = self._open.get(first_row)
        if entry is None:
            begin = first_row - self._base
            entry = self._open[first_row] = [self._values.block(begin, begin + n_rows), self.users]
            self._end = first_row + n_rows
        return entry[0]

    def release(self, first_row: int) -> None:
        """One user has committed the task; the block is dropped after the last."""
        entry = self._open.get(first_row)
        if entry is not None:
            entry[1] -= 1
            if entry[1] == 0:
                del self._open[first_row]


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

    def __init__(self, sink: Any, kinds: tuple[str, ...], ids: TaskIds | None = None):
        self.sink = sink
        self.ids = ids or TaskIds(1)
        schema = sink._score_schema
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

    def reset(self) -> None:
        """Forget the tasks of an earlier batch that failed."""
        self._open.clear()

    # --- the tees -------------------------------------------------------------------

    def _open_task(self, first_row: int, n_rows: int) -> None:
        """The matrix and the IDs of the task, reserved once for all its tees."""
        if first_row in self._open:
            return
        ids = self.ids.block(first_row, n_rows)
        self._open[first_row] = [self.sink.target(first_row, n_rows), ids, len(self.segments)]

    def _copy(self, index: int, first_row: int, rows: ndarray) -> None:
        entry = self._open[first_row]
        matrix, ids, _ = entry
        matrix[:, self.segments[index]] = rows
        entry[2] -= 1
        if entry[2] == 0:  # the last tee of the task
            del self._open[first_row]
            self.ids.release(first_row)
            if ids is None:
                self.sink.commit(first_row, matrix)
            else:
                self.sink.commit(first_row, matrix, doc_ids=ids)
