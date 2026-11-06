#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The sinks of a run of a group of heads: a ``vector_sink`` for the vectors of the encoder
and a ``score_sink`` for the scores of every head, each alone or both in one pass
(docs/architecture/scores.md, "Vectors and scores in one run").

This module checks the sinks before a text is read, shares the document IDs of each task
between them, and reports the rows that are durable in all of them: the minimum of the
counts of the sinks, each checked on its own. It adds no protocol between the sinks: each
folder is a valid prefix on its own, and what exceeds the confirmed rows may be in one folder
only. The runtime (``fse._runtime``) is not changed: it takes a vector sink and tees already.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fse._encoder._batches import check_sink, durable, flush
from fse._encoder._doc_ids import id_mode
from fse._runtime._sinks import sink_dtype, wants_inverse_norms
from fse._store._base_sink import ShardSinkBase

from ._score_target import ScoreRun, TaskIds, check_score_sink


def start(
    heads: Any,
    kinds: tuple[str, ...],
    return_vectors: bool,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: bool,
) -> SinkRun | None:
    """Check the sinks against ``heads`` and the document-ID mode, attach the run to the
    score sink and return the run (None without a sink). Raises before a text is read."""
    if vector_sink is None and score_sink is None:
        if doc_ids:
            raise ValueError("doc_ids=True needs a vector_sink or a score_sink with doc_ids")
        return None
    if vector_sink is not None:
        _check_vector_sink(heads, return_vectors, vector_sink)
    if score_sink is not None:
        check_score_sink(score_sink, heads)
    _check_modes(vector_sink, score_sink, doc_ids)
    _check_apart(vector_sink, score_sink)
    if score_sink is not None:
        if vector_sink is not None and getattr(score_sink, "_exists", False):
            _check_rows(vector_sink, score_sink, doc_ids)
        score_sink._attach(heads.paths(return_vectors=return_vectors))
        if vector_sink is not None:  # _attach may have set first_row anew
            _check_rows(vector_sink, score_sink, doc_ids)
    return SinkRun(kinds, vector_sink, score_sink, doc_ids)


def _check_vector_sink(heads: Any, return_vectors: bool, sink: Any) -> None:
    if not return_vectors:
        raise ValueError(
            "vector_sink takes the vectors of return_vectors=True; pass return_vectors=True "
            "(the heads on raw means then take the unfolded path, see Heads.paths)"
        )
    dim = heads._first.encoder.output_dim
    hint = f"; the vector_sink takes the vectors of the encoder ({dim} values)"
    check_sink(sink, None, heads.encoder_fingerprint, dim, hint)
    if isinstance(sink, ShardSinkBase):
        sink._check_fresh()


def _check_modes(vector_sink: Any, score_sink: Any, doc_ids: bool) -> None:
    """``doc_ids`` must be True exactly when each sink given stores document IDs."""
    for name, sink in (("vector_sink", vector_sink), ("score_sink", score_sink)):
        if sink is None:
            continue
        stores = id_mode(sink)
        if doc_ids and not stores:
            value = getattr(sink, "has_doc_ids", False)
            if not isinstance(value, bool):
                raise TypeError(
                    f"has_doc_ids of the {name} must be True for doc_ids, not {type(value).__name__}"
                )
        if doc_ids != stores:
            raise ValueError(
                f"doc_ids=True is needed exactly when the sink has doc_ids (the {name} "
                f"{'has' if stores else 'has no'} doc_ids, the call has doc_ids={doc_ids}); "
                "pass (texts, doc_ids) batches to a sink with doc_ids"
            )


def _check_apart(vector_sink: Any, score_sink: Any) -> None:
    """Two stores need two folders that do not contain each other."""
    if not (isinstance(vector_sink, ShardSinkBase) and isinstance(score_sink, ShardSinkBase)):
        return
    one, other = Path(vector_sink.path).resolve(), Path(score_sink.path).resolve()
    if one == other or one.is_relative_to(other) or other.is_relative_to(one):
        raise ValueError("vector_sink and score_sink must write to two folders apart")


def _check_rows(vector_sink: Any, score_sink: Any, doc_ids: bool) -> None:
    """Two stores that already hold rows of different numbers take a run only with
    document IDs: the position is the only link between them, and an append would shift it."""
    if not (isinstance(vector_sink, ShardSinkBase) and isinstance(score_sink, ShardSinkBase)):
        return
    if doc_ids or vector_sink.first_row == score_sink.first_row:
        return
    raise ValueError(
        f"the vector_sink holds {vector_sink.first_row} rows and the score_sink "
        f"{score_sink.first_row}; without doc_ids the position links the two stores and an "
        "append would shift it. Pass doc_ids=True with IDs of your own that name each row "
        "uniquely, or write into two new folders"
    )


class SinkRun:
    """The batches of a ``predict_batches`` call into a vector sink, a score sink or both."""

    def __init__(self, kinds: tuple[str, ...], vector_sink: Any, score_sink: Any, doc_ids: bool):
        self.sinks = [s for s in (vector_sink, score_sink) if s is not None]
        self.ids = TaskIds(len(self.sinks))
        self.score = None if score_sink is None else ScoreRun(score_sink, kinds, self.ids)
        self.vectors = vector_sink
        self.arguments: dict[str, Any] = {"tee_bytes": 8 if doc_ids else 0}
        if self.score is not None:
            self.arguments.update(
                tees=self.score.tees, extras=self.score.extras, tee_bytes=self.score.tee_bytes
            )
        if vector_sink is not None:
            self.arguments.update(
                vector_sink=_VectorIds(vector_sink, self.ids) if doc_ids else vector_sink,
                vector_dtype=sink_dtype(vector_sink),
                inverse=wants_inverse_norms(vector_sink),
            )
        self._last: list[int | None] = [None] * len(self.sinks)

    def infer(
        self, pipeline: Any, source: Any, kinds: tuple[str, ...], workers: Any, row: int, ids: Any
    ):
        """The batch ``source`` starting at ``row`` of the run (``(arrays, vectors)`` of the
        pipeline); ``ids`` are its document IDs, given exactly if the sinks store them."""
        if self.score is not None:
            self.score.reset()
        vectors = self.arguments.get("vector_sink")
        if isinstance(vectors, _VectorIds):
            vectors.pending.clear()
        arguments = {"outputs": kinds, "workers": workers, "first_row": row, **self.arguments}
        with self.ids.batch(source, row, ids):
            return pipeline.infer(source, **arguments)

    def durable(self, done: int) -> int | None:
        """The rows durable in every sink after a batch that ended at ``done`` rows of the
        run: the minimum of the counts, each checked on its own; None if one sink
        promises nothing. Every sink publishes first."""
        counts = [
            durable(sink, done, last) for sink, last in zip(self.sinks, self._last, strict=True)
        ]
        self._last = counts
        known = [c for c in counts if c is not None]
        return min(known) if len(known) == len(counts) else None

    def flush(self) -> None:
        """Flush the vector sink, then the score sink, after the last batch."""
        for sink in self.sinks:
            flush(sink)


class _VectorIds:
    """The vector sink with the document IDs of each task: it gets the block of the task at
    its reservation and hands it to the sink at the commit."""

    def __init__(self, sink: Any, ids: TaskIds):
        self.sink, self.ids = sink, ids
        self.pending: dict[int, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sink, name)

    def target(self, first_row: int, n_rows: int) -> Any:
        block = self.ids.block(first_row, n_rows)
        target = self.sink.target(first_row, n_rows)
        self.pending[first_row] = block
        return target

    def commit(self, first_row: int, rows: Any) -> None:
        block = self.pending.pop(first_row)
        self.ids.release(first_row)
        self.sink.commit(first_row, rows, doc_ids=block)
