#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The sinks of a run of heads, a group or a single head: a ``vector_sink`` for the vectors of
the encoder and a ``score_sink`` for the scores of every head, each alone or both in one
pass (docs/architecture/scores.md, "Vectors and scores in one run"). A single head is a group
of one: ``start_lone`` names it as the sink does.

This module checks the sinks before a text is read, shares the document IDs of each task
between them, and reports the rows that are durable in all of them: the minimum of the
counts of the sinks, each checked on its own. It adds no protocol between the sinks: each
folder is a valid prefix on its own, and what exceeds the confirmed rows may be in one folder
only. The runtime (``fse._runtime``) is not changed: it takes a vector sink and tees already.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from pathlib import Path
from typing import Any

from fse._encoder._batches import _named, batch_size, batch_source, check_sink, durable, flush
from fse._encoder._doc_ids import check_modes, split_pair
from fse._runtime import LengthSettings, streaming
from fse._runtime._sinks import sink_dtype, wants_inverse_norms
from fse._store._base_sink import ShardSinkBase

from ._score_target import ScoreRun, TaskIds, check_score_sink


def start(
    heads: Mapping[str, Any],
    kinds: tuple[str, ...],
    return_vectors: bool,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: bool,
    method: str,
) -> SinkRun | None:
    """Check the sinks against ``heads`` (the models by name) and the document-ID mode, attach
    the run to the score sink and return the run (None without a sink). Raises before a text
    is read."""
    if vector_sink is None and score_sink is None:
        check_modes([], doc_ids, method, "a vector_sink or a score_sink")
        return None
    if vector_sink is not None:
        _check_vector_sink(next(iter(heads.values())), return_vectors, vector_sink)
    if score_sink is not None:
        check_score_sink(score_sink, heads)
    given = (("vector_sink", vector_sink), ("score_sink", score_sink))
    check_modes([(role, s) for role, s in given if s is not None], doc_ids, method)
    _check_apart(vector_sink, score_sink)
    if score_sink is not None:
        if vector_sink is not None and getattr(score_sink, "_exists", False):
            _check_rows(vector_sink, score_sink, doc_ids)
        score_sink._attach({n: m.path(return_vectors=return_vectors) for n, m in heads.items()})
        if vector_sink is not None:  # _attach may have set first_row anew
            _check_rows(vector_sink, score_sink, doc_ids)
    return SinkRun(kinds, vector_sink, score_sink, doc_ids)


def start_lone(
    model: Any,
    kinds: tuple[str, ...],
    return_vectors: bool,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: bool,
    method: str,
    starter: Callable[..., Any] = start,
) -> Any:
    """``start`` (or ``starter``, which takes the same arguments) for a single head. The score
    sink names it: a sink of one head."""
    schema = getattr(score_sink, "score_schema", None)
    if vector_sink is not None and starter is start:  # the checks of a group come in this order
        _check_vector_sink(model, return_vectors, vector_sink)
    name = "head"
    if schema is not None:
        names = [head.name for head in schema.heads]
        if len(names) != 1:
            raise ValueError(
                f"the sink was made for the heads {names}; a single head writes a sink of one head"
            )
        name = names[0]
    return starter({name: model}, kinds, return_vectors, vector_sink, score_sink, doc_ids, method)


def _check_vector_sink(first: Any, return_vectors: bool, sink: Any) -> None:
    if not return_vectors:
        raise ValueError(
            "vector_sink takes the vectors of return_vectors=True; pass return_vectors=True "
            "(a head on raw means then takes the unfolded path, see path() and Heads.paths)"
        )
    dim = first.encoder.output_dim
    hint = f"; the vector_sink takes the vectors of the encoder ({dim} values)"
    check_sink(sink, None, first.encoder_fingerprint, dim, hint)
    if isinstance(sink, ShardSinkBase):
        sink._check_fresh()


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


@streaming
def run_batches(
    tokenizer: str,
    pipeline: Any,
    items: Iterable[Any],
    kinds: tuple[str, ...],
    workers: Any,
    run: SinkRun | None,
    doc_ids: bool,
    method: str,
    make: Callable[[int, int, list, Any, int | None, Any], Any],
    lengths: LengthSettings | None = None,
    rule: tuple[int | None, int | None] = (None, None),
) -> Generator[Any, None, None]:
    """The batches of ``items`` through ``pipeline`` (a ``HeadsPipeline``), into the sinks of
    ``run`` if there is one. ``make(first_row, n_rows, rows, vectors, durable_rows, report)``
    builds each batch from the arrays of the heads and the ``LengthReport`` of its rows
    (None unless ``lengths`` asks for one under ``rule``, that of the encoder). Shared by a
    group and a single head."""
    row, last = 0, None
    for k, item in enumerate(items):
        ids = None
        if doc_ids:
            item, ids = split_pair(k, item, method)
        with (
            _named(k, row, batch_size(k, item)),
            batch_source(item, tokenizer) as (source, n),
        ):
            collector = None if lengths is None else lengths.collector(row, rule)
            if run is None:
                rows, vectors = pipeline.infer(
                    source,
                    outputs=kinds,
                    workers=workers,
                    first_row=row,
                    lengths=lengths,
                    collector=collector,
                )
            else:
                rows, vectors = run.infer(
                    pipeline, source, kinds, workers, row, ids, lengths, collector
                )
                last = run.durable(row + n)
            report = None if collector is None else collector.report()
            batch = make(row, n, rows, vectors, last, report)
        yield batch
        row += n
    if run is not None:
        run.flush()


class SinkRun:
    """The batches of a ``predict_batches`` call into a vector sink, a score sink or both."""

    def __init__(
        self,
        kinds: tuple[str, ...],
        vector_sink: Any,
        score_sink: Any,
        doc_ids: bool,
        discard: list[Any] | None = None,
    ):
        """``discard`` is a sink for the main output of every head, for a run that keeps
        only the stored scores and vectors (a resumed run)."""
        self.sinks = [s for s in (vector_sink, score_sink) if s is not None]
        self.ids = TaskIds(len(self.sinks))
        self.score = None if score_sink is None else ScoreRun(score_sink, kinds, self.ids)
        self.vectors = vector_sink
        self.arguments: dict[str, Any] = {"tee_bytes": 8 if doc_ids else 0}
        if discard is not None:
            self.arguments["sinks"] = discard
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
        self,
        pipeline: Any,
        source: Any,
        kinds: tuple[str, ...],
        workers: Any,
        row: int,
        ids: Any,
        lengths: Any = None,
        collector: Any = None,
    ):
        """The batch ``source`` starting at ``row`` of the run (``(arrays, vectors)`` of the
        pipeline); ``ids`` are its document IDs, given exactly if the sinks store them.
        ``lengths`` and ``collector`` are those of ``HeadsPipeline.infer``."""
        if self.score is not None:
            self.score.reset()
        vectors = self.arguments.get("vector_sink")
        if isinstance(vectors, _VectorIds):
            vectors.pending.clear()
        arguments = {
            "outputs": kinds,
            "workers": workers,
            "first_row": row,
            "lengths": lengths,
            "collector": collector,
            **self.arguments,
        }
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


class Shifted:
    """``sink`` for a run whose rows are numbered ``delta`` rows later than the sink counts
    them: row ``r`` of the run is row ``r + delta`` of the sink. Everything else is the
    sink's own."""

    def __init__(self, sink: Any, delta: int):
        self.sink, self.delta = sink, delta

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sink, name)

    def target(self, first_row: int, n_rows: int) -> Any:
        return self.sink.target(first_row + self.delta, n_rows)

    def commit(self, first_row: int, rows: Any, **keywords: Any) -> None:
        self.sink.commit(first_row + self.delta, rows, **keywords)
