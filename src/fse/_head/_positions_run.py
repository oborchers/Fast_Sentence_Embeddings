#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""A stream of heads that resumes: ``predict_batches(batches, vector_sink=, score_sink=,
positions=True)`` of ``fse.Heads``, ``Classifier`` and ``Regressor`` with sinks made with
``resume=True`` (docs/architecture/scores.md, "Resuming a stream").

The batches are ``(texts, position)``, or ``(texts, doc_ids, position)`` with
``doc_ids=True``. The sinks keep the position of the batch each shard ends in
(``fse._encoder._positions``). Two stores written together may hold different numbers of
rows, because a run stopped between their shards; the stream is replayed from the anchor of the
shorter one. A batch is then cut at the records each store holds: those both hold are dropped,
those only the longer holds go to the shorter store alone, the rest to both. Each cut is a run of
the heads of its own, so every record is computed as the uninterrupted run computes it, bit for
bit, and the outputs of a batch are the outputs of its cuts in order. A batch the stores hold
completely gives empty outputs, so every batch of the stream gives one result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fse._encoder._batches import _named, batch_source, check_sink
from fse._encoder._doc_ids import check_modes
from fse._encoder._positions import Replay, check_ids, rows_of, split, window
from fse._encoder._resume import ResumableSinkMarker
from fse._encoder._sources import opened
from fse._runtime import LengthSettings

from ._score_target import check_score_sink
from ._sink_run import Shifted, SinkRun


class _Member:
    """A sink of the run: a ``vector`` or ``score`` sink, with the head paths a score sink
    attaches by."""

    def __init__(self, role: str, sink: Any, paths: Mapping[str, str]):
        self.role, self.sink, self.paths = role, sink, paths

    @property
    def new(self) -> bool:
        """Whether the store does not exist yet: it is made with the first batch it gets."""
        return bool(self.sink._new) if hasattr(self.sink, "_new") else not self.sink._exists

    def attach(self, position: str | None) -> None:
        if self.role == "score":
            self.sink._attach_stream(position, self.paths)
        else:
            self.sink._attach_stream(position)


def _check_sink(role: str, sink: Any) -> None:
    if not (isinstance(sink, ResumableSinkMarker) and sink.resume is True):
        raise ValueError(
            f"positions=True needs the {role}_sink to be made with resume=True, such as "
            "fse.ScoreSink(path, heads, resume=True)"
        )


def _check_apart(vector_sink: Any, score_sink: Any) -> None:
    """Two stores need two folders that do not contain each other."""
    if vector_sink is None or score_sink is None:
        return
    one, other = Path(vector_sink.path).resolve(), Path(score_sink.path).resolve()
    if one == other or one.is_relative_to(other) or other.is_relative_to(one):
        raise ValueError("vector_sink and score_sink must write to two folders apart")


def check_history(first: Any, second: Any) -> None:
    """Two stores resumed together must share their history: the same shard size, the same
    anchor at the start and the same anchor at every shard boundary both hold."""
    one, other = first._stream, second._stream
    same = first.shard_rows == second.shard_rows and one.root == other.root
    held = dict(other.chain)
    same = same and all(held.get(end, anchor) == anchor for end, anchor in one.chain)
    if not same:
        raise ValueError(
            "the two stores do not share their history (the size of their shards, the position "
            "of their start or the positions at their shard boundaries differ), so one stream "
            "cannot resume both; write two new folders"
        )


class Stream:
    """The sinks of a run with positions, and the replay rows they hold."""

    def __init__(
        self, members: list[_Member], kinds: tuple[str, ...], doc_ids: bool, tokenizer: str
    ):
        self.members, self.kinds, self.doc_ids, self.tokenizer = members, kinds, doc_ids, tokenizer
        rows = [0 if m.new else m.sink.first_row for m in members]
        shorter = members[rows.index(min(rows))]
        self.start = 0 if shorter.new else shorter.sink._stream.start[0]  # the row of the seek
        self.replay = Replay([n - self.start for n in rows])
        self._runs: dict[tuple[int, ...], SinkRun] = {}

    def begin(self, index: int, position: str, before: int) -> None:
        """The sink ``index`` takes records of a batch that starts at replay row ``before``: a
        new store is made, with the position of the first batch (or the start of the other
        store) as the anchor of its root; then the batch is the anchor for the shards."""
        member = self.members[index]
        if member.new and not member.sink._ready:
            others = [m for m in self.members if m is not member and not m.new]
            root = others[0].sink._stream.root.position if others else position
            if root != position:
                raise ValueError(
                    "the stream must start at the position the other store started from: the "
                    "store of this sink is new and takes the start of the other"
                )
            member.attach(root)
        member.sink._stream.anchors.begin(position, self.start + before)

    def run(self, who: tuple[int, ...]) -> SinkRun:
        """The runs of the heads into the sinks ``who`` (their rows shifted to the rows the
        run counts)."""
        found = self._runs.get(who)
        if found is None:
            sinks: dict[str, Any] = {}
            for index in who:
                member = self.members[index]
                shift = self.replay.shift(index)
                sinks[member.role] = Shifted(member.sink, shift) if shift else member.sink
            found = self._runs[who] = SinkRun(
                self.kinds, sinks.get("vector"), sinks.get("score"), self.doc_ids
            )
        return found

    def durable(self) -> int:
        """The replay rows durable in every sink, once each one published."""
        counts = [m.sink.publish() for m in self.members]
        return self.replay.durable(counts, all(m.sink._synced for m in self.members))

    def flush(self) -> None:
        """Flush the vector sink, then the score sink, after the last batch."""
        for member in self.members:
            member.sink.flush()


def _check_sinks(
    heads: Mapping[str, Any],
    return_vectors: bool,
    present: list[tuple[str, Any]],
    doc_ids: bool,
    method: str,
) -> None:
    """The checks of the sinks of ``positions=True`` that read nothing of the stream."""
    sinks = dict(present)
    for role, sink in present:
        _check_sink(role, sink)
    first = next(iter(heads.values()))
    if "vector" in sinks:
        if not return_vectors:
            raise ValueError(
                "vector_sink takes the vectors of return_vectors=True; pass return_vectors=True "
                "(a head on raw means then takes the unfolded path, see path() and Heads.paths)"
            )
        dim = first.encoder.output_dim
        hint = f"; the vector_sink takes the vectors of the encoder ({dim} values)"
        check_sink(sinks["vector"], None, first.encoder_fingerprint, dim, hint, resumed=True)
    if "score" in sinks:
        check_score_sink(sinks["score"], heads, resumed=True)
    check_modes(present, doc_ids, method)
    _check_apart(sinks.get("vector"), sinks.get("score"))


def start(
    heads: Mapping[str, Any],
    kinds: tuple[str, ...],
    return_vectors: bool,
    vector_sink: Any,
    score_sink: Any,
    doc_ids: bool,
    method: str,
) -> Stream:
    """The checks of ``predict_batches(positions=True)`` that read nothing of the stream, and
    the attach to the stores that exist (verified and made durable before the first batch is
    read; a new store is made with the first batch that brings it a record). ``heads`` are the
    models by name."""
    given = [("vector", vector_sink), ("score", score_sink)]
    present = [(role, sink) for role, sink in given if sink is not None]
    if not present:
        raise ValueError(
            "positions=True needs a vector_sink or a score_sink made with resume=True, such as "
            "fse.ScoreSink(path, heads, resume=True)"
        )
    _check_sinks(heads, return_vectors, present, doc_ids, method)
    paths = {n: m.path(return_vectors=return_vectors) for n, m in heads.items()}
    members = [_Member(role, sink, paths) for role, sink in present]
    for member in members:
        member.sink._check_stream()
    for member in sorted(members, key=lambda m: m.new):  # stores that exist go first
        if not member.new:
            member.attach(None)
    existing = [m for m in members if not m.new]
    if len(existing) == 2:
        check_history(existing[0].sink, existing[1].sink)
    return Stream(members, kinds, doc_ids, next(iter(heads.values()))._tokenizer)


def _merge(parts: list[tuple[list, Any]]) -> tuple[list, Any]:
    """The arrays of the heads and the vectors of several cuts of a batch, in order."""
    if len(parts) == 1:
        return parts[0]
    rows = [np.concatenate([part[0][i] for part in parts]) for i in range(len(parts[0][0]))]
    vectors = [part[1] for part in parts]
    return rows, None if any(v is None for v in vectors) else np.concatenate(vectors)


def stream_batches(
    pipeline: Any,
    items: Iterable[Any],
    stream: Stream,
    workers: Any,
    method: str,
    make: Callable[[int, int, list, Any, int | None, Any], Any],
    lengths: LengthSettings | None = None,
    rule: tuple[int | None, int | None] = (None, None),
) -> Iterator[Any]:
    """The batches of ``items`` through ``pipeline`` (a ``HeadsPipeline``) into the sinks of
    ``stream``. ``make(first_row, n_rows, rows, vectors, durable_rows, report)`` builds each
    result from the arrays of the heads, the vectors and the ``LengthReport`` of the records
    computed (``first_row`` is their first replay row)."""
    last = 0
    for k, item in enumerate(items):
        before = stream.replay.done
        with _named(k, before, None):
            first, rows, found, vectors, report = _batch(
                pipeline, stream, workers, split(k, item, stream.doc_ids, method), k, lengths, rule
            )
            last = max(last, stream.durable())
        yield make(before + first, rows - first, found, vectors, last, report)
    stream.replay.check_end()
    stream.flush()


def _batch(
    pipeline: Any,
    stream: Stream,
    workers: Any,
    item: tuple[Any, Any, str],
    k: int,
    lengths: LengthSettings | None,
    rule: tuple[int | None, int | None],
) -> tuple[int, int, list, Any, Any]:
    """Run the records of batch ``k`` (``item``: its texts, IDs and position) that the stores
    do not hold; gives the records cut off, the records of the batch, their arrays, the
    vectors and the ``LengthReport``."""
    texts, ids, position = item
    before = stream.replay.done
    rows = rows_of(k, texts, stream.tokenizer)
    if ids is not None:
        check_ids(k, ids, rows)
    pieces = stream.replay.pieces(rows)
    first = pieces[0].lo if pieces else rows
    collector = None if lengths is None else lengths.collector(before + first, rule)
    for index in sorted({i for piece in pieces for i in piece.who}):
        stream.begin(index, position, before)
    parts = [
        _piece(pipeline, stream, workers, rows, (texts, ids), piece, lengths, collector)
        for piece in pieces
    ] or [_empty(pipeline, stream, workers, lengths, collector)]
    found, vectors = _merge(parts)
    return first, rows, found, vectors, None if collector is None else collector.report()


def _piece(
    pipeline: Any,
    stream: Stream,
    workers: Any,
    rows: int,
    batch: tuple[Any, Any],
    piece: Any,
    lengths: LengthSettings | None,
    collector: Any,
) -> tuple[list, Any]:
    """The cut ``piece`` of a batch through the heads into its sinks."""
    texts, ids = window(*batch, piece.lo, piece.hi, rows)
    with batch_source(texts, stream.tokenizer) as (source, _):
        return stream.run(piece.who).infer(
            pipeline, source, stream.kinds, workers, piece.run_row, ids, lengths, collector
        )


def _empty(
    pipeline: Any, stream: Stream, workers: Any, lengths: LengthSettings | None, collector: Any
) -> tuple[list, Any]:
    """The empty outputs, of the right shape, of a batch the stores hold completely."""
    with opened([], stream.tokenizer) as source:
        return pipeline.infer(
            source,
            outputs=stream.kinds,
            workers=workers,
            first_row=0,
            lengths=lengths,
            collector=collector,
        )
