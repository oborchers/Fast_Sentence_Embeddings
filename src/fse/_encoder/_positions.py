#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Resuming a stream: ``Encoder.encode_batches(batches, sink=, positions=True)`` with a sink
made with ``resume=True`` (docs/architecture/store.md, "Stream positions").

The batches are ``(texts, position)``, or ``(texts, doc_ids, position)`` with ``doc_ids=True``;
``position`` is an opaque text of the caller that names where the source delivers the first
record of the batch from. The sink stores, with every shard, the position of the batch the
shard ends in. After a crash the caller reads ``sink.source_position``, ``(P, skip)``, seeks
the source to ``P`` and gives its batches again; the first ``skip`` records, which the store
holds already, are dropped here without being tokenized or written, and every later record is
written as the run that never stopped would have written it.

A stream has no end and no identity to hash, so nothing but the anchors is stored. The records
the source delivers after the seek are numbered from 0, the *replay rows*; a batch is a window
onto them. Two stores written together (the vectors and the scores of one run of heads) may
hold different numbers of rows: ``Replay`` cuts a batch at the rows each of them holds.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Sequence
from typing import Any, NamedTuple

import numpy as np

from fse._runtime import LengthSettings, Pipeline, streaming
from fse._text import ArrowSource, is_arrow
from fse._text._ids import IdSource

from ._batches import EncodedBatch, _named, batch_size, batch_source, flush, run_into
from ._resume import is_resume_sink


class Piece(NamedTuple):
    """The rows ``[lo, hi)`` of a batch go to the sinks ``who`` (indices), numbered from
    ``run_row`` in the run (the rows into the sink that holds the fewest rows)."""

    lo: int
    hi: int
    who: tuple[int, ...]
    run_row: int


class Replay:
    """The arithmetic of the replay rows of a run into ``skips`` (the records each sink drops
    after the position of the seek: the rows it holds less the row of the anchor the seek
    goes to). The records below the least skip are held by every sink and dropped; those
    between the least and the greatest skip go to the sinks that hold fewer rows; the rest go
    to all."""

    def __init__(self, skips: Sequence[int]):
        self.skips = tuple(skips)
        self.low, self.high = min(self.skips), max(self.skips)
        self.done = 0  # the replay rows delivered

    def pieces(self, rows: int) -> list[Piece]:
        """The pieces of the next batch of ``rows`` records, which then counts as delivered."""
        start = self.done
        found = []
        ahead = tuple(k for k, skip in enumerate(self.skips) if skip == self.low)
        for lo, hi, who in (
            (self.low, self.high, ahead),
            (self.high, start + rows, tuple(range(len(self.skips)))),
        ):
            first, last = max(start, lo), min(start + rows, hi)
            if first < last:
                found.append(Piece(first - start, last - start, who, first - self.low))
        self.done = start + rows
        return found

    def shift(self, index: int) -> int:
        """How far the run rows of sink ``index`` lie before the rows the sink counts: the run
        counts from the sink that holds the fewest rows."""
        return self.low - self.skips[index]

    def durable(self, counts: Sequence[int], synced: bool) -> int:
        """The replay rows that are durable in every sink: those delivered, but at most those
        every sink holds durable (the rows it held when it attached, which were synced then,
        and the rows it published since, ``counts``); 0 where a sink is not synced."""
        if not synced:
            return 0
        held = min(skip + count for skip, count in zip(self.skips, counts, strict=True))
        return min(self.done, held)

    def check_end(self) -> None:
        """The stream ended: it must have delivered the records the stores hold after the
        position of the seek."""
        if self.done < self.high:
            raise ValueError(
                f"the stream delivered {self.done} records from the stored position, fewer than "
                f"the {self.high} records the stores hold after it; the position of the seek or "
                "the source is not the one the store was written from"
            )


def split(k: int, item: Any, doc_ids: bool, method: str) -> tuple[Any, Any, str]:
    """The ``(texts, doc_ids, position)`` of batch ``k``, a ``(texts, position)`` tuple or with
    ``doc_ids=True`` a ``(texts, doc_ids, position)`` tuple."""
    size = 3 if doc_ids else 2
    shape = "(texts, doc_ids, position)" if doc_ids else "(texts, position)"
    if not isinstance(item, tuple) or len(item) != size:
        raise TypeError(f"{method}: batch {k} must be a {shape} tuple")
    if doc_ids and item[1] is None:
        raise TypeError(f"{method}: batch {k} must give its doc_ids, not None")
    return (item[0], item[1], item[2]) if doc_ids else (item[0], None, item[1])


def rows_of(k: int, texts: Any, tokenizer: str) -> int:
    """The records of batch ``k``, which the batch counts (an Arrow column is opened for it)."""
    rows = batch_size(k, texts)
    if rows is not None:
        return rows
    with batch_source(texts, tokenizer) as (_, n):
        return n


def window(texts: Any, ids: Any, first: int, last: int, rows: int) -> tuple[Any, Any]:
    """The records ``first`` to ``last - 1`` of a batch of ``rows`` records and their document
    IDs; the batch itself where it is whole. A list or tuple is cut by slicing; an Arrow column
    is read into a list of its texts (once per resume: only the batch that holds the end of the
    stored records is cut); IDs come as an int64 array."""
    if (first, last) == (0, rows):
        return texts, ids
    texts = _arrow_texts(texts, first, last) if is_arrow(texts) else texts[first:last]
    if ids is not None:
        with IdSource(ids) as values:
            ids = np.asarray(values.block(first, last))
    return texts, ids


def _arrow_texts(column: Any, first: int, last: int) -> list[str]:
    with ArrowSource(column) as source:
        if 0 <= source._column.null_row < last:
            raise ValueError(f"the Arrow column holds a null at row {source._column.null_row}")
        return [source._column.row_bytes(row).decode("utf-8") for row in range(first, last)]


def check_ids(k: int, ids: Any, rows: int) -> None:
    """The document IDs of batch ``k`` must be one per record, also where the records the
    store holds are dropped."""
    with IdSource(ids) as values:
        if len(values) != rows:
            raise ValueError(f"batch {k}: doc_ids has {len(values)} rows for {rows} texts")


def start(sink: Any) -> None:
    """The checks of ``encode_batches(positions=True)`` that read nothing of the stream, and
    the attach to a store that exists (it is verified and made durable before the first batch
    is read; a new store is made with the first non-empty batch)."""
    if not is_resume_sink(sink):
        raise ValueError(
            "positions=True needs a sink made with resume=True, such as "
            "fse.ShardSink(path, encoder, resume=True)"
        )
    sink._check_stream()
    if not sink._new:
        sink._attach_stream(None)


class _Run:
    """The state of a run: the sink, the replay rows and the rows durable so far."""

    def __init__(
        self,
        pipeline: Pipeline,
        tokenizer: str,
        sink: Any,
        workers: Any,
        doc_ids: bool,
        lengths: LengthSettings | None,
        rule: tuple[int | None, int | None],
    ):
        self.pipeline, self.tokenizer, self.sink, self.workers = pipeline, tokenizer, sink, workers
        self.doc_ids, self.lengths, self.rule = doc_ids, lengths, rule
        stream = sink._stream  # None for a new store, which holds no record
        self.replay = Replay([0 if stream is None else stream.skip])
        self.last = 0

    def batch(self, k: int, item: Any) -> Any:
        """The result of batch ``k``: its records not yet in the store go through the pipeline
        into the sink."""
        texts, ids, position = split(k, item, self.doc_ids, "Encoder.encode_batches")
        sink = self.sink
        before = self.replay.done
        with _named(k, before, None):
            rows = rows_of(k, texts, self.tokenizer)
            if ids is not None:
                check_ids(k, ids, rows)
            found = self.replay.pieces(rows)
            first = found[0].lo if found else rows
            lengths = self.lengths
            collector = None if lengths is None else lengths.collector(before + first, self.rule)
            if found:
                self._compute((texts, ids, position, before), found[0], rows, collector)
            self.last = max(self.last, self.replay.durable([sink.publish()], sink._synced))
        result = EncodedBatch(before + first, rows - first, None, self.last)
        return (result, collector.report()) if collector is not None else result

    def _compute(
        self, batch: tuple[Any, Any, str, int], piece: Piece, rows: int, collector: Any
    ) -> None:
        """Compute the records of ``piece`` of ``batch`` (its texts, IDs, position and the
        replay row it starts at) into the sink; a new store is made first, with the position
        of the batch as the anchor of its root."""
        texts, ids, position, before = batch
        sink = self.sink
        if not sink._ready:
            sink._attach_stream(position)
        sink._stream.anchors.begin(position, sink._stream.start[0] + before)
        part, part_ids = window(texts, ids, piece.lo, piece.hi, rows)
        with batch_source(part, self.tokenizer) as (source, _):
            run_into(
                self.pipeline,
                source,
                sink,
                self.workers,
                piece.run_row,
                part_ids,
                self.lengths,
                collector,
            )

    def finish(self) -> None:
        """The stream ended: it must have delivered what the store holds; then flush."""
        self.replay.check_end()
        flush(self.sink)


@streaming
def batches(
    pipeline: Pipeline,
    items: Iterable[Any],
    tokenizer: str,
    *,
    sink: Any,
    workers: Any,
    doc_ids: bool = False,
    lengths: LengthSettings | None = None,
    rule: tuple[int | None, int | None] = (None, None),
) -> Generator[Any, None, None]:
    """The EncodedBatch of every item of ``items``, ``(texts, position)`` or
    ``(texts, doc_ids, position)``; with ``lengths`` of a level, the pair of it and the
    ``LengthReport`` of the records computed. ``first_row`` is the replay row of the first
    record computed, ``n_rows`` the records computed (0 for a batch the store holds) and
    ``durable_rows`` the replay rows that are durable in the sink."""
    run = _Run(pipeline, tokenizer, sink, workers, doc_ids, lengths, rule)
    for k, item in enumerate(items):
        yield run.batch(k, item)
    run.finish()
