#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The tasks of a source: one chunk each, and whether a source is small enough to run inline.

A task turns its sentences (the input interval) into the rows ``first_row`` to
``first_row + n_rows - 1`` (the output interval); with one row per sentence both have
the same length. Token-id inputs (2.1) add task types with their own ``batch``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fse._contract import Batch
from fse._stream import StreamSource
from fse._text import (
    ArrowSource,
    Chunk,
    StringSource,
    TokenListSource,
    check_chunk_bytes,
    default_chunk_bytes,
)
from fse._text_file import FileSource

# A source of at most this many bytes runs in the calling thread: threads would cost more
# than they save (bench_runtime.py latency).
INLINE_MAX_BYTES = 1 << 16
# The chunks of a collection of a stream: a collection is not the whole input and the
# window spans collections, so a stream keeps its smaller tasks.
STREAM_TASKS_PER_COLLECTION = 8

SOURCES = (StringSource, FileSource, TokenListSource, StreamSource, ArrowSource)


@dataclass(frozen=True)
class Task:
    source: Any  # has encode(chunk, lookup, workspace) -> Batch
    chunk: Chunk

    @property
    def first_row(self) -> int:
        return self.chunk.first_row

    @property
    def n_rows(self) -> int:
        return self.chunk.n_sentences

    def batch(self, lookup: Any, workspace: Any) -> Batch:
        return self.source.encode(self.chunk, lookup, workspace)

    def prefix(self, rows: int) -> Task:
        """The task of its first ``rows`` sentences (1 to all of them)."""
        return (
            self if rows == self.n_rows else Task(self.source, self.source.prefix(self.chunk, rows))
        )


@dataclass(frozen=True)
class Plan:
    tasks: Iterator[Task]
    rows: int | None  # None for a stream
    inline: bool


def plan(
    source: object,
    *,
    workers: int,
    chunk_bytes: int | None,
    chunk_rows: int,
    tokenizer: str,
    stop: threading.Event | None = None,
    first_row: int = 0,
) -> Plan:
    """Scan ``source`` and plan one task per chunk of at most ``chunk_rows`` rows; an error
    of the scan names its row from ``first_row`` (the row of the run where it starts)."""
    if not isinstance(source, SOURCES):
        names = ", ".join(cls.__name__ for cls in SOURCES)
        raise TypeError(f"source must be one of {names}, not {type(source).__name__}")
    if isinstance(source, TokenListSource) and tokenizer != "external":
        raise ValueError("token lists are for the external tokenizer")
    chunk_bytes = check_chunk_bytes(chunk_bytes)  # before a stream is read
    if isinstance(source, StreamSource):
        return _stream_plan(source, chunk_bytes, chunk_rows, stop, first_row)
    inline = source.nbytes <= INLINE_MAX_BYTES  # scanned as one range, run in the caller
    scan, error = source.scan_until_error(
        workers=1 if inline else workers,
        chunk_bytes=chunk_bytes,
        chunk_rows=chunk_rows,
        first_row=first_row,
    )
    chunks = scan.chunks
    if chunk_bytes is None and not inline and workers > 1:  # the automatic plan only
        chunks = source.split_tail(chunks, workers)
    return Plan(_then(chunks, source, error), scan.sentences, inline)


def _then(chunks: tuple[Chunk, ...], source: Any, error: Exception | None) -> Iterator[Task]:
    """The tasks of ``chunks``, then the error of the scan (an invalid sentence after
    them), which the executor raises after the tasks before it, in row order."""
    for chunk in chunks:
        yield Task(source, chunk)
    if error is not None:
        raise error


def _stream_plan(
    source: StreamSource,
    chunk_bytes: int | None,
    chunk_rows: int,
    stop: threading.Event | None,
    first_row: int,
) -> Plan:
    """The first collection is read now, to decide whether the stream runs inline. Once
    ``stop`` is set (a task failed), the stream is not read any further."""
    collections = source.collections(stop, first_row)
    first = next(collections, None)
    if first is None:
        return Plan(iter(()), None, True)
    inline = first.last and first.source.nbytes <= INLINE_MAX_BYTES

    def tasks() -> Iterator[Task]:
        for collection in _chain(first, collections):
            size = chunk_bytes or default_chunk_bytes(
                collection.source.nbytes, 1, STREAM_TASKS_PER_COLLECTION
            )
            scan, error = collection.source.scan_until_error(
                chunk_bytes=size, chunk_rows=chunk_rows, first_row=first_row + collection.first_row
            )
            for c in scan.chunks:
                chunk = Chunk(c.start, c.end, collection.first_row + c.first_row, c.n_sentences)
                yield Task(collection.source, chunk)
            if error is not None:
                raise error

    return Plan(tasks(), None, inline)


def _chain(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from rest
