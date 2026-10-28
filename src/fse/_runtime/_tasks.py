#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The tasks of a source: one chunk each, and whether a source is small enough to run inline.

A task turns its sentences (the input interval) into the rows ``first_row`` to
``first_row + n_rows - 1`` (the output interval); with one row per sentence both have
the same length. Group mode (PR 8) and token-id inputs (2.1) add task types with their
own ``batch``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fse._contract import Batch
from fse._text import Chunk, FileSource, StringSource, TokenListSource

# A source of at most this many bytes and one chunk of rows runs in the calling thread:
# threads would cost more than they save.
INLINE_MAX_BYTES = 1 << 16

SOURCES = (StringSource, FileSource, TokenListSource)


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


@dataclass(frozen=True)
class Plan:
    tasks: Iterator[Task]
    rows: int | None  # None for a stream
    inline: bool


def plan(
    source: object, *, workers: int, chunk_bytes: int | None, chunk_rows: int, tokenizer: str
) -> Plan:
    """Scan ``source`` and plan one task per chunk of at most ``chunk_rows`` rows."""
    if not isinstance(source, SOURCES):
        names = ", ".join(cls.__name__ for cls in SOURCES)
        raise TypeError(f"source must be one of {names}, not {type(source).__name__}")
    if isinstance(source, TokenListSource) and tokenizer != "external":
        raise ValueError("token lists are for the external tokenizer")
    scan = source.scan(workers=workers, chunk_bytes=chunk_bytes, chunk_rows=chunk_rows)
    inline = source.nbytes <= INLINE_MAX_BYTES and scan.sentences <= chunk_rows
    return Plan(iter([Task(source, chunk) for chunk in scan.chunks]), scan.sentences, inline)
