#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Resuming a run into a store: ``Encoder.encode(texts, sink=, shard=)`` with a sink made with
``resume=True`` (docs/architecture/store.md, "Resume stores").

A source that can be read again (a list, tuple or array of str, token lists or a text file)
has an identity: a fingerprint of its bytes, its number of rows and its form. The store
records it with the partition it holds. A resumed run hashes the source, lets the sink
compare the identity with the store (a changed source is refused before any row is
reserved), and runs the rows after those the store holds, up to the end of its partition,
through a ``Window``. The rows are the rows of an uninterrupted run, bit for bit.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from fse._container import Hasher, checksum, fingerprint
from fse._partition import BLOCK_ROWS, partition
from fse._stream import StreamSource
from fse._text import ArrowSource, StringSource, TokenListSource, Window, is_arrow
from fse._text._ids import IdSource
from fse._text_file import FileSource

from ._doc_ids import ShardSinkMarker
from ._inputs import COLLECTIONS, refuse
from ._sources import SOURCES, opened

_BLOCK_BYTES = 1 << 24  # the bytes of a file hashed (and searched for lines) at a time
_BLOCK_IDS = 1 << 16  # the document IDs hashed at a time
_FORMS = {StringSource: "strings", TokenListSource: "tokens", FileSource: "lines"}


class ResumableSinkMarker(ShardSinkMarker):
    """Base of the sinks that can resume (``ShardSinkBase``). ``resume`` is True for one
    made with ``resume=True``: it takes the rows of ``encode(texts, shard=)`` only."""

    resume = False


def is_resume_sink(sink: Any) -> bool:
    """Whether ``sink`` is a sink of fse made with ``resume=True``."""
    return isinstance(sink, ResumableSinkMarker) and sink.resume is True


def refuse_resume_sink(sink: Any) -> None:
    """The entries that do not resume refuse a sink made with ``resume=True``, before they
    read anything."""
    if is_resume_sink(sink):
        raise ValueError(
            "a sink made with resume=True takes the rows of Encoder.encode(texts, sink=sink) "
            "of a source that can be read again; use a sink without resume here"
        )


def wanted(sink: Any, shard: Any) -> bool:
    """Whether a call to ``encode`` runs as a resumed run."""
    return shard is not None or is_resume_sink(sink)


def check_shard(shard: Any) -> tuple[int, int]:
    """``shard`` as ``(i, n)``: two ints (no bool) with ``n >= 1`` and ``0 <= i < n``, or
    (0, 1) for None, the whole source."""
    if shard is None:
        return 0, 1
    if (
        not isinstance(shard, tuple)
        or len(shard) != 2
        or any(isinstance(v, bool) or not isinstance(v, int | np.integer) for v in shard)
    ):
        raise TypeError(f"shard must be a tuple (i, n) of two ints, not {shard!r}")
    index, count = int(shard[0]), int(shard[1])
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard must be (i, n) with n >= 1 and 0 <= i < n, not {shard!r}")
    return index, count


def check_texts(texts: Any) -> None:
    """Raise unless ``texts`` can be read again: a stream or an Arrow column cannot (its
    rows are not known before the run, and a run that stops cannot read them twice)."""
    if isinstance(texts, StreamSource | ArrowSource) or is_arrow(texts):
        raise TypeError(
            f"a resumed run reads its source more than once; {type(texts).__name__} cannot be "
            "read again. Pass a list, tuple or array of str, token lists or a path of a text file"
        )
    if isinstance(texts, (*SOURCES, os.PathLike, *COLLECTIONS)):
        return
    refuse(texts)
    raise TypeError(
        f"a resumed run reads its source more than once; a {type(texts).__name__} is a "
        "stream and cannot be read again. Pass a list, tuple or array of str, token lists "
        "or a path of a text file"
    )


def check_arguments(texts: Any, sink: Any, shard: Any) -> tuple[int, int]:
    """The partition of a resumed run, after the checks that read nothing of the source."""
    part = check_shard(shard)
    if not is_resume_sink(sink):
        raise ValueError(
            "shard= needs a sink made with resume=True, such as "
            "fse.ShardSink(path, encoder, resume=True)"
        )
    sink._check_open()
    check_texts(texts)
    return part


@dataclass(frozen=True)
class Identity:
    """What a source is: the fingerprint of its bytes, its form and its number of rows, and
    for a file the byte at which every 16,384th line starts."""

    fingerprint: str
    form: str
    rows: int
    nbytes: int
    marks: tuple[int, ...] = ()


def _identity(form: str, rows: int, entries: dict, ids: bool) -> str:
    metadata = {
        "fse.kind": "text_source",
        "fse.source_form": form,
        "fse.rows": str(rows),
        "fse.doc_ids": "int64" if ids else "none",
    }
    return fingerprint(entries, metadata)


def _id_entry(values: IdSource) -> tuple[tuple[str, tuple[int, ...], str], int]:
    hasher, nbytes = Hasher(), 0
    for begin in range(0, len(values), _BLOCK_IDS):
        block = values.block(begin, min(len(values), begin + _BLOCK_IDS))
        hasher.update(block)
        nbytes += block.nbytes
    return ("I64", (len(values),), hasher.checksum()), nbytes


def _hash_lines(source: FileSource) -> tuple[str, int, int, tuple[int, ...]]:
    """One pass over the bytes of the file: their checksum, the number of lines, the number
    of bytes and the byte at which every 16,384th line starts."""
    data = source._array()
    hasher, count, marks = Hasher(), 0, [0]
    for begin in range(0, len(data), _BLOCK_BYTES):
        block = data[begin : begin + _BLOCK_BYTES]
        hasher.update(block)
        newlines = np.flatnonzero(block == 10)
        starts = np.arange(count + 1, count + 1 + len(newlines))  # the lines after a newline
        marks.extend((begin + newlines[starts % BLOCK_ROWS == 0] + 1).tolist())
        count += len(newlines)
    rows = count + (1 if len(data) and data[-1] != 10 else 0)
    return hasher.checksum(), rows, len(data), tuple(marks)


def identify(source: Any, values: IdSource | None) -> Identity:
    """The identity of ``source``, with the IDs ``values`` if the sink stores them. The
    whole source is hashed: a change of one byte, of the order or of an ID changes it."""
    form = next(name for kind, name in _FORMS.items() if isinstance(source, kind))
    entries: dict[str, tuple[str, tuple[int, ...], str]]
    if form == "lines":
        digest, rows, nbytes, marks = _hash_lines(source)
        entries = {"text": ("U8", (nbytes,), digest)}
    else:
        rows = len(source)
        text = source._array()
        entries = {
            "offsets": ("I64", (rows + 1,), checksum(source._offsets)),
            "text": ("U8", (len(text),), checksum(text)),
        }
        nbytes, marks = len(text) + 8 * (rows + 1), ()
    if values is not None:
        if len(values) != rows:
            raise ValueError(f"doc_ids has {len(values)} rows for {rows} texts")
        entries["doc_ids"], extra = _id_entry(values)
        nbytes += extra
    source._check_open()  # a file that changed while it was hashed
    return Identity(_identity(form, rows, entries, values is not None), form, rows, nbytes, marks)


def _line_start(source: FileSource, identity: Identity, row: int) -> int:
    """The byte at which line ``row`` starts, from the mark before it."""
    data = source._array()
    if row >= identity.rows:
        return len(data)
    base = row // BLOCK_ROWS
    position, left = identity.marks[base], row - base * BLOCK_ROWS
    while left:
        block = data[position : position + (1 << 20)]
        newlines = np.flatnonzero(block == 10)
        if len(newlines) >= left:
            return position + int(newlines[left - 1]) + 1
        left -= len(newlines)
        position += len(block)
    return position


def window(source: Any, identity: Identity, first: int, last: int) -> Window:
    """The rows ``first`` to ``last - 1`` of ``source`` as a source of their own."""
    if identity.form == "lines":
        low = _line_start(source, identity, first)
        return Window.of_bytes(source, low, _line_start(source, identity, last), last - first)
    return Window.of_rows(source, first, last)


@dataclass
class Resumed:
    """A resumed run: the rows the store held (``kept``), the source row where the run
    starts and the window of rows to run, with the IDs of those."""

    source: Window
    ids: Any
    kept: int
    first_row: int
    base: Any

    @property
    def rows(self) -> int:
        return len(self.source)

    def check_unchanged(self) -> None:
        """Raise if the file changed during the run."""
        self.base._check_open()


def add_row_note(error: Exception, first_row: int) -> None:
    """Say which row of the source the rows of an error of a resumed run are: the run counts
    from its first row (the rows the store holds are not part of it)."""
    error.add_note(
        f"this run resumed at row {first_row} of the source; add {first_row} to the row of "
        "this message for the row in the source"
    )


@contextmanager
def prepared(
    texts: Any, sink: Any, shard: tuple[int, int], values: Any, tokenizer: str
) -> Iterator[Resumed]:
    """Open the source of a resumed run, hash it, let ``sink`` compare the identity with its
    store (``ValueError`` if they differ, before any row is reserved) and give the window of
    the rows still to run. ``values`` are the document IDs of the call, or None."""
    with opened(texts, tokenizer) as source, _ids(values) as ids:
        start = time.perf_counter()
        identity = identify(source, ids)
        seconds = time.perf_counter() - start
        kept = sink._attach_source(identity, shard, seconds)
        first, last = partition(identity.rows, *shard)
        if kept > last - first:
            raise ValueError(
                f"the store holds {kept} rows, more than the {last - first} rows of its part"
            )
        run = window(source, identity, first + kept, last)
        try:
            yield Resumed(
                run,
                None if ids is None else ids.window(first + kept, last),
                kept,
                first + kept,
                source,
            )
        finally:
            run.close()


@contextmanager
def _ids(values: Any) -> Iterator[IdSource | None]:
    if values is None:
        yield None
        return
    with IdSource(values) as opened_ids:
        yield opened_ids
