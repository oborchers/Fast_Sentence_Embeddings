#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Sentences from an iterator (a queue consumer, a crawler, log lines), read once.

The runtime reads the iterator in the calling thread, a collection of at most
``chunk_rows`` sentences or about ``chunk_bytes`` bytes at a time, and only when its
window of tasks has room: a slow sink slows the source, and memory stays bounded by one
collection and the window (docs/architecture/runtime.md, "Runtime"). The workers keep encoding the
tasks taken before. Row numbers count from the first sentence of the stream.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from fse._text import MAX_CHUNK_ROWS, StringSource


@dataclass(frozen=True)
class Collection:
    first_row: int
    source: StringSource
    last: bool  # the stream ends (or fails) after it


class StreamSource:
    """Sentences (str) from any iterable that is no str, bytes, set or mapping."""

    def __init__(
        self, texts: Iterable[str], *, chunk_rows: int = MAX_CHUNK_ROWS, chunk_bytes: int = 1 << 23
    ):
        if isinstance(texts, str | bytes | bytearray | memoryview):
            raise TypeError(f"texts must be an iterable of str, not a {type(texts).__name__}")
        if isinstance(texts, set | frozenset | Mapping):
            raise TypeError(f"texts must be ordered, not a {type(texts).__name__}")
        try:
            self._texts = iter(texts)
        except TypeError:
            raise TypeError(f"texts must be iterable, not {type(texts).__name__}") from None
        self.chunk_rows = _bounded(chunk_rows, "chunk_rows", MAX_CHUNK_ROWS)
        self.chunk_bytes = _bounded(chunk_bytes, "chunk_bytes", 1 << 40)
        self._read = False

    def collections(
        self, stop: threading.Event | None = None, first_row: int = 0
    ) -> Iterator[Collection]:
        """The collections of the stream, in order. An error of the iterator or a bad
        sentence is raised after the collection of the sentences before it, naming its row
        from ``first_row`` (the row of the run of the first sentence). Once ``stop`` is set,
        no further sentence is read."""
        if self._read:
            raise RuntimeError("a StreamSource is read once")
        self._read = True
        row = 0
        while stop is None or not stop.is_set():
            encoded, error, ended = self._collect(first_row + row, stop)
            if encoded:
                yield Collection(row, StringSource.encoded(encoded), ended or error is not None)
                row += len(encoded)
            if error is not None:
                raise error
            if ended:
                return

    def _collect(
        self, row: int, stop: threading.Event | None
    ) -> tuple[list[bytes], Exception | None, bool]:
        """Up to chunk_rows sentences or about chunk_bytes bytes; a tight loop, as it runs
        once per sentence in the calling thread while the workers wait for tasks."""
        encoded: list[bytes] = []
        append, rows, limit, size = encoded.append, self.chunk_rows, self.chunk_bytes, 0
        try:
            for text in self._texts:
                data = (
                    text.encode("utf-8") if type(text) is str else _fail(text, row + len(encoded))
                )
                append(data)
                size += len(data)
                if len(encoded) >= rows or size >= limit:
                    return encoded, None, False
                if stop is not None and stop.is_set():  # a task failed: read no further
                    return encoded, None, True
        except UnicodeEncodeError as err:  # a lone surrogate: name the row of the stream
            return encoded, _with_row(err, row + len(encoded)), False
        except Exception as error:  # raised once the sentences before it are done
            return encoded, error, False
        return encoded, None, True


def _fail(text: object, row: int) -> bytes:
    """A sentence that is no exact str: a str subclass is encoded, anything else raises."""
    if isinstance(text, str):
        return str.encode(text, "utf-8")
    raise TypeError(f"string {row} is a {type(text).__name__}, not a str")


def _with_row(err: UnicodeEncodeError, row: int) -> UnicodeEncodeError:
    return UnicodeEncodeError(
        err.encoding, err.object, err.start, err.end, f"{err.reason} in string {row}"
    )


def _bounded(value: object, name: str, top: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or not 1 <= value <= top:
        raise ValueError(f"{name} must be an int from 1 to {top}, not {value!r}")
    return int(value)
