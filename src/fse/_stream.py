#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Sentences from an iterator (a queue consumer, a crawler, log lines), read once.

The runtime reads the iterator in the calling thread, a collection of at most
``chunk_rows`` sentences or about ``chunk_bytes`` bytes at a time, and only when its
window of tasks has room: a slow sink slows the source, and memory stays bounded by one
collection and the window (ARCHITECTURE.md, "Runtime"). The workers keep encoding the
tasks taken before. Row numbers count from the first sentence of the stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from fse._text import MAX_CHUNK_ROWS, StringSource, encode_string


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

    def collections(self) -> Iterator[Collection]:
        """The collections of the stream, in order. An error of the iterator or a bad
        sentence is raised after the collection of the sentences before it."""
        if self._read:
            raise RuntimeError("a StreamSource is read once")
        self._read = True
        row = 0
        while True:
            encoded, error, ended = self._collect(row)
            if encoded:
                yield Collection(row, StringSource.encoded(encoded), ended or error is not None)
                row += len(encoded)
            if error is not None:
                raise error
            if ended:
                return

    def _collect(self, row: int) -> tuple[list[bytes], Exception | None, bool]:
        encoded: list[bytes] = []
        size = 0
        while len(encoded) < self.chunk_rows and size < self.chunk_bytes:
            try:
                text = next(self._texts)
                encoded.append(encode_string(text, "string", row + len(encoded)))
            except StopIteration:
                return encoded, None, True
            except Exception as error:  # raised once the sentences before it are done
                return encoded, error, False
            size += len(encoded[-1])
        return encoded, None, False


def _bounded(value: object, name: str, top: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= top:
        raise ValueError(f"{name} must be an int from 1 to {top}, not {value!r}")
    return value
