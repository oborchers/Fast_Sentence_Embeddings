#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Chunks and scans of the sources whose sentences are one UTF-8 buffer (``_Buffered``):
``scan`` validates and counts the sentences in threads without the GIL and cuts them into
chunks, ``encode`` turns one chunk into a Batch (ARCHITECTURE.md, "Text sources")."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _native
from fse._contract import Batch

from ._plan import offset_cuts

# Bytes per chunk: small enough to balance workers, large enough that the per-chunk
# overhead does not matter.
_CHUNK_MIN_BYTES = 1 << 16
_CHUNK_MAX_BYTES = 1 << 20
# About this many bytes per scan task, so a failed scan stops after bounded work.
_SCAN_RANGE_MAX_BYTES = 1 << 26
# Sources smaller than this are scanned in the calling thread: starting threads would cost
# more than the scan (about 16 ms of one core at 64 MB/s).
_SCAN_INLINE_BYTES = 1 << 20


@dataclass(frozen=True)
class Chunk:
    """Sentences of a source that one worker encodes into one batch.

    ``start`` and ``end`` are bytes in a file and string numbers in memory; the chunk
    holds rows ``first_row`` to ``first_row + n_sentences - 1``.
    """

    start: int
    end: int
    first_row: int
    n_sentences: int


@dataclass(frozen=True)
class Scan:
    """The result of scanning a source: its number of sentences and its chunks."""

    sentences: int
    chunks: tuple[Chunk, ...]


def _decode_error(data: Any, start: int, end: int, sentence: int) -> UnicodeDecodeError:
    raw = bytes(data[start:end])
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as err:
        return UnicodeDecodeError(
            err.encoding,
            err.object,
            err.start,
            err.end,
            f"{err.reason} in sentence {sentence} (byte {start + err.start} of the source)",
        )
    raise AssertionError(f"no UTF-8 error in sentence {sentence}")  # pragma: no cover


def _scan_pool(workers: int, total: int) -> AbstractContextManager[ThreadPoolExecutor | None]:
    """Threads for a scan, or None to scan in the calling thread: with one worker and for
    sources under _SCAN_INLINE_BYTES. The chunks are the same either way."""
    if workers == 1 or total < _SCAN_INLINE_BYTES:
        return nullcontext(None)
    return ThreadPoolExecutor(max_workers=workers)


def _integer(value: object) -> int | None:
    """``value`` as an int if it is one (numpy integers too, bool not), else None."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        return None
    return int(value)


def check_chunk_bytes(chunk_bytes: object) -> int | None:
    """None (a default from the source size) or a positive int."""
    if chunk_bytes is None:
        return None
    value = _integer(chunk_bytes)
    if value is None or value < 1:
        raise ValueError(f"chunk_bytes must be a positive int, not {chunk_bytes!r}")
    return value


def _chunk_sizes(
    total: int, workers: object, chunk_bytes: object, chunk_rows: object
) -> tuple[int, int, int]:
    """The checked workers, the chunk size in bytes (by default from the source size) and
    in rows."""
    count = _integer(workers)
    if count is None or count < 1:
        raise ValueError(f"workers must be a positive int, not {workers!r}")
    size = check_chunk_bytes(chunk_bytes)
    if size is None:
        size = max(_CHUNK_MIN_BYTES, min(_CHUNK_MAX_BYTES, total // (8 * count)))
    rows = _native.MAX_CHUNK_ROWS if chunk_rows is None else _integer(chunk_rows)
    if rows is None or not 1 <= rows <= _native.MAX_CHUNK_ROWS:
        raise ValueError(
            f"chunk_rows must be an int from 1 to {_native.MAX_CHUNK_ROWS}, not {chunk_rows!r}"
        )
    return count, size, rows


def _append(chunks: list[Chunk], chunk: Chunk, *, merge: bool, remainder: Any, chunk_rows: int):
    """Append ``chunk``; the first chunk of a scan range takes in the last chunk of the
    range before if that one is a remainder (it ended at the range boundary below both
    chunk limits) and their rows fit."""
    last = chunks[-1] if merge and chunks else None
    if last is not None and remainder(last) and last.n_sentences + chunk.n_sentences <= chunk_rows:
        chunks[-1] = Chunk(
            last.start, chunk.end, last.first_row, last.n_sentences + chunk.n_sentences
        )
    else:
        chunks.append(chunk)


class _Buffered:
    """A source whose sentences are one UTF-8 buffer, with offsets for strings."""

    _data: Any
    _offsets: ndarray | None

    @property
    def nbytes(self) -> int:
        """The bytes of the UTF-8 text of all sentences (with newlines for a file)."""
        return len(self._data)

    def _positions(self) -> int:
        """Bytes in line mode, strings otherwise."""
        return len(self._data) if self._offsets is None else len(self._offsets) - 1

    def _range_bounds(self, workers: int, chunk_bytes: int) -> list[int]:
        """Scan ranges of similar byte size that start at sentence starts."""
        size = self._positions()
        total = self.nbytes
        # 4 ranges per worker for balance, but none smaller than a chunk: each range gives
        # at least one chunk, and every chunk costs its task's fixed checks.
        wanted = max(4 * workers, -(-total // _SCAN_RANGE_MAX_BYTES))
        n_ranges = max(1, min(wanted, total // chunk_bytes, max(size, 1)))
        bounds = [0]
        for k in range(1, n_ranges):
            target = total * k // n_ranges
            position = self._start_at_or_after(target)
            if position > bounds[-1]:
                bounds.append(position)
        if bounds[-1] != size:
            bounds.append(size)
        return bounds

    def _start_at_or_after(self, byte: int) -> int:
        """The first position at or after ``byte`` where a sentence starts."""
        if self._offsets is None:
            if byte == 0 or self._data[byte - 1] == 10:
                return byte
            newline = bytes(self._data[byte : byte + (1 << 20)]).find(b"\n")
            while newline < 0 and byte < len(self._data):
                byte += 1 << 20
                newline = bytes(self._data[byte : byte + (1 << 20)]).find(b"\n")
            return len(self._data) if newline < 0 else byte + newline + 1
        return int(np.searchsorted(self._offsets, byte, side="right")) - 1

    def scan(
        self, *, workers: int = 1, chunk_bytes: int | None = None, chunk_rows: int | None = None
    ) -> Scan:
        """Validate the UTF-8 of every sentence and cut the source into chunks of about
        ``chunk_bytes`` bytes and at most ``chunk_rows`` sentences (MAX_CHUNK_ROWS)."""
        scan, error = self.scan_until_error(
            workers=workers, chunk_bytes=chunk_bytes, chunk_rows=chunk_rows
        )
        if error is not None:
            raise error
        return scan

    def scan_until_error(
        self, *, workers: int = 1, chunk_bytes: int | None = None, chunk_rows: int | None = None
    ) -> tuple[Scan, Exception | None]:
        """As ``scan``, but the first invalid sentence ends the chunks instead of raising:
        the chunks of the sentences before it and its error (or None). The runtime encodes
        those sentences first, so an earlier row that fails later in its pipeline is
        reported before it (ARCHITECTURE.md, "Runtime", Errors)."""
        self._check_open()
        total = self.nbytes
        workers, chunk_bytes, chunk_rows = _chunk_sizes(total, workers, chunk_bytes, chunk_rows)
        if self._positions() == 0:
            return Scan(0, ()), None
        if self._offsets is not None:
            return self._offset_scan(workers, chunk_bytes, chunk_rows)
        scan = self._scanner(chunk_bytes, chunk_rows)
        ranges = list(pairwise(self._range_bounds(workers, chunk_bytes)))

        def remainder(c: Chunk) -> bool:
            return self._span_bytes(c) < chunk_bytes and c.n_sentences < chunk_rows

        sentences = 0
        chunks: list[Chunk] = []
        with _scan_pool(workers, total) as pool:
            for span, result in zip(
                ranges, (map if pool is None else pool.map)(scan, ranges), strict=True
            ):
                error_byte, error_sentence = result[4], result[5]
                error = None
                if error_byte >= 0:
                    error = self._error(error_byte, sentences + error_sentence)
                    result = self._valid_part(scan, span[0], error_byte, error_sentence)
                count, _, _, range_chunks, _, _ = result
                firsts = [row for _, _, row in range_chunks] + [count]
                for k, ((start, end, row), following) in enumerate(
                    zip(range_chunks, firsts[1:], strict=True)
                ):
                    chunk = Chunk(start, end, sentences + row, following - row)
                    _append(chunks, chunk, merge=k == 0, remainder=remainder, chunk_rows=chunk_rows)
                sentences += count
                if error is not None:
                    return Scan(sentences, tuple(chunks)), error
        return Scan(sentences, tuple(chunks)), None

    def _offset_scan(
        self, workers: int, chunk_bytes: int, chunk_rows: int
    ) -> tuple[Scan, Exception | None]:
        """The scan of sentences with offsets: validated, then cut from the offsets into
        chunks of about equal bytes (``offset_cuts``)."""
        rows, error = self._validated(workers, chunk_bytes)
        assert self._offsets is not None
        cuts = offset_cuts(self._offsets[: rows + 1], chunk_bytes, chunk_rows).tolist()
        chunks = tuple(Chunk(a, b, a, b - a) for a, b in pairwise(cuts))
        return Scan(rows, chunks), error

    def _validated(self, workers: int, chunk_bytes: int) -> tuple[int, Exception | None]:
        """The sentences before the first invalid one, and its error (or None)."""
        scan = self._scanner(1 << 62, _native.MAX_CHUNK_ROWS)
        ranges = list(pairwise(self._range_bounds(workers, chunk_bytes)))
        sentences = 0
        with _scan_pool(workers, self.nbytes) as pool:
            for result in (map if pool is None else pool.map)(scan, ranges):
                count, _, _, _, error_byte, error_sentence = result
                if error_byte >= 0:
                    return sentences + error_sentence, self._error(
                        error_byte, sentences + error_sentence
                    )
                sentences += count
        return sentences, None

    def _valid_part(self, scan: Callable, first: int, byte: int, sentences: int) -> Any:
        """The scan of the valid sentences of a range before its first invalid one."""
        if not sentences:
            return 0, 0, 0, [], -1, -1
        return scan((first, self._start_of(first, byte, sentences)))

    def _start_of(self, first: int, byte: int, sentences: int) -> int:
        """The position of the sentence that holds ``byte``, the sentence ``sentences`` of
        the range from ``first``: a string or row number, or in line mode its first byte."""
        if not self._lines():
            return first + sentences
        newlines = np.flatnonzero(self._array()[first:byte] == 10)
        return first + int(newlines[-1]) + 1 if len(newlines) else first

    def _lines(self) -> bool:
        """Whether positions are bytes of lines (a file) rather than string numbers."""
        return self._offsets is None

    def prefix(self, chunk: Chunk, rows: int) -> Chunk:
        """The chunk of the first ``rows`` sentences of ``chunk`` (1 to all of them)."""
        if not 0 < rows <= chunk.n_sentences:
            raise ValueError(f"a prefix of {rows} of the {chunk.n_sentences} sentences of a chunk")
        if rows == chunk.n_sentences:
            return chunk
        if self._lines():
            newlines = np.flatnonzero(self._array()[chunk.start : chunk.end] == 10)
            end = chunk.start + int(newlines[rows - 1]) + 1
        else:
            end = chunk.start + rows
        return Chunk(chunk.start, end, chunk.first_row, rows)

    def _scanner(self, chunk_bytes: int, chunk_rows: int) -> Callable[[tuple[int, int]], Any]:
        """The scan of one range of positions, as ``_native.scan_spans`` gives it: (sentences,
        tokens, empty, chunks, error byte, error sentence relative to the range)."""
        data, offsets = self._array(), self._offsets

        def scan(span: tuple[int, int]):
            return _native.scan_spans(data, offsets, span[0], span[1], chunk_bytes, chunk_rows)

        return scan

    def _span_bytes(self, chunk: Chunk) -> int:
        """The bytes of the sentences of ``chunk``."""
        if self._offsets is None:
            return chunk.end - chunk.start
        return int(self._offsets[chunk.end] - self._offsets[chunk.start])

    def _error(self, byte: int, sentence: int) -> UnicodeDecodeError:
        """The error of an invalid UTF-8 byte of the scan: ``byte`` where the scan gave it,
        ``sentence`` the row."""
        if self._offsets is not None:
            k = int(np.searchsorted(self._offsets, byte, side="right")) - 1
            return _decode_error(
                self._data, int(self._offsets[k]), int(self._offsets[k + 1]), sentence
            )
        start = byte
        while start > 0 and self._data[start - 1] != 10:
            start -= 1
        end = byte
        while end < len(self._data) and self._data[end] != 10:
            end += 1
        return _decode_error(self._data, start, end, sentence)

    def _check_open(self) -> None:
        """Raise if the source was closed; only a FileSource can be."""

    def _array(self) -> ndarray:
        self._check_open()
        return (
            np.frombuffer(self._data, dtype=np.uint8) if len(self._data) else np.zeros(0, np.uint8)
        )

    def encode(
        self, chunk: Chunk, lookup: _native.TextLookup, workspace: _native.Workspace
    ) -> Batch:
        """The Batch of one chunk of this source."""
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunk must be a Chunk, not {type(chunk).__name__}")
        batch = lookup.encode(
            self._array(), self._offsets, chunk.start, chunk.end, chunk.first_row, workspace
        )
        if batch.n_sentences != chunk.n_sentences:
            raise RuntimeError("the source changed since it was scanned")
        return batch
