#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Chunks and scans of the sources whose sentences are one UTF-8 buffer (``_Buffered``):
``scan`` validates and counts the sentences in threads without the GIL and cuts them into
chunks, ``encode`` turns one chunk into a Batch (docs/architecture/text.md, "Text sources")."""

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
# Tasks per worker of a source below TASKS_PER_WORKER * workers * _CHUNK_MAX_BYTES: a task
# takes the GIL back at about ten points, which costs about 19 us at one worker but 0.2
# to 0.34 ms of CPU with 10 threads, so fewer and larger tasks are faster; two per worker
# leave room to balance cores of different speed (docs/architecture/runtime.md, "Runtime", Tasks).
TASKS_PER_WORKER = 2
# About this many bytes per scan task, so a failed scan stops after bounded work.
_SCAN_RANGE_MAX_BYTES = 1 << 26
# Sources smaller than this are scanned in the calling thread: starting threads would cost
# more than the scan (about 16 ms of one core at 64 MB/s).
_SCAN_INLINE_BYTES = 1 << 20
# Bytes searched at a time for a newline on the path of an error (``prefix``, ``_start_of``).
_LINE_BLOCK = 1 << 20


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
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fse-text")


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


def default_chunk_bytes(total: int, workers: int, per_worker: int | None = None) -> int:
    """The bytes per chunk of a source of ``total`` bytes: about ``per_worker`` chunks
    (TASKS_PER_WORKER) per worker, within 64 KiB and 1 MiB."""
    per = TASKS_PER_WORKER if per_worker is None else per_worker
    return max(_CHUNK_MIN_BYTES, min(_CHUNK_MAX_BYTES, -(-total // (per * workers))))


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
        size = default_chunk_bytes(total, count)
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
    # True only for a buffer that the native join wrote from exact str (``_verify``): it
    # is valid UTF-8 by construction and is not scanned again (docs/architecture/text.md, "Text
    # sources"). Everything else (bytes of a caller, of a str subclass's own encode, of a
    # file or a producer) is scanned.
    _verified = False

    def _verify(self, data: bytes, offsets: ndarray) -> None:
        """Mark the buffer of a successful native join as valid. The join reads only exact,
        ready str, writes each code point in its shortest UTF-8 form (ASCII copied, a
        surrogate fails the pass, CPython keeps code points at most U+10FFFF), and writes
        the offsets itself, from 0 to the length of the data; they are made read-only."""
        if not (
            isinstance(data, bytes)
            and isinstance(offsets, ndarray)
            and offsets.dtype == np.int64
            and offsets.ndim == 1
            and len(offsets) >= 1
            and offsets[0] == 0
            and offsets[-1] == len(data)
        ):
            raise AssertionError("the native join gave a buffer without its offsets")
        offsets.flags.writeable = False
        self._verified = True

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
        self,
        *,
        workers: int = 1,
        chunk_bytes: int | None = None,
        chunk_rows: int | None = None,
        first_row: int = 0,
    ) -> tuple[Scan, Exception | None]:
        """As ``scan``, but the first invalid sentence ends the chunks instead of raising:
        the chunks of the sentences before it and its error (or None). The runtime encodes
        those sentences first, so an earlier row that fails later in its pipeline is
        reported before it (docs/architecture/runtime.md, "Runtime", Errors). The error names its row
        from ``first_row``, the row of the run of this source's first sentence."""
        self._check_open()
        total = self.nbytes
        workers, chunk_bytes, chunk_rows = _chunk_sizes(total, workers, chunk_bytes, chunk_rows)
        if self._positions() == 0:
            return Scan(0, ()), None
        if self._offsets is not None:
            return self._offset_scan(workers, chunk_bytes, chunk_rows, first_row)
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
                error_byte, error_sentence = result[2], result[3]
                error = None
                if error_byte >= 0:
                    error = self._error(error_byte, sentences + error_sentence, first_row)
                    result = self._valid_part(scan, span[0], error_byte, error_sentence)
                count, range_chunks, _, _ = result
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
        self, workers: int, chunk_bytes: int, chunk_rows: int, first_row: int
    ) -> tuple[Scan, Exception | None]:
        """The scan of sentences with offsets: validated, then cut from the offsets into
        chunks of about equal bytes (``offset_cuts``)."""
        rows, error = self._validated(workers, chunk_bytes, first_row)
        assert self._offsets is not None
        cuts = offset_cuts(self._offsets[: rows + 1], chunk_bytes, chunk_rows).tolist()
        chunks = tuple(Chunk(a, b, a, b - a) for a, b in pairwise(cuts))
        return Scan(rows, chunks), error

    def _validated(
        self, workers: int, chunk_bytes: int, first_row: int
    ) -> tuple[int, Exception | None]:
        """The sentences before the first invalid one, and its error (or None); a verified
        buffer has no invalid sentence."""
        if self._verified:
            return self._positions(), None
        scan = self._scanner(1 << 62, _native.MAX_CHUNK_ROWS)
        ranges = list(pairwise(self._range_bounds(workers, chunk_bytes)))
        sentences = 0
        with _scan_pool(workers, self.nbytes) as pool:
            for result in (map if pool is None else pool.map)(scan, ranges):
                count, _, error_byte, error_sentence = result
                if error_byte >= 0:
                    return sentences + error_sentence, self._error(
                        error_byte, sentences + error_sentence, first_row
                    )
                sentences += count
        return sentences, None

    def _valid_part(self, scan: Callable, first: int, byte: int, sentences: int) -> Any:
        """The scan of the valid sentences of a range before its first invalid one."""
        if not sentences:
            return 0, [], -1, -1
        return scan((first, self._start_of(first, byte, sentences)))

    def _start_of(self, first: int, byte: int, sentences: int) -> int:
        """The position of the sentence that holds ``byte``, the sentence ``sentences`` of
        the range from ``first``: a string or row number, or in line mode its first byte
        (searched back from ``byte`` in blocks, so no array of the range is made)."""
        if not self._lines():
            return first + sentences
        data = self._array()
        end = byte
        while end > first:
            start = max(first, end - _LINE_BLOCK)
            newlines = np.flatnonzero(data[start:end] == 10)
            if len(newlines):
                return start + int(newlines[-1]) + 1
            end = start
        return first

    def _lines(self) -> bool:
        """Whether positions are bytes of lines (a file) rather than string numbers."""
        return self._offsets is None

    def prefix(self, chunk: Chunk, rows: int) -> Chunk:
        """The chunk of the first ``rows`` sentences of ``chunk`` (1 to all of them)."""
        if not 0 < rows <= chunk.n_sentences:
            raise ValueError(f"a prefix of {rows} of the {chunk.n_sentences} sentences of a chunk")
        if rows == chunk.n_sentences:
            return chunk
        end = self._line_end(chunk.start, chunk.end, rows) if self._lines() else chunk.start + rows
        return Chunk(chunk.start, end, chunk.first_row, rows)

    def split_tail(self, chunks: tuple[Chunk, ...], workers: int) -> tuple[Chunk, ...]:
        """``chunks`` with the last ``workers`` of them cut into two parts of about equal
        rows (factoring), so the workers that finish last hold the least work. A chunk of
        fewer than 2 sentences, or one of whose parts would be below _CHUNK_MIN_BYTES, stays
        whole. The chunks stay contiguous, in row order and non-empty, and the result
        depends only on the arguments."""
        first = max(0, len(chunks) - workers)
        return (
            *chunks[:first],
            *(part for chunk in chunks[first:] for part in self._halves(chunk)),
        )

    def _halves(self, chunk: Chunk) -> tuple[Chunk, ...]:
        """``chunk`` cut into its first ``n // 2`` sentences and the rest, or whole."""
        rows = chunk.n_sentences
        if rows < 2 or self._span_bytes(chunk) < 2 * _CHUNK_MIN_BYTES:
            return (chunk,)
        head = self.prefix(chunk, rows // 2)
        rest = Chunk(
            head.end, chunk.end, chunk.first_row + head.n_sentences, rows - head.n_sentences
        )
        if min(self._span_bytes(head), self._span_bytes(rest)) < _CHUNK_MIN_BYTES:
            return (chunk,)  # the rows are unequal in bytes: a part would be too small
        return head, rest

    def _line_end(self, start: int, end: int, rows: int) -> int:
        """The byte after the ``rows``-th newline from ``start``, searched in blocks up to
        it, so no array of the chunk is made."""
        data = self._array()
        while start < end:
            stop = min(end, start + _LINE_BLOCK)
            newlines = np.flatnonzero(data[start:stop] == 10)
            if len(newlines) >= rows:
                return start + int(newlines[rows - 1]) + 1
            rows -= len(newlines)
            start = stop
        raise AssertionError("a chunk has fewer lines than its sentences")  # pragma: no cover

    def _scanner(self, chunk_bytes: int, chunk_rows: int) -> Callable[[tuple[int, int]], Any]:
        """The scan of one range of positions, as ``_native.scan_spans`` gives it: (sentences,
        chunks, error byte, error sentence relative to the range)."""
        data, offsets = self._array(), self._offsets

        def scan(span: tuple[int, int]):
            return _native.scan_spans(data, offsets, span[0], span[1], chunk_bytes, chunk_rows)

        return scan

    def _span_bytes(self, chunk: Chunk) -> int:
        """The bytes of the sentences of ``chunk``."""
        if self._offsets is None:
            return chunk.end - chunk.start
        return int(self._offsets[chunk.end] - self._offsets[chunk.start])

    def _error(self, byte: int, sentence: int, first_row: int = 0) -> UnicodeDecodeError:
        """The error of an invalid UTF-8 byte of the scan: ``byte`` where the scan gave it,
        ``sentence`` the row, named from ``first_row``."""
        if self._offsets is not None:
            k = int(np.searchsorted(self._offsets, byte, side="right")) - 1
            return _decode_error(
                self._data, int(self._offsets[k]), int(self._offsets[k + 1]), first_row + sentence
            )
        start = byte
        while start > 0 and self._data[start - 1] != 10:
            start -= 1
        end = byte
        while end < len(self._data) and self._data[end] != 10:
            end += 1
        return _decode_error(self._data, start, end, first_row + sentence)

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

    def encode_with(self, chunk: Chunk, run: Callable[..., Any]) -> Any:
        """``run(text, first, last, rows, first_row, after_encode)`` on the text of
        ``chunk``: a backend's stages 1 to 4 in one native call (``CpuBackend.text_rows``),
        which checks the sentence count of the chunk as ``encode`` does and runs
        ``after_encode`` (the source's own check after encoding, or None) before the
        errors of the stages after it."""
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunk must be a Chunk, not {type(chunk).__name__}")
        return run(
            (self._array(), self._offsets),
            chunk.start,
            chunk.end,
            chunk.n_sentences,
            chunk.first_row,
            self._after_encode(),
        )

    def _after_encode(self) -> Callable[[], None] | None:
        """The source's check after a chunk was encoded, or None."""
        return None
