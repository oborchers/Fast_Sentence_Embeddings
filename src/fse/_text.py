"""Text sources: sentences of text to batches, stages 1 and 2 (ARCHITECTURE.md, "Tokenizers").

A source holds its sentences as one UTF-8 buffer: strings in memory with their offsets
(``StringSource``, ``TokenListSource``), or the lines of a memory-mapped file
(``FileSource``). ``scan`` validates and counts them in threads without the GIL and cuts
them into chunks; ``encode`` turns one chunk into a Batch with a ``TextLookup`` and a
per-worker ``Workspace``. Every sentence is one row, empty ones included.
"""

from __future__ import annotations

import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _native
from fse._contract import EXTERNAL_TOKENIZER, Batch, check_tokenizer
from fse._wordvectors import WordVectors

# Compressed files hold no text lines; read them decompressed.
_COMPRESSED_SUFFIXES = {".gz", ".bz2", ".xz", ".lzma", ".zst", ".zstd", ".z", ".zip"}
# Bytes per chunk: small enough to balance workers, large enough that the per-chunk
# overhead does not matter.
_CHUNK_MIN_BYTES = 1 << 16
_CHUNK_MAX_BYTES = 1 << 20
# About this many bytes per scan task, so a failed scan stops after bounded work.
_SCAN_RANGE_MAX_BYTES = 1 << 26
# Sources smaller than this are scanned in the calling thread: starting threads would cost
# more than the scan (about 16 ms of one core at 64 MB/s).
_SCAN_INLINE_BYTES = 1 << 20
_BOM = b"\xef\xbb\xbf"


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


def tokenize(text: str, tokenizer: str = "fse_words@1") -> list[str]:
    """The tokens of one sentence by a tokenizer, as the native text sources split it."""
    return _native.tokenize(text, tokenizer)


def make_lookup(
    word_vectors: WordVectors, tokenizer: str, *, max_ngrams: int | None = None
) -> _native.TextLookup:
    """The lookup of stage 2 for word vectors: their vocabulary and FastText buckets.

    FastText vectors need ``max_ngrams``, the number of ngrams an unknown word uses at
    most (1 to ``MAX_NGRAMS_LIMIT``); other vectors take none.
    """
    if not isinstance(word_vectors, WordVectors):
        raise TypeError(f"word_vectors must be WordVectors, not {type(word_vectors).__name__}")
    tokenizer = check_tokenizer(tokenizer)
    table = _native.VocabTable(word_vectors.vocab_bytes, word_vectors.vocab_offsets)
    fasttext = word_vectors.fasttext
    if fasttext is None:
        if max_ngrams is not None:
            raise ValueError("max_ngrams applies only to FastText vectors")
        return _native.TextLookup(table, tokenizer)
    count = _integer(max_ngrams)
    if count is None:
        raise TypeError("FastText vectors need max_ngrams, an int")
    max_ngrams = count
    if not 1 <= max_ngrams <= _native.MAX_NGRAMS_LIMIT:
        raise ValueError(
            f"max_ngrams must be between 1 and {_native.MAX_NGRAMS_LIMIT}, not {max_ngrams}"
        )
    params = (fasttext.min_n, fasttext.max_n, fasttext.bucket)
    return _native.TextLookup(table, tokenizer, params, max_ngrams)


MAX_CHUNK_ROWS = _native.MAX_CHUNK_ROWS
MAX_NGRAMS_LIMIT = _native.MAX_NGRAMS_LIMIT


def new_workspace() -> _native.Workspace:
    """The buffers one worker thread reuses for every chunk it encodes."""
    return _native.Workspace()


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


def _append(chunks: list[Chunk], chunk: Chunk, *, merge: bool, chunk_rows: int):
    """Append ``chunk``; the first chunk of a scan range takes in the last chunk of the
    range before, a remainder that ended at the range boundary, if their rows fit."""
    last = chunks[-1] if merge and chunks else None
    if last is not None and last.n_sentences + chunk.n_sentences <= chunk_rows:
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
        total = len(self._data)
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
        total = len(self._data)
        workers, chunk_bytes, chunk_rows = _chunk_sizes(total, workers, chunk_bytes, chunk_rows)
        data = self._array()
        if self._positions() == 0:
            return Scan(0, ())
        ranges = list(pairwise(self._range_bounds(workers, chunk_bytes)))

        def scan(span: tuple[int, int]):
            return _native.scan_spans(
                data, self._offsets, span[0], span[1], chunk_bytes, chunk_rows
            )

        sentences = 0
        chunks: list[Chunk] = []
        with _scan_pool(workers, total) as pool:
            for result in (map if pool is None else pool.map)(scan, ranges):
                count, _, _, range_chunks, error_byte, error_sentence = result
                if error_byte >= 0:
                    raise self._error(error_byte, sentences + error_sentence)
                firsts = [row for _, _, row in range_chunks] + [count]
                for k, ((start, end, row), following) in enumerate(
                    zip(range_chunks, firsts[1:], strict=True)
                ):
                    chunk = Chunk(start, end, sentences + row, following - row)
                    _append(chunks, chunk, merge=k == 0, chunk_rows=chunk_rows)
                sentences += count
        return Scan(sentences, tuple(chunks))

    def _error(self, byte: int, sentence: int) -> UnicodeDecodeError:
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


def _check_container(texts: object, what: str) -> None:
    if isinstance(texts, set | frozenset):
        raise TypeError(f"{what} must be ordered, not a set; pass sorted(...) instead")
    if not isinstance(texts, list | tuple | ndarray):
        raise TypeError(f"{what} must be a list, tuple or numpy array, not {type(texts).__name__}")


def encode_string(text: object, what: str, row: int) -> bytes:
    """The UTF-8 bytes of sentence ``row``; a non-str or a lone surrogate raises."""
    if not isinstance(text, str):
        raise TypeError(f"{what} {row} is a {type(text).__name__}, not a str")
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as err:
        raise UnicodeEncodeError(
            err.encoding, err.object, err.start, err.end, f"{err.reason} in {what} {row}"
        ) from None


def _join(encoded: list[bytes]) -> tuple[bytes, ndarray]:
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    return b"".join(encoded), offsets


def _encode_strings(texts: Any, what: str) -> tuple[bytes, ndarray]:
    return _join([encode_string(text, what, i) for i, text in enumerate(texts)])


class StringSource(_Buffered):
    """Sentences as strings in memory, one row each."""

    def __init__(self, texts: Any):
        _check_container(texts, "texts")
        self._data, self._offsets = _encode_strings(texts, "string")

    @classmethod
    def encoded(cls, sentences: list[bytes]) -> StringSource:
        """A source of sentences already encoded (and checked) as UTF-8."""
        source = cls.__new__(cls)
        source._data, source._offsets = _join(sentences)
        return source

    def __len__(self) -> int:
        return self._positions()


class TokenListSource(_Buffered):
    """Sentences given as lists of tokens, for the external tokenizer only.

    Tokens must be non-empty and hold no whitespace, so splitting the joined sentence
    at whitespace gives them back exactly.
    """

    def __init__(self, sentences: Any):
        _check_container(sentences, "sentences")
        joined = []
        for i, tokens in enumerate(sentences):
            if not isinstance(tokens, list | tuple):
                raise TypeError(f"sentence {i} is a {type(tokens).__name__}, not a list of str")
            for token in tokens:
                if not isinstance(token, str):
                    raise TypeError(f"sentence {i} holds a {type(token).__name__}, not a str")
                if not token or token.split() != [token]:
                    raise ValueError(
                        f"sentence {i} holds the token {token!r}: tokens are non-empty and hold no whitespace"
                    )
            joined.append(" ".join(tokens))
        self._data, self._offsets = _encode_strings(joined, "sentence")

    def __len__(self) -> int:
        return self._positions()

    def encode(
        self, chunk: Chunk, lookup: _native.TextLookup, workspace: _native.Workspace
    ) -> Batch:
        if lookup.tokenizer != _native.TOKENIZER_EXTERNAL:
            raise ValueError(f"token lists are for the {EXTERNAL_TOKENIZER} tokenizer")
        return super().encode(chunk, lookup, workspace)


def _fingerprint(stat: os.stat_result) -> tuple[int, int, int, int]:
    """Identity and version of a file: device, inode, size and modification time."""
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


class FileSource(_Buffered):
    """The lines of a local UTF-8 text file, one sentence per line, memory-mapped.

    A line ends at "\\n", which is not part of it; the last line needs none. The file must
    not change while the source is open: a change of its device, inode, size or
    modification time raises at the next encode, and truncating a mapped file can end the
    process with SIGBUS, as for any memory map. Use it as a context manager, or close it.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"{self.path} is not a regular file")
        if self.path.suffix.lower() in _COMPRESSED_SUFFIXES:
            raise ValueError(f"{self.path} looks compressed; FileSource reads plain UTF-8 text")
        self._offsets = None
        self._closed = False
        self._mapped: mmap.mmap | None = None
        with open(self.path, "rb") as file:
            self._stat = _fingerprint(os.fstat(file.fileno()))
            if self._stat[2]:
                self._mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self._data = memoryview(self._mapped) if self._mapped is not None else b""
        if bytes(self._data[:3]) == _BOM:
            self.close()
            raise ValueError(f"{self.path} starts with a UTF-8 byte order mark; remove it")

    def _check_open(self) -> None:
        """Raise if the source is closed, or the file differs from the mapped one."""
        if self._closed:
            raise ValueError(f"the FileSource of {self.path} is closed")
        self._check_unchanged()

    def __len__(self) -> int:
        data = self._array()
        if not len(data):
            return 0
        count = sum(
            int(np.count_nonzero(data[i : i + (1 << 26)] == 10))
            for i in range(0, len(data), 1 << 26)
        )
        self._check_unchanged()
        return count + (0 if data[-1] == 10 else 1)

    def _check_unchanged(self) -> None:
        try:
            current = _fingerprint(os.stat(self.path))
        except OSError:
            current = None
        if current != self._stat:
            raise RuntimeError(f"{self.path} changed while it was read")

    def encode(
        self, chunk: Chunk, lookup: _native.TextLookup, workspace: _native.Workspace
    ) -> Batch:
        self._check_open()
        batch = super().encode(chunk, lookup, workspace)
        self._check_unchanged()  # a change while the chunk was encoded
        return batch

    def scan(
        self, *, workers: int = 1, chunk_bytes: int | None = None, chunk_rows: int | None = None
    ) -> Scan:
        result = super().scan(workers=workers, chunk_bytes=chunk_bytes, chunk_rows=chunk_rows)
        self._check_unchanged()  # a change during the scan
        return result

    def close(self) -> None:
        """Release the memory map."""
        self._closed = True
        if self._mapped is not None:
            try:
                if isinstance(self._data, memoryview):
                    self._data.release()
                self._mapped.close()
            except BufferError:
                pass  # arrays still view the map; it is released with the last of them
            self._mapped = None
            self._data = b""

    def __enter__(self) -> FileSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
