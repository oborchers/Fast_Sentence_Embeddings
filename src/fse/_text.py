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
    if isinstance(max_ngrams, bool) or not isinstance(max_ngrams, int):
        raise TypeError("FastText vectors need max_ngrams, an int")
    if not 1 <= max_ngrams <= _native.MAX_NGRAMS_LIMIT:
        raise ValueError(
            f"max_ngrams must be between 1 and {_native.MAX_NGRAMS_LIMIT}, not {max_ngrams}"
        )
    params = (fasttext.min_n, fasttext.max_n, fasttext.bucket)
    return _native.TextLookup(table, tokenizer, params, max_ngrams)


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


class _Buffered:
    """A source whose sentences are one UTF-8 buffer, with offsets for strings."""

    _data: Any
    _offsets: ndarray | None

    def _positions(self) -> int:
        """Bytes in line mode, strings otherwise."""
        return len(self._data) if self._offsets is None else len(self._offsets) - 1

    def _range_bounds(self, workers: int) -> list[int]:
        """Scan ranges of similar byte size that start at sentence starts."""
        size = self._positions()
        total = len(self._data)
        n_ranges = max(1, min(max(4 * workers, -(-total // _SCAN_RANGE_MAX_BYTES)), max(size, 1)))
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

    def scan(self, *, workers: int = 1, chunk_bytes: int | None = None) -> Scan:
        """Validate the UTF-8 of every sentence and cut the source into chunks."""
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError(f"workers must be a positive int, not {workers!r}")
        total = len(self._data)
        if chunk_bytes is None:
            chunk_bytes = max(_CHUNK_MIN_BYTES, min(_CHUNK_MAX_BYTES, total // (8 * workers)))
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
            raise ValueError(f"chunk_bytes must be a positive int, not {chunk_bytes!r}")
        data = self._array()
        if self._positions() == 0:
            return Scan(0, ())
        ranges = list(pairwise(self._range_bounds(workers)))

        def scan(span: tuple[int, int]):
            return _native.scan_spans(data, self._offsets, span[0], span[1], chunk_bytes)

        sentences = 0
        chunks: list[Chunk] = []
        with _scan_pool(workers, total) as pool:
            for result in (map if pool is None else pool.map)(scan, ranges):
                count, _, _, range_chunks, error_byte, error_sentence = result
                if error_byte >= 0:
                    raise self._error(error_byte, sentences + error_sentence)
                firsts = [row for _, _, row in range_chunks] + [count]
                for (start, end, row), following in zip(range_chunks, firsts[1:], strict=True):
                    chunks.append(Chunk(start, end, sentences + row, following - row))
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


def _encode_strings(texts: Any, what: str) -> tuple[bytes, ndarray]:
    encoded = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"{what} {i} is a {type(text).__name__}, not a str")
        try:
            encoded.append(text.encode("utf-8"))
        except UnicodeEncodeError as err:
            raise UnicodeEncodeError(
                err.encoding, err.object, err.start, err.end, f"{err.reason} in {what} {i}"
            ) from None
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    return b"".join(encoded), offsets


class StringSource(_Buffered):
    """Sentences as strings in memory, one row each."""

    def __init__(self, texts: Any):
        _check_container(texts, "texts")
        self._data, self._offsets = _encode_strings(texts, "string")

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

    def scan(self, *, workers: int = 1, chunk_bytes: int | None = None) -> Scan:
        result = super().scan(workers=workers, chunk_bytes=chunk_bytes)
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
