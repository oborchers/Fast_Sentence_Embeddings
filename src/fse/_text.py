"""Text sources: sentences of text to batches, stages 1 and 2 (ARCHITECTURE.md, "Tokenizers").

A source holds its sentences as one UTF-8 buffer: strings in memory with their offsets
(``StringSource``, ``TokenLists``), or the lines of a memory-mapped file
(``fse._text_file.FileSource``). ``scan`` validates and counts them in threads without the GIL and cuts
them into chunks; ``encode`` turns one chunk into a Batch with a ``TextLookup`` and a
per-worker ``Workspace``. Every sentence is one row, empty ones included.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy import ndarray

from fse import _native
from fse._contract import EXTERNAL_TOKENIZER, Batch, check_tokenizer
from fse._cpus import usable_cpus
from fse._wordvectors import WordVectors

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


def tokenize(text: str, tokenizer: str = "fse_words@1") -> list[str]:
    """The tokens of one sentence by a tokenizer, as the native text sources split it."""
    return _native.tokenize(text, tokenizer)


@dataclass(frozen=True, eq=False)
class Vocabulary:
    """The VocabTable of one vocabulary, built once and shared by every lookup of it."""

    table: _native.VocabTable
    fingerprint: str


def _check_word_vectors(word_vectors: object) -> WordVectors:
    if not isinstance(word_vectors, WordVectors):
        raise TypeError(f"word_vectors must be WordVectors, not {type(word_vectors).__name__}")
    return word_vectors


def make_vocabulary(word_vectors: WordVectors) -> Vocabulary:
    """The VocabTable of the words of ``word_vectors``, for ``make_lookup``."""
    wv = _check_word_vectors(word_vectors)
    data, offsets, fingerprint = wv._vocabulary_snapshot()
    return Vocabulary(_native.VocabTable(data, offsets), fingerprint)


def make_lookup(
    word_vectors: WordVectors,
    tokenizer: str,
    *,
    max_ngrams: int | None = None,
    vocabulary: Vocabulary | None = None,
) -> _native.TextLookup:
    """The lookup of stage 2 for word vectors: their vocabulary and FastText buckets.

    FastText vectors need ``max_ngrams``, the number of ngrams an unknown word uses at
    most (1 to ``MAX_NGRAMS_LIMIT``); other vectors take none. ``vocabulary`` reuses the
    table of ``make_vocabulary``; it must be one of the same words in the same order.
    """
    _check_word_vectors(word_vectors)
    tokenizer = check_tokenizer(tokenizer)
    if vocabulary is None:
        vocabulary = make_vocabulary(word_vectors)
    elif not isinstance(vocabulary, Vocabulary):
        raise TypeError(f"vocabulary must be a Vocabulary, not {type(vocabulary).__name__}")
    elif vocabulary.fingerprint != word_vectors.vocabulary_fingerprint:
        raise ValueError(
            f"the vocabulary {vocabulary.fingerprint} is not the one of the word vectors "
            f"({word_vectors.vocabulary_fingerprint})"
        )
    table = vocabulary.table
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
TextLookup = _native.TextLookup
Workspace = _native.Workspace


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

        def remainder(c: Chunk) -> bool:
            size = (
                c.end - c.start
                if self._offsets is None
                else self._offsets[c.end] - self._offsets[c.start]
            )
            return size < chunk_bytes and c.n_sentences < chunk_rows

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
                    _append(chunks, chunk, merge=k == 0, remainder=remainder, chunk_rows=chunk_rows)
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


# Threads that join token lists: the pass is bound by reading memory, which a few threads
# saturate (85.7 million WikiText tokens on an M1 Pro, in a fresh process: 2.1 s with one
# thread, 0.65 s with 8).
_JOIN_THREADS = 8


def _token_lists(sentences: Any) -> tuple[bytes, ndarray]:
    """The joined token lists in Python, for the inputs the native pass does not take and
    for the errors, which it leaves to this path (types first, then tokens, then the UTF-8
    of every sentence)."""
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
    return _encode_strings(joined, "sentence")


class TokenLists(_Buffered):
    """Sentences given as lists of tokens, read once, for encoders with the external
    tokenizer.

    ``fit``, ``fit_transform`` and ``encode`` take lists of token lists as they are, but
    each call reads them again. ``TokenLists(sentences)`` reads them once, so one object
    can be fitted on and encoded any number of times::

        texts = fse.TokenLists(sentences)
        encoder = fse.SIF(tokenizer="external").fit(wv, texts)
        rows = encoder.encode(texts)

    ``sentences`` is a list, tuple or numpy array of lists or tuples of str. Tokens must
    be non-empty and hold no whitespace (``str.isspace``), so splitting the joined
    sentence at whitespace gives them back exactly. Later changes to ``sentences`` do
    not change the object.
    """

    def __init__(self, sentences: Any):
        _check_container(sentences, "sentences")
        joined = _native.join_token_lists(sentences, min(_JOIN_THREADS, usable_cpus()))
        # None: types the native pass does not read (subclasses, an array); False: an
        # invalid token. The Python path decides both: it raises the error, or succeeds
        # where a subclass changed the lists while they were read.
        if joined is None or joined is False:
            joined = _token_lists(sentences)
        self._data, self._offsets = joined

    def __len__(self) -> int:
        return self._positions()

    def encode(
        self, chunk: Chunk, lookup: _native.TextLookup, workspace: _native.Workspace
    ) -> Batch:
        if lookup.tokenizer != _native.TOKENIZER_EXTERNAL:
            raise ValueError(f"token lists are for the {EXTERNAL_TOKENIZER} tokenizer")
        return super().encode(chunk, lookup, workspace)
