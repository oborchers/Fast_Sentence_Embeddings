#!/usr/bin/env cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Parts adapted from gensim (LGPL 2.1, used under the GPL 3); see NOTICE.

"""Native routines of fse 2.0: FastText ngram hashing, the vocabulary table and the
UTF-8 scan of text, all without the GIL. The weighted mean kernel follows."""

import cython
import numpy as np

cimport numpy as np

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pyport cimport PY_SSIZE_T_MAX
from libc.stdint cimport SIZE_MAX
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memchr, memcmp, memcpy, memset

np.import_array()

cdef unsigned char _MB_MASK = 0xC0
cdef unsigned char _MB_START = 0x80


cdef inline uINT_t _ft_hash(const unsigned char *bytez, size_t start, size_t stop) noexcept nogil:
    """FNV-1a over bytez[start:stop], with each byte sign-extended as in fastText."""
    cdef uINT_t h = 2166136261
    cdef size_t k
    for k in range(start, stop):
        h = h ^ <uINT_t>(<signed char>bytez[k])
        h = h * 16777619
    return h


cdef size_t ft_ngram_hashes_into(
    const unsigned char *bytez,
    size_t num_bytes,
    unsigned int min_n,
    unsigned int max_n,
    uINT_t bucket,
    uINT_t *out,
    size_t max_out,
) noexcept nogil:
    """Write the bucket of the first max_out ngrams of the UTF-8 word "<word>" into out.

    Returns the number written, min(number of ngrams, max_out). A bucket of 0 means the
    model has no ngram vectors, so no ngram is written.
    """
    cdef size_t i, j, n
    cdef size_t written = 0
    if bucket == 0:
        return 0
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue
        j = i
        n = 1
        while j < num_bytes and n <= max_n:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                if written == max_out:
                    return written
                out[written] = _ft_hash(bytez, i, j) % bucket
                written += 1
            n += 1
    return written


def ft_ngram_hashes(word, unsigned int min_n, unsigned int max_n, uINT_t bucket, size_t max_ngrams):
    """Return the ngram buckets of word as gensim's ft_ngram_hashes(...)[:max_ngrams] does."""
    cdef bytes utf8_word = ("<%s>" % word).encode("utf-8")
    out = np.empty(max(len(utf8_word) * max(max_n, 1), 1), dtype=np.uint32)
    cdef uINT_t[::1] view = out
    cdef size_t n = ft_ngram_hashes_into(
        utf8_word, len(utf8_word), min_n, max_n, bucket, &view[0], min(max_ngrams, view.shape[0])
    )
    return out[:n].tolist()




# A token of a FastText job that is not in the vocabulary. Its UTF-8 bytes "<token>" are
# stored with the job and hashed into ngram buckets by the worker, without the GIL.
cdef uINT_t OOV_TOKEN = <uINT_t>0xFFFFFFFE


# More ngrams than any word of a real corpus has; a bound that keeps buffer sizes in range.
MAX_NGRAMS_LIMIT = 1 << 24

# Sentences of a native chunk at most, besides its byte size (see scan_spans).
cdef Py_ssize_t CHUNK_MAX_ROWS = 1 << 16
MAX_CHUNK_ROWS = CHUNK_MAX_ROWS


cdef struct _Buffer:
    char *data
    Py_ssize_t size  # bytes used
    Py_ssize_t capacity  # bytes allocated


cdef int _reserve(_Buffer *b, Py_ssize_t extra) except -1:
    """Grow b so that extra more bytes fit, doubling the capacity."""
    cdef Py_ssize_t needed
    cdef Py_ssize_t capacity
    cdef char *data
    if extra > PY_SSIZE_T_MAX - b.size:
        raise MemoryError("the job buffer would exceed the address space")
    needed = b.size + extra
    if needed <= b.capacity:
        return 0
    capacity = b.capacity * 2 if b.capacity <= PY_SSIZE_T_MAX // 2 else PY_SSIZE_T_MAX
    capacity = max(needed, capacity, 4096)
    data = <char *>realloc(b.data, capacity)
    if data == NULL:
        raise MemoryError("could not grow the job buffer")
    b.data = data
    b.capacity = capacity
    return 0


cdef inline int _push_u32(_Buffer *b, uINT_t value) except -1:
    _reserve(b, sizeof(uINT_t))
    (<uINT_t *>(b.data + b.size))[0] = value
    b.size += sizeof(uINT_t)
    return 0


cdef inline int _push_i64(_Buffer *b, np.int64_t value) except -1:
    _reserve(b, sizeof(np.int64_t))
    (<np.int64_t *>(b.data + b.size))[0] = value
    b.size += sizeof(np.int64_t)
    return 0


cdef inline int _push_bytes(_Buffer *b, const char *data, Py_ssize_t n) except -1:
    if n == 0:
        return 0  # b.data may still be NULL, and memcpy needs valid pointers
    _reserve(b, n)
    memcpy(b.data + b.size, data, n)
    b.size += n
    return 0


cdef object _to_array(_Buffer *b, dtype):
    """Copy the used part of b into a new numpy array of dtype."""
    out = np.empty(b.size // np.dtype(dtype).itemsize, dtype=dtype)
    if b.size:
        memcpy(np.PyArray_DATA(out), b.data, b.size)
    return out



cdef uINT_t NOT_FOUND = <uINT_t>0xFFFFFFFF

# Return codes of the native routines, translated into exceptions with the GIL.
cdef enum:
    NATIVE_OK = 0
    NATIVE_NO_MEMORY = -1
    NATIVE_BAD_UTF8 = -2
    NATIVE_BAD_ROW = -3


cdef inline int _utf8_len(const unsigned char *p, Py_ssize_t i, Py_ssize_t end) noexcept nogil:
    """Length of the valid UTF-8 sequence at p[i], or 0 where CPython's decoder fails."""
    cdef unsigned char b = p[i]
    cdef unsigned char c
    if b < 0x80:
        return 1
    if b < 0xC2:
        return 0
    if b < 0xE0:
        if end - i >= 2 and (p[i + 1] & 0xC0) == 0x80:
            return 2
        return 0
    if b < 0xF0:
        if end - i < 3:
            return 0
        c = p[i + 1]
        if (b == 0xE0 and c < 0xA0) or (b == 0xED and c > 0x9F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80:
            return 0
        return 3
    if b < 0xF5:
        if end - i < 4:
            return 0
        c = p[i + 1]
        if (b == 0xF0 and c < 0x90) or (b == 0xF4 and c > 0x8F):
            return 0
        if (c & 0xC0) != 0x80 or (p[i + 2] & 0xC0) != 0x80 or (p[i + 3] & 0xC0) != 0x80:
            return 0
        return 4
    return 0


cdef inline bint _is_space(const unsigned char *p, Py_ssize_t i, int n) noexcept nogil:
    """Whether the valid UTF-8 sequence of length n at p[i] satisfies str.isspace()."""
    cdef unsigned char b = p[i]
    cdef unsigned char c, d
    if n == 1:
        return (9 <= b <= 13) or (28 <= b <= 32)
    if n == 2:
        return b == 0xC2 and (p[i + 1] == 0x85 or p[i + 1] == 0xA0)
    if n == 3:
        c = p[i + 1]
        d = p[i + 2]
        if b == 0xE1:
            return c == 0x9A and d == 0x80  # U+1680
        if b == 0xE2:
            if c == 0x80:  # U+2000 to U+200A, U+2028, U+2029, U+202F
                return d <= 0x8A or d == 0xA8 or d == 0xA9 or d == 0xAF
            return c == 0x81 and d == 0x9F  # U+205F
        if b == 0xE3:
            return c == 0x80 and d == 0x80  # U+3000
    return False


# Words of at most this many UTF-8 bytes are stored in their slot, so looking one up
# usually reads a single cache line; longer words are compared in the arena.
cdef enum:
    INLINE_KEY_BYTES = 16


cdef inline np.uint64_t _load_partial(const unsigned char *p, Py_ssize_t k) noexcept nogil:
    """The k <= 8 bytes at p as a little-endian word, zero above them."""
    cdef np.uint64_t w = 0
    cdef Py_ssize_t i
    if k == 8:
        memcpy(&w, p, 8)
        return w
    for i in range(k):
        w |= (<np.uint64_t>p[i]) << (8 * i)
    return w


cdef inline np.uint64_t _mix(np.uint64_t h, np.uint64_t w) noexcept nogil:
    h = (h ^ w) * 0xBF58476D1CE4E5B9ULL
    return h ^ (h >> 29)


cdef inline np.uint64_t _load_masked(const unsigned char *p, Py_ssize_t k) noexcept nogil:
    """The k <= 8 bytes at p as a word, zero above them; reads 8 bytes."""
    cdef np.uint64_t w
    memcpy(&w, p, 8)
    return w if k >= 8 else w & ((1ULL << (8 * k)) - 1)


cdef inline np.uint64_t _hash_word(
    const unsigned char *p, Py_ssize_t n, np.uint64_t *key, bint padded
) noexcept nogil:
    """The slot hash of a word, 8 bytes per step. For a word of at most INLINE_KEY_BYTES
    bytes, key receives its bytes as two zero-padded words, as a slot stores them. With
    padded, 16 bytes from p are readable, so the words are loaded whole and masked.

    Only VocabTable uses it, for building and looking up, so it is no persistence format.
    """
    cdef np.uint64_t h = 0x9E3779B97F4A7C15ULL ^ <np.uint64_t>n
    cdef Py_ssize_t left = n
    if n <= INLINE_KEY_BYTES:
        if padded:
            key[0] = _load_masked(p, n)
            key[1] = _load_masked(p + 8, n - 8) if n > 8 else 0
        else:
            key[0] = _load_partial(p, n if n < 8 else 8)
            key[1] = _load_partial(p + 8, n - 8) if n > 8 else 0
        h = _mix(_mix(h, key[0]), key[1])
    else:
        while left >= 8:
            h = _mix(h, _load_partial(p, 8))
            p += 8
            left -= 8
        h = _mix(h, _load_partial(p, left))
    h *= 0x94D049BB133111EBULL
    return h ^ (h >> 32)


# One slot of the open-addressing table, 32 bytes. value is NOT_FOUND for a free slot.
cdef struct VocabSlot:
    np.uint64_t hash
    uINT_t value
    np.uint32_t length
    # The word itself, zero-padded, if length <= INLINE_KEY_BYTES, else its offset in the
    # arena (int64).
    np.uint64_t key[2]


cdef struct VocabView:
    const VocabSlot *slots
    const unsigned char *arena
    np.uint64_t mask


cdef inline uINT_t _vocab_lookup(
    const VocabView *v, const unsigned char *p, Py_ssize_t n, bint padded=False
) noexcept nogil:
    """Vocabulary index of the n bytes at p, or NOT_FOUND; padded as for _hash_word."""
    cdef np.uint64_t key[2]
    cdef np.uint64_t h = _hash_word(p, n, key, padded)
    cdef np.uint64_t j = h & v.mask
    cdef const VocabSlot *slot
    while True:
        slot = &v.slots[j]
        if slot.value == NOT_FOUND:
            return NOT_FOUND
        if slot.hash == h and slot.length == n:
            if n <= INLINE_KEY_BYTES:
                if slot.key[0] == key[0] and slot.key[1] == key[1]:
                    return slot.value
            elif memcmp(v.arena + <np.int64_t>slot.key[0], p, n) == 0:
                return slot.value
        j = (j + 1) & v.mask


cdef inline np.int64_t _vocab_insert(
    VocabSlot *slots, np.uint64_t mask, const unsigned char *arena, np.int64_t offset,
    Py_ssize_t n, uINT_t index,
) noexcept nogil:
    """Insert the n bytes at arena[offset] as word index; returns -1, or the index of the
    word already stored with the same bytes."""
    cdef const unsigned char *p = arena + offset
    cdef np.uint64_t key[2]
    cdef np.uint64_t h = _hash_word(p, n, key, False)
    cdef np.uint64_t j = h & mask
    while slots[j].value != NOT_FOUND:
        if slots[j].hash == h and slots[j].length == n:
            if n <= INLINE_KEY_BYTES:
                if slots[j].key[0] == key[0] and slots[j].key[1] == key[1]:
                    return slots[j].value
            elif memcmp(arena + <np.int64_t>slots[j].key[0], p, n) == 0:
                return slots[j].value
        j = (j + 1) & mask
    slots[j].hash = h
    slots[j].value = index
    slots[j].length = <np.uint32_t>n
    if n <= INLINE_KEY_BYTES:
        slots[j].key[0] = key[0]
        slots[j].key[1] = key[1]
    else:
        slots[j].key[0] = <np.uint64_t>offset
        slots[j].key[1] = 0
    return -1


cdef class VocabTable:
    """Immutable map from the UTF-8 bytes of a word to its vocabulary index.

    Built without the GIL from the vocabulary of word vectors: ``vocab_bytes`` (uint8)
    holds the words one after another, word i is ``vocab_bytes[vocab_offsets[i]:
    vocab_offsets[i + 1]]``. Open addressing with linear probing, at most half full. Each
    32-byte slot holds the hash, the index and, for words of up to INLINE_KEY_BYTES bytes,
    the word itself, so a lookup usually reads one cache line; longer words are compared
    in vocab_bytes, which the table keeps (it may be memory-mapped).
    """

    cdef object _bytes
    cdef object _slots
    cdef Py_ssize_t n_words
    cdef VocabView view

    def __init__(self, vocab_bytes, vocab_offsets):
        if (
            not isinstance(vocab_bytes, np.ndarray) or vocab_bytes.dtype != np.uint8
            or vocab_bytes.ndim != 1 or not vocab_bytes.flags.c_contiguous
        ):
            raise ValueError("vocab_bytes must be a C-contiguous uint8 vector")
        if (
            not isinstance(vocab_offsets, np.ndarray) or vocab_offsets.dtype != np.int64
            or vocab_offsets.ndim != 1 or not vocab_offsets.flags.c_contiguous
        ):
            raise ValueError("vocab_offsets must be a C-contiguous int64 vector")
        if (
            len(vocab_offsets) == 0 or vocab_offsets[0] != 0
            or vocab_offsets[len(vocab_offsets) - 1] != len(vocab_bytes)
        ):
            raise ValueError(f"vocab_offsets must run from 0 to {len(vocab_bytes)}, the number of bytes")
        # Own copies, so arrays the caller still holds cannot change a checked table; a
        # read-only vocab_bytes (word vectors, a memory map) is kept as it is.
        vocab_offsets = vocab_offsets.copy()
        if vocab_bytes.flags.writeable:
            vocab_bytes = vocab_bytes.copy()
        lengths = vocab_offsets[1:] - vocab_offsets[: len(vocab_offsets) - 1]
        if (lengths < 0).any():
            raise ValueError("vocab_offsets must not decrease")
        if len(lengths) and int(lengths.max()) >= 2**32:
            raise ValueError(f"word {int(lengths.argmax())} has 2**32 or more bytes")
        cdef Py_ssize_t n = len(vocab_offsets) - 1
        # Ids from OOV_TOKEN on are reserved.
        if n >= <Py_ssize_t>OOV_TOKEN:
            raise ValueError(f"a vocabulary holds fewer than {OOV_TOKEN} words, not {n}")
        cdef np.uint64_t size = 2
        while size < 2 * <np.uint64_t>n:
            size <<= 1
        # Raw bytes, so the slot layout is the C struct's, whatever numpy would pad.
        self._slots = np.empty(size * sizeof(VocabSlot), dtype=np.uint8)
        self._bytes = vocab_bytes
        self.n_words = n
        cdef VocabSlot *slots = <VocabSlot *>np.PyArray_DATA(self._slots)
        cdef const unsigned char *arena = <const unsigned char *>np.PyArray_DATA(vocab_bytes)
        cdef const np.int64_t[::1] offsets = vocab_offsets
        cdef np.uint64_t mask = size - 1
        cdef np.uint64_t j
        cdef Py_ssize_t i
        cdef np.int64_t dup = -1
        with nogil:
            memset(slots, 0, size * sizeof(VocabSlot))
            for j in range(size):
                slots[j].value = NOT_FOUND
            for i in range(n):
                dup = _vocab_insert(slots, mask, arena, offsets[i], offsets[i + 1] - offsets[i], <uINT_t>i)
                if dup >= 0:
                    break
        if dup >= 0:
            raise ValueError(f"word {i} is also word {dup} of the vocabulary")
        self.view.slots = slots
        self.view.arena = arena
        self.view.mask = mask

    def __reduce__(self):
        raise TypeError("a VocabTable cannot be pickled; build it from the word vectors")

    def __len__(self):
        return self.n_words

    def get(self, word):
        """Vocabulary index of word (str or bytes), or None; for tests."""
        encoded = word if isinstance(word, bytes) else (<str?>word).encode("utf-8")
        cdef uINT_t index = _vocab_lookup(&self.view, encoded, len(encoded))
        return None if index == NOT_FOUND else index


cdef Py_ssize_t _count_tokens(
    const unsigned char *p, Py_ssize_t start, Py_ssize_t end, Py_ssize_t *error
) noexcept nogil:
    """Number of str.split() tokens in p[start:end]; sets error to the first bad byte."""
    cdef Py_ssize_t i = start
    cdef Py_ssize_t tokens = 0
    cdef bint inside = False
    cdef int m
    while i < end:
        m = _utf8_len(p, i, end)
        if m == 0:
            error[0] = i
            return tokens
        if _is_space(p, i, m):
            inside = False
        elif not inside:
            inside = True
            tokens += 1
        i += m
    return tokens


def _check_span(const unsigned char[::1] data, offsets, Py_ssize_t first, Py_ssize_t last):
    """Raise unless [first, last) is a valid range of whole sentences of data.

    The native routines read through raw pointers without bounds checks, so every range
    and offset they get is checked here: lines must start at first and at last; offsets
    must cover the range, stay inside data and never decrease.
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef np.int64_t[::1] item_offsets
    cdef Py_ssize_t i
    if offsets is None:
        if not 0 <= first <= last <= n:
            raise ValueError(f"range [{first}, {last}) is outside the {n} bytes of the source")
        if first > 0 and data[first - 1] != 10:
            raise ValueError(f"range start {first} is not the start of a line")
        if 0 < last < n and data[last - 1] != 10:
            raise ValueError(f"range end {last} is not the start of a line")
        return
    item_offsets = offsets
    if not 0 <= first <= last <= item_offsets.shape[0] - 1:
        raise ValueError(f"range [{first}, {last}) is outside the {item_offsets.shape[0] - 1} strings")
    if item_offsets[first] < 0 or item_offsets[last] > n:
        raise ValueError("string offsets point outside the source")
    for i in range(first, last):
        if item_offsets[i + 1] < item_offsets[i]:
            raise ValueError(f"string offsets decrease at {i}")


def scan_spans(const unsigned char[::1] data, offsets, Py_ssize_t first, Py_ssize_t last,
               Py_ssize_t chunk_bytes):
    """Validate and count the sentences of one scan range, without the GIL.

    Line mode (offsets is None): the range holds the lines that start at byte positions
    in [first, last); the caller aligns first and last to line starts. Item mode: the
    range holds the strings first to last - 1 of a TextList, delimited by offsets.

    Returns (sentences, tokens, empty, chunks, error_byte, error_sentence): chunks is a
    list of (start, end, first_sentence) with positions in the same unit as the range and
    sentence numbers relative to the range; error_byte is -1 if the range is valid.
    """
    _check_span(data, offsets, first, last)
    cdef Py_ssize_t n = data.shape[0]
    cdef const unsigned char *p = &data[0] if n else NULL
    cdef bint lines = offsets is None
    cdef np.int64_t[::1] item_offsets
    cdef const np.int64_t *item_ptr = NULL
    if not lines:
        item_offsets = offsets
        item_ptr = &item_offsets[0]
    cdef Py_ssize_t sentences = 0, tokens = 0, empty = 0, count
    cdef Py_ssize_t error = -1
    cdef Py_ssize_t pos = first, span_end, next_pos, chunk_start = first, chunk_first = 0
    cdef const unsigned char *newline
    chunk_list = []
    cdef np.int64_t[::1] chunk_buffer
    cdef Py_ssize_t n_chunks = 0

    # Chunk boundaries are collected in C and converted to a list at the end. Every chunk
    # but the last holds at least chunk_bytes bytes or CHUNK_MAX_ROWS sentences, which
    # bounds their number. The row limit keeps a chunk of empty lines, which take no
    # bytes, from holding all rows, so training into windows needs only a few of them.
    if chunk_bytes < 1:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    cdef Py_ssize_t span_bytes = (last - first) if lines else (item_ptr[last] - item_ptr[first])
    # A line takes at least its "\n" (the last none), a string at least its entry.
    cdef Py_ssize_t max_rows = (span_bytes + 1) if lines else (last - first)
    cdef Py_ssize_t max_chunks = span_bytes // chunk_bytes + max_rows // CHUNK_MAX_ROWS + 2
    chunk_arr = np.empty(3 * max_chunks, dtype=np.int64)
    chunk_buffer = chunk_arr
    with nogil:
        while pos < last:
            if lines:
                newline = <const unsigned char *>memchr(p + pos, 10, n - pos)
                span_end = (newline - p) if newline != NULL else n
                next_pos = span_end + 1 if newline != NULL else n
            else:
                span_end = item_ptr[pos + 1]
                next_pos = pos + 1
            count = _count_tokens(p, item_ptr[pos] if not lines else pos, span_end, &error)
            if error >= 0:
                break
            tokens += count
            if count == 0:
                empty += 1
            sentences += 1
            pos = next_pos
            if (
                (lines and pos - chunk_start >= chunk_bytes)
                or (not lines and (item_ptr[pos] - item_ptr[chunk_start]) >= chunk_bytes)
                or sentences - chunk_first >= CHUNK_MAX_ROWS
            ) and pos < last and n_chunks + 1 < max_chunks:
                chunk_buffer[3 * n_chunks] = chunk_start
                chunk_buffer[3 * n_chunks + 1] = pos
                chunk_buffer[3 * n_chunks + 2] = chunk_first
                n_chunks += 1
                chunk_start = pos
                chunk_first = sentences
    if error >= 0:
        return sentences, tokens, empty, [], error, sentences
    if chunk_start < last:
        chunk_buffer[3 * n_chunks] = chunk_start
        chunk_buffer[3 * n_chunks + 1] = last
        chunk_buffer[3 * n_chunks + 2] = chunk_first
        n_chunks += 1
    chunk_list = [tuple(chunk_arr[3 * c : 3 * c + 3].tolist()) for c in range(n_chunks)]
    return sentences, tokens, empty, chunk_list, -1, -1


cdef int _push_u32_nogil(_Buffer *b, uINT_t value) noexcept nogil:
    if <Py_ssize_t>sizeof(uINT_t) > b.capacity - b.size:
        if _grow_nogil(b, sizeof(uINT_t)) != 0:
            return -1
    (<uINT_t *>(b.data + b.size))[0] = value
    b.size += sizeof(uINT_t)
    return 0


cdef int _push_i64_nogil(_Buffer *b, np.int64_t value) noexcept nogil:
    if <Py_ssize_t>sizeof(np.int64_t) > b.capacity - b.size:
        if _grow_nogil(b, sizeof(np.int64_t)) != 0:
            return -1
    (<np.int64_t *>(b.data + b.size))[0] = value
    b.size += sizeof(np.int64_t)
    return 0


cdef int _push_bytes_nogil(_Buffer *b, const unsigned char *data, Py_ssize_t n) noexcept nogil:
    if n == 0:
        return 0  # b.data may still be NULL, and memcpy needs valid pointers
    if n > b.capacity - b.size:
        if _grow_nogil(b, n) != 0:
            return -1
    memcpy(b.data + b.size, data, n)
    b.size += n
    return 0


cdef int _grow_nogil(_Buffer *b, Py_ssize_t extra) noexcept nogil:
    """Like _reserve, without the GIL: returns -1 instead of raising."""
    cdef Py_ssize_t capacity
    cdef char *data
    if extra > PY_SSIZE_T_MAX - b.size:
        return -1
    capacity = b.capacity * 2 if b.capacity <= PY_SSIZE_T_MAX // 2 else PY_SSIZE_T_MAX
    capacity = max(b.size + extra, capacity, 4096)
    data = <char *>realloc(b.data, capacity)
    if data == NULL:
        return -1
    b.data = data
    b.capacity = capacity
    return 0



# ---------------------------------------------------------------------------------------
# The weighted mean kernel (weighted_mean@1): stages 2 and 3 of a batch, without the GIL.
# ---------------------------------------------------------------------------------------

from libc.stdint cimport int64_t, uint32_t, uint64_t

cdef extern from "_native_kernel.h":
    ctypedef struct fse_params:
        int dtype
        int dim
        size_t row_bytes
        const void *rows
        const float *scales
        const float *weights
        const void *ngram_rows
        const float *ngram_scales
        float oov_weight
    ctypedef struct fse_batch:
        int64_t n_sentences
        const int64_t *offsets
        const uint32_t *tokens
        const int64_t *ngram_offsets
        const uint32_t *ngrams
    float fse_half_table[65536]
    int fse_has_isa(int isa) nogil
    int64_t fse_run(int isa, const fse_params *p, const fse_batch *b, float *out,
                    int64_t *counts, int finish) nogil
    void fse_dot16_rows(int isa, const float *a, const float *b, int64_t n, int dim, float *out) nogil
    uint64_t fse_test_fpenv_set(int flush, int round_up)
    uint64_t fse_fpenv_push()
    void fse_fpenv_pop(uint64_t saved)
    void fse_test_fpenv_restore(uint64_t saved)
    uint64_t fse_test_fpenv_get()
    int64_t fse_remove_components(int isa, float *rows, int64_t n, int dim,
                                  const float *components, int k, const float *weights,
                                  float *scratch) nogil
    int64_t fse_project(int isa, const float *rows, int64_t n, int dim, const float *basis,
                        int k, float *out) nogil
    int64_t fse_normalize(int isa, float *rows, int64_t n, int dim, int64_t *unnormalizable) nogil
    int64_t fse_score(int isa, int dtype, const void *stored, const float *scales, int64_t n,
                      int dim, const float *w, float bias, float *out, float *scratch) nogil

_ISAS = {"portable": 0, "avx2": 1, "neon": 2}
_DTYPES = {np.dtype(np.float32): 0, np.dtype(np.float16): 1, np.dtype(np.int8): 2}

with np.errstate(invalid="ignore"):
    # The bit patterns include NaN payloads, whose conversion numpy reports.
    _half_values = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
for _i in range(65536):
    fse_half_table[_i] = _half_values[_i]
del _half_values, _i


def available_isas():
    """The instruction sets of the kernel this CPU runs, the fastest last."""
    return tuple(name for name, isa in _ISAS.items() if fse_has_isa(isa))


def default_isa():
    """The instruction set the kernel uses unless told otherwise."""
    isas = available_isas()
    return isas[len(isas) - 1]


# The kernel indexes a row with int, and its loops step up to 32 values past an index.
MAX_DIM = 1 << 30


def _check_dim(dim):
    if dim > MAX_DIM:
        raise ValueError(f"the kernel supports at most {MAX_DIM} dimensions, not {dim}")
    return dim


cdef class _KernelEnv:
    """The floating-point environment of the kernel for a whole call: the arithmetic and the
    numpy conversions of the wrapper (bias, oov_weight, input arrays) run in it."""

    cdef uint64_t saved

    def __enter__(self):
        self.saved = fse_fpenv_push()

    def __exit__(self, *exc):
        fse_fpenv_pop(self.saved)
        return False


def _test_fpenv_set(flush=False, round_up=False):
    """For the tests: set flush-to-zero and rounding up; returns the previous environment."""
    return fse_test_fpenv_set(1 if flush else 0, 1 if round_up else 0)


def _test_fpenv_restore(saved):
    """For the tests: restore an environment returned by _test_fpenv_set."""
    fse_test_fpenv_restore(saved)


def _test_fpenv_get():
    """For the tests: the floating-point control register (MXCSR or FPCR)."""
    return fse_test_fpenv_get()


def _check_isa(isa):
    if isa is None:
        return _ISAS[default_isa()]
    if isa not in _ISAS:
        raise ValueError(f"isa must be one of {sorted(_ISAS)}, not {isa!r}")
    if not fse_has_isa(_ISAS[isa]):
        raise ValueError(f"this CPU cannot run the {isa} kernel; available: {available_isas()}")
    return _ISAS[isa]


def _float32_vector(value, name, length):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != (length,):
        raise ValueError(f"{name} must be a float32 vector of {length} values")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _rows(value, name):
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype not in _DTYPES:
        raise ValueError(f"{name} must be a matrix of float32, float16 or int8 rows")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


cdef class WeightedMean:
    """The parameters of weighted_mean@1 on one set of word rows, checked once.

    ``vectors`` (and ``ngram_vectors``) are the stored rows as float32, float16 or int8
    codes, with ``scales`` (``ngram_scales``) for int8; ``weights`` one float32 weight per
    word; ``oov_weight`` the weight of unknown FastText words. ``pool`` and ``accumulate``
    give the bits of ``fse._reference``; several threads may call them at once.
    """

    cdef fse_params params
    cdef int isa
    cdef readonly int dim
    cdef readonly object isa_name
    cdef Py_ssize_t n_words
    cdef Py_ssize_t n_buckets
    cdef bint fasttext
    cdef object _keep

    def __init__(self, vectors, weights, scales=None, ngram_vectors=None, ngram_scales=None,
                 oov_weight=None, *, isa=None):
        self.isa = _check_isa(isa)
        self.isa_name = next(k for k, v in _ISAS.items() if v == self.isa)
        vectors = _rows(vectors, "vectors")
        self.n_words = vectors.shape[0]
        self.dim = _check_dim(vectors.shape[1])
        if self.dim < 1:
            raise ValueError("vectors need at least one dimension")
        int8 = vectors.dtype == np.int8
        weights = _float32_vector(weights, "weights", self.n_words)
        if int8 != (scales is not None):
            raise ValueError("scales are given exactly for int8 vectors")
        if int8:
            scales = _float32_vector(scales, "scales", self.n_words)
        self.fasttext = ngram_vectors is not None
        if self.fasttext:
            ngram_vectors = _rows(ngram_vectors, "ngram_vectors")
            if ngram_vectors.dtype != vectors.dtype or ngram_vectors.shape[1] != self.dim:
                raise ValueError("ngram_vectors must have the dtype and dimension of vectors")
            self.n_buckets = ngram_vectors.shape[0]
            if int8 != (ngram_scales is not None):
                raise ValueError("ngram_scales are given exactly for int8 ngram vectors")
            if int8:
                ngram_scales = _float32_vector(ngram_scales, "ngram_scales", self.n_buckets)
        elif ngram_scales is not None:
            raise ValueError("ngram_scales without ngram_vectors")
        self.params.dtype = _DTYPES[vectors.dtype]
        self.params.dim = self.dim
        self.params.row_bytes = <size_t>(self.dim * vectors.itemsize)
        self.params.rows = np.PyArray_DATA(vectors)
        self.params.scales = <const float *>np.PyArray_DATA(scales) if int8 else NULL
        self.params.weights = <const float *>np.PyArray_DATA(weights)
        self.params.ngram_rows = np.PyArray_DATA(ngram_vectors) if self.fasttext else NULL
        self.params.ngram_scales = (
            <const float *>np.PyArray_DATA(ngram_scales) if self.fasttext and int8 else NULL
        )
        # As the reference: float32(oov_weight); None only without unknown words.
        with _KernelEnv(), np.errstate(over="ignore", invalid="ignore"):  # as the reference
            self.params.oov_weight = np.float32(oov_weight) if oov_weight is not None else 0
        self._keep = (vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight)

    def _check_batch(self, batch):
        from fse._contract import OOV_TOKEN, Batch, _check_offsets

        if not isinstance(batch, Batch):
            raise TypeError(f"batch must be a Batch, not {type(batch).__name__}")
        tokens = batch.tokens
        # Batch checked its arrays when it was made; the kernel reads through raw pointers,
        # so the offsets are checked again here.
        _check_offsets(batch.offsets, len(tokens), "offsets")
        unknown = tokens == OOV_TOKEN
        if unknown.any():
            if batch.ngrams is None or batch.ngram_offsets is None:
                raise ValueError("an unknown FastText word needs ngram buckets")
            n_unknown = int(np.count_nonzero(unknown))
            if len(batch.ngram_offsets) != n_unknown + 1:
                raise ValueError(f"ngram_offsets must have {n_unknown + 1} entries for {n_unknown} unknown words")
            _check_offsets(batch.ngram_offsets, len(batch.ngrams), "ngram_offsets")
            if not self.fasttext:
                raise ValueError("an unknown FastText word needs ngram vectors")
            if self._keep[5] is None:
                raise ValueError("an unknown FastText word needs oov_weight")
            if len(batch.ngrams) and int(batch.ngrams.max()) >= self.n_buckets:
                raise ValueError(f"ngram buckets must be below {self.n_buckets}")
        known = tokens[~unknown]
        if len(known) and int(known.max()) >= self.n_words:
            raise ValueError(f"token ids must be below {self.n_words}, the number of words")

    cdef int64_t _run(self, batch, float[:, ::1] out, int64_t *counts, int finish) except? -2:
        cdef fse_batch b
        cdef const np.int64_t[::1] offsets = batch.offsets
        cdef const np.uint32_t[::1] tokens = batch.tokens
        cdef const np.int64_t[::1] ngram_offsets
        cdef const np.uint32_t[::1] ngrams
        b.n_sentences = offsets.shape[0] - 1
        b.offsets = <const int64_t *>&offsets[0]
        b.tokens = <const uint32_t *>&tokens[0] if tokens.shape[0] else NULL
        b.ngram_offsets = NULL
        b.ngrams = NULL
        if batch.ngram_offsets is not None:
            ngram_offsets = batch.ngram_offsets
            ngrams = batch.ngrams
            b.ngram_offsets = <const int64_t *>&ngram_offsets[0]
            b.ngrams = <const uint32_t *>&ngrams[0] if ngrams.shape[0] else NULL
        cdef float *target = &out[0, 0] if out.shape[0] else NULL
        cdef int64_t bad
        if b.n_sentences == 0:
            return -1
        with nogil:
            bad = fse_run(self.isa, &self.params, &b, target, counts, finish)
        return bad

    def _out(self, batch, out):
        n = batch.n_sentences
        if out is None:
            return np.empty((n, self.dim), dtype=np.float32)
        if (
            not isinstance(out, np.ndarray)
            or out.dtype != np.float32
            or out.shape != (n, self.dim)
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError(f"out must be a writable C-contiguous float32 array of shape {(n, self.dim)}")
        for kept in self._keep[:5]:
            if kept is not None and np.may_share_memory(out, kept):
                raise ValueError("out must not share memory with the vectors or weights")
        for kept in (batch.offsets, batch.tokens, batch.ngram_offsets, batch.ngrams):
            if kept is not None and np.may_share_memory(out, kept):
                raise ValueError("out must not share memory with the batch")
        return out

    def pool(self, batch, out=None):
        """weighted_mean@1 of every sentence of ``batch``, into ``out`` or a new array."""
        self._check_batch(batch)
        out = self._out(batch, out)
        bad = self._run(batch, out, NULL, 1)
        if bad >= 0:
            raise ValueError(f"row {batch.first_row + bad} contains NaN or infinite values")
        return out

    def accumulate(self, batch):
        """The accumulators and the token counts of every sentence of ``batch``."""
        self._check_batch(batch)
        accs = np.empty((batch.n_sentences, self.dim), dtype=np.float32)
        counts = np.zeros(batch.n_sentences, dtype=np.int64)
        cdef np.int64_t[::1] view = counts
        bad = self._run(batch, accs, <int64_t *>&view[0] if counts.shape[0] else NULL, 0)
        if bad >= 0:
            raise ValueError(f"row {batch.first_row + bad} contains NaN or infinite values")
        return accs, counts


def pool_weighted_mean(batch, vectors, weights, scales=None, ngram_vectors=None,
                       ngram_scales=None, oov_weight=None, *, isa=None):
    """``weighted_mean@1`` with the signature of ``fse._reference.pool_weighted_mean``."""
    return WeightedMean(vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight, isa=isa).pool(batch)


def accumulate(batch, vectors, weights, scales=None, ngram_vectors=None, ngram_scales=None,
               oov_weight=None, *, isa=None):
    """``fse._reference.accumulate`` in the kernel."""
    return WeightedMean(vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight, isa=isa).accumulate(batch)


# ---------------------------------------------------------------------------------------
# dot16, the postprocessing steps and score, as in fse._reference.
# ---------------------------------------------------------------------------------------


def _float32_rows(rows, name="rows"):
    if not isinstance(rows, np.ndarray) or rows.dtype != np.float32 or rows.ndim != 2:
        raise ValueError(f"{name} must be a float32 matrix")
    return np.ascontiguousarray(rows)


def dot16(rows, other, *, isa=None):
    """dot16 of every row with ``other`` (one vector, or one row per row)."""
    with _KernelEnv():
        return _dot16(rows, other, isa)


def _dot16(rows, other, isa):
    cdef int code = _check_isa(isa)
    rows = _float32_rows(rows)
    other = np.ascontiguousarray(np.broadcast_to(np.asarray(other, dtype=np.float32), rows.shape))
    cdef const float[:, ::1] a = rows
    cdef const float[:, ::1] b = other
    out = np.empty(rows.shape[0], dtype=np.float32)
    cdef float[::1] o = out
    cdef int n = _check_dim(rows.shape[1])
    if a.shape[0] == 0 or n == 0:
        out[:] = 0
        return out
    with nogil:
        fse_dot16_rows(code, &a[0, 0], &b[0, 0], a.shape[0], n, &o[0])
    return out


def postprocess(rows, steps, first_row=0, *, isa=None):
    """``fse._reference.postprocess`` in the kernel: the same rows, bits and errors."""
    with _KernelEnv():
        return _postprocess(rows, steps, first_row, isa)


def _postprocess(rows, steps, first_row, isa):
    from fse._contract import Normalize, Project, RemoveComponents, check_steps

    cdef int code = _check_isa(isa)
    out = np.array(rows, dtype=np.float32, copy=True, order="C")
    if out.ndim != 2:
        raise ValueError("rows must be a matrix")
    _check_dim(out.shape[1])
    _check_finite_rows(out, first_row)
    cdef float[:, ::1] view
    cdef const float[:, ::1] params
    cdef const float[::1] weights
    cdef float[::1] scratch
    cdef float[:, ::1] projected
    cdef int64_t bad, unnormalizable
    cdef int64_t n
    cdef int dim, k
    for step in check_steps(steps, out.shape[1]):
        n, dim = out.shape[0], out.shape[1]
        if n == 0:
            if isinstance(step, Project):
                out = np.empty((0, step.output_dim), dtype=np.float32)
            continue
        view = out
        if isinstance(step, RemoveComponents):
            params = step.components
            k = params.shape[0]
            # The kernel reads through raw pointers: check the shapes again, as numpy lets
            # a caller reshape the arrays of a checked step.
            if not 1 <= k <= dim or params.shape[1] != dim:
                raise ValueError(f"components must have shape (k, {dim}) with 1 <= k <= {dim}")
            if step.weights is not None and step.weights.shape != (k,):
                raise ValueError(f"weights must have shape ({k},)")
            scratch = np.empty(dim, dtype=np.float32)
            if step.weights is not None:
                weights = step.weights
                with nogil:
                    bad = fse_remove_components(code, &view[0, 0], n, dim, &params[0, 0], k, &weights[0], &scratch[0])
            else:
                with nogil:
                    bad = fse_remove_components(code, &view[0, 0], n, dim, &params[0, 0], k, NULL, &scratch[0])
        elif isinstance(step, Project):
            params = step.basis
            k = params.shape[0]
            if not 1 <= k <= dim or params.shape[1] != dim:
                raise ValueError(f"basis must have shape (k, {dim}) with 1 <= k <= {dim}")
            result = np.empty((n, k), dtype=np.float32)
            projected = result
            with nogil:
                bad = fse_project(code, &view[0, 0], n, dim, &params[0, 0], k, &projected[0, 0])
            out = result
        else:
            with nogil:
                bad = fse_normalize(code, &view[0, 0], n, dim, &unnormalizable)
            if unnormalizable >= 0:
                raise ValueError(
                    f"row {first_row + unnormalizable} cannot be normalized: the sum of its "
                    "squares underflows or overflows float32"
                )
        if bad >= 0:
            raise ValueError(f"row {first_row + bad} contains NaN or infinite values")
    return out


def _check_finite_rows(rows, first_row):
    bad = np.flatnonzero(~np.isfinite(rows).all(axis=1))
    if len(bad):
        raise ValueError(f"row {first_row + int(bad[0])} contains NaN or infinite values")


def score(stored, scales, weights, bias=0.0, *, isa=None):
    """``fse._reference.score`` in the kernel: dot16(stored, w) * scale + bias per row."""
    with _KernelEnv():
        return _score(stored, scales, weights, bias, isa)


def _score(stored, scales, weights, bias, isa):
    cdef int code = _check_isa(isa)
    stored = _rows(stored, "stored")
    cdef Py_ssize_t n = stored.shape[0]
    cdef int dim = _check_dim(stored.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):  # as the reference
        weights = np.ascontiguousarray(weights, dtype=np.float32)
    if weights.shape != (dim,):
        raise ValueError(f"weights must hold {dim} values")
    cdef bint int8 = stored.dtype == np.int8
    if int8 != (scales is not None):
        raise ValueError("scales are given exactly for int8 rows")
    if int8:
        scales = _float32_vector(scales, "scales", n)
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out
    cdef float[::1] o = out
    cdef const float[::1] w = weights
    cdef float[::1] scratch = np.empty(max(dim, 1), dtype=np.float32)
    cdef const void *data = np.PyArray_DATA(stored)
    cdef const float *sc = <const float *>np.PyArray_DATA(scales) if int8 else NULL
    cdef int dtype = _DTYPES[stored.dtype]
    cdef float b
    with np.errstate(over="ignore", invalid="ignore"):  # as the reference
        b = np.float32(bias)
    cdef int64_t bad
    with nogil:
        bad = fse_score(code, dtype, data, sc, n, dim, &w[0] if dim else NULL, b, &o[0], &scratch[0])
    if bad >= 0:
        raise ValueError(f"the score of row {bad} overflows float32")
    return out


# ---------------------------------------------------------------------------------------
# Tokenizers (ARCHITECTURE.md, "Tokenizers"), as in fse._reference_text.
# ---------------------------------------------------------------------------------------

cdef extern from "_native_text.h":
    ctypedef struct fse_tokens:
        const unsigned char *bytes
        int64_t *starts
        int64_t *ends
        int64_t n_tokens
    void fse_text_init()
    uint32_t fse_simple_lower(uint32_t cp) nogil
    int64_t fse_tokenize(int tokenizer, const unsigned char *p, int64_t n, unsigned char *cand,
                         fse_tokens *out) nogil
    int FSE_TOKENIZER_EXTERNAL
    int FSE_TOKENIZER_FSE_WORDS

fse_text_init()

TOKENIZER_EXTERNAL = FSE_TOKENIZER_EXTERNAL
TOKENIZER_FSE_WORDS = FSE_TOKENIZER_FSE_WORDS
_TOKENIZERS = {"external": FSE_TOKENIZER_EXTERNAL, "fse_words@1": FSE_TOKENIZER_FSE_WORDS}


def _tokenizer_code(tokenizer):
    from fse._contract import check_tokenizer

    return _TOKENIZERS[check_tokenizer(tokenizer)]


def _simple_lower(code_points):
    """The simple lowercase of every uint32 code point, by the native table; for tests."""
    cdef const uint32_t[::1] cps = np.ascontiguousarray(code_points, dtype=np.uint32)
    out = np.empty(cps.shape[0], dtype=np.uint32)
    cdef uint32_t[::1] o = out
    cdef Py_ssize_t i
    with nogil:
        for i in range(cps.shape[0]):
            o[i] = fse_simple_lower(cps[i])
    return out


def _raise_decode_error(data, position, where):
    """Raise CPython's error for invalid UTF-8 data, whose first bad byte the native
    decoder found at position; a disagreement between the two is a bug and says so."""
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as err:
        if err.start != position:
            raise AssertionError(
                f"the native decoder stopped at byte {position}, CPython at {err.start}"
            ) from err
        raise UnicodeDecodeError(err.encoding, err.object, err.start, err.end, err.reason + where) from None
    raise AssertionError(f"the native decoder rejected valid UTF-8 at byte {position}")


def tokenize(text, tokenizer="fse_words@1"):
    """The tokens of one sentence (str, or UTF-8 bytes) by the native tokenizer."""
    cdef int code = _tokenizer_code(tokenizer)
    if isinstance(text, str):
        data = (<str>text).encode("utf-8")
    elif isinstance(text, bytes):
        data = text
    else:
        raise TypeError(f"text must be str or bytes, not {type(text).__name__}")
    cdef const unsigned char *p = data
    cdef int64_t n = len(data)
    cdef unsigned char *cand = <unsigned char *>malloc(2 * n + 8)
    cdef int64_t *starts = <int64_t *>malloc((n + 1) * sizeof(int64_t))
    cdef int64_t *ends = <int64_t *>malloc((n + 1) * sizeof(int64_t))
    cdef fse_tokens out
    cdef int64_t bad
    try:
        if cand == NULL or starts == NULL or ends == NULL:
            raise MemoryError("could not allocate the token buffers")
        out.starts = starts
        out.ends = ends
        with nogil:
            bad = fse_tokenize(code, p, n, cand, &out)
        if bad >= 0:
            _raise_decode_error(data, bad, "")
        return [out.bytes[starts[i]:ends[i]].decode("utf-8") for i in range(out.n_tokens)]
    finally:
        free(cand)
        free(starts)
        free(ends)


# ---------------------------------------------------------------------------------------
# Stages 1 and 2: sentences of text to batches, without the GIL.
# ---------------------------------------------------------------------------------------

# A sentence longer than this is an error: fse reads one sentence per line or string.
cdef Py_ssize_t _MAX_SENTENCE_BYTES = 1 << 30
MAX_SENTENCE_BYTES = _MAX_SENTENCE_BYTES

cdef enum:
    ENCODE_OK = 0
    ENCODE_NO_MEMORY = 1
    ENCODE_BAD_UTF8 = 2
    ENCODE_TOO_LONG = 3


cdef inline int _ensure(_Buffer *b, Py_ssize_t extra) noexcept nogil:
    """Make room for extra more bytes; returns -1 if memory runs out."""
    if extra <= b.capacity - b.size:
        return 0
    return _grow_nogil(b, extra)


cdef class Workspace:
    """The buffers one worker reuses for every batch it encodes from text.

    A Workspace is used by one thread at a time; give every worker its own.
    """

    cdef _Buffer cand, starts, ends, word, tokens, offsets, ngram_offsets, ngrams
    cdef bint busy

    def __dealloc__(self):
        free(self.cand.data)
        free(self.starts.data)
        free(self.ends.data)
        free(self.word.data)
        free(self.tokens.data)
        free(self.offsets.data)
        free(self.ngram_offsets.data)
        free(self.ngrams.data)

    def __reduce__(self):
        raise TypeError("a Workspace belongs to one worker and cannot be pickled")


cdef class TextLookup:
    """Stages 1 and 2 for one vocabulary: the tokenizer, the lookup and FastText hashing.

    Immutable and shared by all workers; ``encode`` turns a range of sentences into a
    Batch. Without ``fasttext`` (min_n, max_n, bucket), a token that is no word is dropped;
    with it, it becomes OOV_TOKEN with its first ``max_ngrams`` ngram buckets.
    """

    cdef VocabTable table
    cdef readonly int tokenizer
    cdef bint fasttext
    cdef unsigned int min_n, max_n
    cdef uINT_t bucket
    cdef size_t max_ngrams

    def __init__(self, VocabTable table not None, tokenizer, fasttext=None, max_ngrams=0):
        self.table = table
        self.tokenizer = _tokenizer_code(tokenizer)
        self.fasttext = fasttext is not None
        if isinstance(max_ngrams, bool) or not isinstance(max_ngrams, (int, np.integer)):
            raise TypeError(f"max_ngrams must be an int, not {type(max_ngrams).__name__}")
        if self.fasttext:
            if not isinstance(fasttext, tuple) or len(fasttext) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in fasttext
            ):
                raise TypeError(f"fasttext must be a tuple of three ints (min_n, max_n, bucket), not {fasttext!r}")
            min_n, max_n, bucket = fasttext
            if not 1 <= min_n <= max_n or bucket < 0 or bucket >= 2**32:
                raise ValueError(f"fasttext must be (min_n, max_n, bucket) with 1 <= min_n <= max_n, not {fasttext}")
            if not 0 <= max_ngrams <= MAX_NGRAMS_LIMIT:
                raise ValueError(f"max_ngrams must be between 0 and {MAX_NGRAMS_LIMIT}, not {max_ngrams}")
            self.min_n, self.max_n, self.bucket = min_n, max_n, bucket
            self.max_ngrams = max_ngrams
        elif max_ngrams != 0:
            raise ValueError("max_ngrams needs fasttext")

    def __reduce__(self):
        raise TypeError("a TextLookup cannot be pickled; build it from the word vectors")

    cdef int _sentence(self, const unsigned char *p, Py_ssize_t n, Workspace ws,
                       Py_ssize_t *bad) noexcept nogil:
        """Append the tokens of the sentence p[0:n] to the batch buffers of ws."""
        cdef fse_tokens out
        cdef Py_ssize_t k, start, length, cap, kept
        cdef bint padded
        cdef uINT_t *ids
        cdef int64_t err
        cdef uINT_t index
        cdef size_t written
        if n > _MAX_SENTENCE_BYTES:
            return ENCODE_TOO_LONG
        ws.cand.size = 0
        ws.starts.size = 0
        ws.ends.size = 0
        # 32 bytes of slack, so every token of the candidates can be read 16 bytes wide.
        if (_ensure(&ws.cand, 2 * n + 32) or _ensure(&ws.starts, (n + 1) * <Py_ssize_t>sizeof(int64_t))
                or _ensure(&ws.ends, (n + 1) * <Py_ssize_t>sizeof(int64_t))):
            return ENCODE_NO_MEMORY
        out.starts = <int64_t *>ws.starts.data
        out.ends = <int64_t *>ws.ends.data
        err = fse_tokenize(self.tokenizer, p, n, <unsigned char *>ws.cand.data, &out)
        # fse_words@1 tokens lie in the candidate buffer, external ones in the input.
        padded = self.tokenizer == FSE_TOKENIZER_FSE_WORDS
        if err >= 0:
            bad[0] = err
            return ENCODE_BAD_UTF8
        if _ensure(&ws.tokens, out.n_tokens * <Py_ssize_t>sizeof(uINT_t)):
            return ENCODE_NO_MEMORY
        ids = <uINT_t *>(ws.tokens.data + ws.tokens.size)
        kept = 0
        for k in range(out.n_tokens):
            start = out.starts[k]
            length = out.ends[k] - start
            index = _vocab_lookup(&self.table.view, out.bytes + start, length, padded)
            if index != NOT_FOUND:
                ids[kept] = index
                kept += 1
            elif self.fasttext:
                ids[kept] = OOV_TOKEN
                kept += 1
                # The ngrams of "<token>": fewer than (length + 2) * max_n of them.
                ws.word.size = 0
                if _ensure(&ws.word, length + 2):
                    return ENCODE_NO_MEMORY
                ws.word.data[0] = b"<"
                memcpy(ws.word.data + 1, out.bytes + start, length)
                ws.word.data[length + 1] = b">"
                cap = min(<Py_ssize_t>self.max_ngrams, (length + 2) * <Py_ssize_t>self.max_n)
                if _ensure(&ws.ngrams, cap * <Py_ssize_t>sizeof(uINT_t)):
                    return ENCODE_NO_MEMORY
                written = ft_ngram_hashes_into(
                    <const unsigned char *>ws.word.data, length + 2, self.min_n, self.max_n,
                    self.bucket, <uINT_t *>(ws.ngrams.data + ws.ngrams.size), cap,
                )
                ws.ngrams.size += written * sizeof(uINT_t)
                if _push_i64_nogil(&ws.ngram_offsets, ws.ngrams.size // sizeof(uINT_t)):
                    return ENCODE_NO_MEMORY
        ws.tokens.size += kept * sizeof(uINT_t)
        if _push_i64_nogil(&ws.offsets, ws.tokens.size // sizeof(uINT_t)):
            return ENCODE_NO_MEMORY
        return ENCODE_OK

    def encode(self, const unsigned char[::1] data, offsets, Py_ssize_t first, Py_ssize_t last,
               first_row, Workspace ws not None):
        """The Batch of one range of sentences of data, which starts at row first_row.

        Line mode (offsets is None): the sentences are the lines that start in the bytes
        [first, last), which the caller aligns to line starts. Item mode: the sentences
        are the strings first to last - 1, delimited by the int64 offsets.
        """
        from fse._contract import Batch

        _check_span(data, offsets, first, last)
        if ws.busy:
            raise RuntimeError("a Workspace is used by one thread at a time")
        cdef Py_ssize_t n = data.shape[0]
        cdef const unsigned char *p = &data[0] if n else NULL
        cdef bint lines = offsets is None
        cdef const np.int64_t[::1] item_offsets
        cdef const np.int64_t *items = NULL
        if not lines:
            item_offsets = offsets
            items = &item_offsets[0]
        cdef Py_ssize_t pos = first, start, end, sentence = 0, bad = -1
        cdef const unsigned char *newline
        cdef int status = ENCODE_OK
        ws.busy = True
        try:
            ws.tokens.size = 0
            ws.offsets.size = 0
            ws.ngram_offsets.size = 0
            ws.ngrams.size = 0
            with nogil:
                if _push_i64_nogil(&ws.offsets, 0) or _push_i64_nogil(&ws.ngram_offsets, 0):
                    status = ENCODE_NO_MEMORY
                while status == ENCODE_OK and pos < last:
                    if lines:
                        newline = <const unsigned char *>memchr(p + pos, 10, n - pos)
                        start = pos
                        end = (newline - p) if newline != NULL else n
                        pos = end + 1 if newline != NULL else n
                    else:
                        start = items[pos]
                        end = items[pos + 1]
                        pos += 1
                    status = self._sentence(p + start, end - start, ws, &bad)
                    if status == ENCODE_OK:
                        sentence += 1
            if status == ENCODE_NO_MEMORY:
                raise MemoryError("could not grow the buffers of the Workspace")
            if status == ENCODE_TOO_LONG:
                raise ValueError(
                    f"sentence {sentence} of the range has {end - start} bytes, more than "
                    f"{MAX_SENTENCE_BYTES}; fse reads one sentence per line or string"
                )
            if status == ENCODE_BAD_UTF8:
                _raise_decode_error(
                    bytes(data[start:end]), bad,
                    f" in sentence {sentence} of the range (byte {start + bad} of the source)",
                )
            tokens = _to_array(&ws.tokens, np.uint32)
            batch_offsets = _to_array(&ws.offsets, np.int64)
            if not self.fasttext:
                return Batch(first_row, batch_offsets, tokens)
            return Batch(
                first_row, batch_offsets, tokens,
                _to_array(&ws.ngram_offsets, np.int64), _to_array(&ws.ngrams, np.uint32),
            )
        finally:
            ws.busy = False
