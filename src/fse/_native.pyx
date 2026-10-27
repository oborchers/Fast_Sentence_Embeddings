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


cdef inline np.uint64_t _fnv1a64(const unsigned char *p, Py_ssize_t n) noexcept nogil:
    cdef np.uint64_t h = 14695981039346656037ULL
    cdef Py_ssize_t i
    for i in range(n):
        h ^= p[i]
        h *= 1099511628211ULL
    return h


# Words of at most this many UTF-8 bytes are stored in their slot, so looking one up
# usually reads a single cache line; longer words are compared in the arena.
cdef enum:
    INLINE_KEY_BYTES = 16


# One slot of the open-addressing table, 32 bytes. value is NOT_FOUND for a free slot.
cdef struct VocabSlot:
    np.uint64_t hash
    uINT_t value
    np.uint32_t length
    # The word itself if length <= INLINE_KEY_BYTES, else its offset in the arena (int64).
    unsigned char key[INLINE_KEY_BYTES]


cdef struct VocabView:
    const VocabSlot *slots
    const unsigned char *arena
    np.uint64_t mask


cdef inline uINT_t _vocab_lookup(const VocabView *v, const unsigned char *p, Py_ssize_t n) noexcept nogil:
    """Vocabulary index of the n bytes at p, or NOT_FOUND."""
    cdef np.uint64_t h = _fnv1a64(p, n)
    cdef np.uint64_t j = h & v.mask
    cdef const VocabSlot *slot
    cdef np.int64_t offset
    while True:
        slot = &v.slots[j]
        if slot.value == NOT_FOUND:
            return NOT_FOUND
        if slot.hash == h and slot.length == n:
            if n <= INLINE_KEY_BYTES:
                if memcmp(slot.key, p, n) == 0:
                    return slot.value
            else:
                memcpy(&offset, slot.key, sizeof(np.int64_t))
                if memcmp(v.arena + offset, p, n) == 0:
                    return slot.value
        j = (j + 1) & v.mask


cdef class VocabTable:
    """Immutable map from the UTF-8 bytes of a word to its vocabulary index.

    Built once per train or infer call from key_to_index while holding the GIL, then
    read without it: open addressing with linear probing, at most half full. Each slot
    holds the hash, the index and, for words of up to INLINE_KEY_BYTES bytes, the word
    itself, so a lookup touches one slot instead of several arrays.
    """

    cdef bytes arena
    cdef object _slots
    cdef VocabView view

    def __init__(self, dict key_to_index):
        cdef Py_ssize_t n = len(key_to_index)
        cdef np.uint64_t size = 2
        while size < 2 * <np.uint64_t>n:
            size <<= 1
        # Raw bytes, so the slot layout is the C struct's, whatever numpy would pad.
        self._slots = np.zeros(size * sizeof(VocabSlot), dtype=np.uint8)
        cdef VocabSlot *slots = <VocabSlot *>np.PyArray_DATA(self._slots)
        cdef np.uint64_t j
        for j in range(size):
            slots[j].value = NOT_FOUND

        cdef list words = []
        cdef list parts = []
        cdef np.int64_t position = 0
        for word, index in key_to_index.items():
            encoded = (<str?>word).encode("utf-8")
            words.append((encoded, index, position))
            if len(encoded) > INLINE_KEY_BYTES:
                parts.append(encoded)
                position += len(encoded)
        self.arena = b"".join(parts)
        self.view.slots = slots
        self.view.arena = self.arena
        self.view.mask = size - 1

        cdef np.uint64_t h
        cdef np.int64_t offset
        cdef const unsigned char *raw
        cdef Py_ssize_t length
        for encoded, index, offset in words:
            raw = <bytes>encoded
            length = len(encoded)
            h = _fnv1a64(raw, length)
            j = h & self.view.mask
            while slots[j].value != NOT_FOUND:
                j = (j + 1) & self.view.mask
            slots[j].hash = h
            slots[j].value = <uINT_t>index
            slots[j].length = <np.uint32_t>length
            if length <= INLINE_KEY_BYTES:
                memcpy(slots[j].key, raw, length)
            else:
                memcpy(slots[j].key, &offset, sizeof(np.int64_t))

    def __reduce__(self):
        raise TypeError("VocabTable only exists during a train or infer call")

    def get(self, word):
        """Vocabulary index of word (str), or None; for tests."""
        encoded = (<str?>word).encode("utf-8")
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

from libc.stdint cimport int64_t, uint32_t

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
        self.n_words, self.dim = vectors.shape
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
        self.params.oov_weight = np.float32(oov_weight) if oov_weight is not None else 0
        self._keep = (vectors, weights, scales, ngram_vectors, ngram_scales, oov_weight)

    def _check_batch(self, batch):
        from fse._contract import OOV_TOKEN, Batch

        if not isinstance(batch, Batch):
            raise TypeError(f"batch must be a Batch, not {type(batch).__name__}")
        tokens = batch.tokens
        unknown = tokens == OOV_TOKEN
        if unknown.any():
            if batch.ngrams is None or batch.ngram_offsets is None:
                raise ValueError("an unknown FastText word needs ngram buckets")
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

    def _out(self, n, out):
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
        return out

    def pool(self, batch, out=None):
        """weighted_mean@1 of every sentence of ``batch``, into ``out`` or a new array."""
        self._check_batch(batch)
        out = self._out(batch.n_sentences, out)
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
