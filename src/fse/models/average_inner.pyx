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

"""Optimized cython functions for computing sentence embeddings"""

import cython
import numpy as np

cimport numpy as np

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.pyport cimport PY_SSIZE_T_MAX
from cpython.object cimport PyObject
from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.pythread cimport (
    WAIT_LOCK,
    PyThread_acquire_lock,
    PyThread_allocate_lock,
    PyThread_free_lock,
    PyThread_release_lock,
)
from libc.float cimport FLT_EPSILON
from libc.math cimport isfinite
from libc.stdint cimport SIZE_MAX
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memchr, memcmp, memcpy, memset

from array import array
from time import time

from fse.models import _index_tracker

import scipy.linalg.blas as fblas

cdef extern from "Python.h":
    # Raises (TypeError, UnicodeEncodeError) instead of returning NULL silently.
    const char *PyUnicode_AsUTF8AndSize(object unicode, Py_ssize_t *size) except NULL
    # NULL with an exception set on errors in __hash__ or __eq__, NULL without one if the
    # key is missing. Unlike PyDict_GetItem it does not swallow the errors.
    PyObject *PyDict_GetItemWithError(object d, object key) except? NULL

# y += alpha * x, taken from scipy's BLAS. Raises at import if scipy exposes no pointer.
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCapsule_GetPointer(fblas.saxpy._cpointer, NULL)

# Row kernels: mem += factor * row for one stored word or ngram row, in float32. float32
# rows go through BLAS saxpy, which dispatches by CPU itself. float16 and int8 rows have
# portable loops; on x86-64 with GCC or Clang, AVX2 versions are picked at import when the
# CPU has AVX2, FMA and F16C, without changing the build flags. On arm64 the compiler
# vectorizes the portable loops with NEON, and float16 converts with the native __fp16
# type. Every kernel adds the values of a row in index order, so the sum of a sentence
# does not depend on how jobs are split.
cdef extern from *:
    """
    #include <stddef.h>
    #include <stdint.h>
    #include <string.h>
    #if defined(_MSC_VER)
    #define FSE_RESTRICT __restrict
    #else
    #define FSE_RESTRICT __restrict__
    #endif
    #if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    #define FSE_X86_DISPATCH 1
    #include <cpuid.h>
    #include <immintrin.h>
    #endif
    #if defined(__GNUC__) || defined(__clang__)
    #define FSE_PREFETCH(p) __builtin_prefetch((const void *)(p), 0, 3)
    #elif defined(_M_X64)
    #include <xmmintrin.h>
    #define FSE_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
    #else
    #define FSE_PREFETCH(p) ((void)(p))
    #endif

    typedef void (*fse_saxpy_t)(const int *, const float *, const float *, const int *, float *, const int *);
    typedef void (*fse_add_row_t)(float *, const void *, float, int);
    static fse_saxpy_t fse_saxpy = NULL;
    /* Every float16 bit pattern as float32, filled at import. */
    static float fse_half_table[65536];

    /* Load every cache line of a row that is about to be added: one prefetch per 64
       bytes from its start, and one at its last byte for a row that straddles a line. */
    static inline void fse_prefetch_row(const void *row, size_t bytes) {
        const char *p = (const char *)row;
        if (!bytes) return;
        for (size_t offset = 0; offset < bytes; offset += 64) FSE_PREFETCH(p + offset);
        FSE_PREFETCH(p + bytes - 1);
    }

    static void fse_add_f32_blas(float *mem, const void *row, float factor, int n) {
        int one = 1;
        fse_saxpy(&n, &factor, (const float *)row, &one, mem, &one);
    }
    static void fse_add_f16(float *FSE_RESTRICT mem, const void *row, float factor, int n) {
        const uint16_t *FSE_RESTRICT v = (const uint16_t *)row;
        for (int i = 0; i < n; i++) mem[i] += factor * fse_half_table[v[i]];
    }
    #if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    static void fse_add_f16_neon(float *FSE_RESTRICT mem, const void *row, float factor, int n) {
        const __fp16 *FSE_RESTRICT v = (const __fp16 *)row;
        for (int i = 0; i < n; i++) mem[i] += factor * (float)v[i];
    }
    #endif
    static void fse_add_i8(float *FSE_RESTRICT mem, const void *row, float factor, int n) {
        const int8_t *FSE_RESTRICT v = (const int8_t *)row;
        for (int i = 0; i < n; i++) mem[i] += factor * (float)v[i];
    }

    #if defined(FSE_X86_DISPATCH)
    __attribute__((target("avx2,fma,f16c"))) static void fse_add_f16_avx2(float *mem, const void *row, float factor, int n) {
        const uint16_t *v = (const uint16_t *)row;
        __m256 f = _mm256_set1_ps(factor);
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v + i)));
            _mm256_storeu_ps(mem + i, _mm256_fmadd_ps(f, x, _mm256_loadu_ps(mem + i)));
        }
        for (; i < n; i++) mem[i] = __builtin_fmaf(factor, fse_half_table[v[i]], mem[i]);
    }
    __attribute__((target("avx2,fma"))) static void fse_add_i8_avx2(float *mem, const void *row, float factor, int n) {
        const int8_t *v = (const int8_t *)row;
        __m256 f = _mm256_set1_ps(factor);
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(v + i))));
            _mm256_storeu_ps(mem + i, _mm256_fmadd_ps(f, x, _mm256_loadu_ps(mem + i)));
        }
        for (; i < n; i++) mem[i] = __builtin_fmaf(factor, (float)v[i], mem[i]);
    }
    #endif

    /* The kernels in use, by format (0 float32, 1 float16, 2 int8), and the instruction
       set they were chosen for. */
    static fse_add_row_t fse_kernels[3];
    static const char *fse_kernel_isa = "portable";

    /* Choose the kernels: the fastest the CPU supports, or with portable set, the
       fallback of a CPU without AVX2 (saxpy for float32, the portable loops otherwise). */
    static void fse_select_kernels(int portable) {
        fse_kernels[1] = fse_add_f16;
        fse_kernels[2] = fse_add_i8;
        /* float32 rows go through BLAS everywhere: end to end, an own loop was no faster
           on Apple M1 (NEON) or AMD Zen 5 (AVX2), and saxpy keeps float32 results as in
           1.0. The gain for float32 comes from prefetching the rows. */
        fse_kernels[0] = fse_add_f32_blas;
    #if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        if (!portable) {
            fse_kernels[1] = fse_add_f16_neon;
            fse_kernel_isa = "neon";
        } else {
            fse_kernel_isa = "portable";
        }
        return;
    #else
        fse_kernel_isa = "portable";
    #if defined(FSE_X86_DISPATCH)
        __builtin_cpu_init();
        /* F16C through CPUID: __builtin_cpu_supports knows "f16c" only since Clang 18.
           It is bit 29 of ECX in leaf 1; AVX2 implies that the OS saves the AVX state. */
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        int f16c = __get_cpuid(1, &eax, &ebx, &ecx, &edx) && ((ecx >> 29) & 1u);
        if (!portable && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && f16c) {
            fse_kernels[1] = fse_add_f16_avx2;
            fse_kernels[2] = fse_add_i8_avx2;
            fse_kernel_isa = "avx2";
        }
    #endif
    #endif
    }
    """
    ctypedef void (*fse_saxpy_t)(const int *, const float *, const float *, const int *, float *, const int *) noexcept nogil
    fse_saxpy_t fse_saxpy
    float fse_half_table[65536]
    add_row_ptr fse_kernels[3]
    const char *fse_kernel_isa
    void fse_select_kernels(int portable) noexcept nogil
    void fse_prefetch_row(const void *row, size_t bytes) noexcept nogil

fse_saxpy = <fse_saxpy_t>saxpy

# Tokens ahead whose rows are prefetched while a row is added. Measured on Apple M1 and
# AMD Zen 4: distance 8 was best or within noise of the best for every format and size.
cdef enum:
    PREFETCH_DISTANCE = 8

# Code formats of the word vectors, as indices into fse_kernels.
_FORMATS = {"float32": 0, "float16": 1, "int8": 2}
_CODE_TYPES = {"float32": np.float32, "float16": np.float16, "int8": np.int8}

cdef int ONE = <int>1
cdef int ZERO = <int>0

cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t ZEROF = <REAL_t>0.0

cdef class RowLocks:
    """Striped locks that serialize adding sentence averages into shared target rows.

    Only needed when several sentences map to the same row and more than one worker
    trains at the same time. Row ``r`` uses lock ``r % N_ROW_LOCKS``.
    """

    def __cinit__(self):
        cdef int i
        for i in range(N_ROW_LOCKS):
            self.locks[i] = NULL
        for i in range(N_ROW_LOCKS):
            self.locks[i] = PyThread_allocate_lock()
            if self.locks[i] == NULL:
                raise MemoryError("could not allocate row locks")

    def __dealloc__(self):
        cdef int i
        for i in range(N_ROW_LOCKS):
            if self.locks[i] != NULL:
                PyThread_free_lock(self.locks[i])

    def __reduce__(self):
        raise TypeError("RowLocks only exist during training and cannot be pickled")


# FastText ngram hashing, reimplemented after gensim's compute_ngrams_bytes and
# ft_hash_bytes (gensim/models/fasttext_inner.pyx), which port Facebook's fastText
# (src/dictionary.cc). Ngrams are taken over the UTF-8 characters of "<word>" in the
# same order: by start character, then by length. UTF-8 continuation bytes start with
# the bits 10.
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


cdef PyThread_type_lock *_row_locks_of(model):
    """Return the row locks set on the model for the current training, or NULL."""
    locks = getattr(model, "_row_locks", None)
    if locks is None:
        return NULL
    return (<RowLocks?>locks).locks


# A token of a FastText job that is not in the vocabulary. Its UTF-8 bytes "<token>" are
# stored with the job and hashed into ngram buckets by the worker, without the GIL.
cdef uINT_t OOV_TOKEN = <uINT_t>0xFFFFFFFE


# More ngrams than any word of a real corpus has; a bound that keeps buffer sizes in range.
MAX_NGRAMS_LIMIT = 1 << 24

# Sentences of a native chunk at most, besides its byte size (see scan_spans).
cdef Py_ssize_t CHUNK_MAX_ROWS = 1 << 16
MAX_CHUNK_ROWS = CHUNK_MAX_ROWS


cdef class KernelContext:
    """Read-only state of one train or infer call, shared by all workers.

    Built once per call instead of once per job. It keeps references to every array it
    holds a raw pointer into, so they stay alive while workers use the pointers.
    """

    cdef KernelParams params
    cdef dict vocab
    cdef Py_ssize_t n_rows
    # Rows of the target array itself, which a ring of windows never exceeds.
    cdef Py_ssize_t target_rows
    cdef object _keep
    cdef object _locks
    # Native sources opened for this call; closed when the call ends.
    cdef public list sources
    # Set EPS in the rows of every job before averaging, instead of in all rows in advance.
    # Only for one row per sentence and contiguous rows per job (indices are positions).
    cdef public bint prefill_rows

    def __init__(self, model, target):
        _check_inputs(model, target)
        wv = model.wv
        self.vocab = wv.key_to_index
        self.n_rows = target.shape[0]
        self.target_rows = target.shape[0]
        self.params.size = model.sv.vector_size
        self.params.is_ft = model.is_ft
        self.params.word_weights = <REAL_t *>(np.PyArray_DATA(model.word_weights))
        self.params.target = <REAL_t *>(np.PyArray_DATA(target))
        self.params.ring_rows = 0
        self.params.locks = _row_locks_of(model)
        fmt = _word_format(model)
        words, word_scales, ngrams, ngram_scales = model._word_arrays()
        self.params.add_row = fse_kernels[_FORMATS[fmt]]
        self.params.row_bytes = <size_t>model.sv.vector_size * <size_t>np.dtype(_CODE_TYPES[fmt]).itemsize
        self.params.word_vectors = np.PyArray_DATA(words)
        self.params.word_scales = <const REAL_t *>np.PyArray_DATA(word_scales) if word_scales is not None else NULL
        self.params.ngram_scales = <const REAL_t *>np.PyArray_DATA(ngram_scales) if ngram_scales is not None else NULL
        if model.is_ft:
            self.params.ngram_vectors = np.PyArray_DATA(ngrams)
            self.params.min_n = wv.min_n
            self.params.max_n = wv.max_n
            self.params.bucket = wv.bucket
            if not 1 <= model.batch_ngrams <= MAX_NGRAMS_LIMIT:
                raise ValueError(
                    f"batch_ngrams must be between 1 and {MAX_NGRAMS_LIMIT}, got {model.batch_ngrams}"
                )
            self.params.max_ngrams = model.batch_ngrams
            # Unknown words are weighted like the most heavily weighted known word.
            self.params.oov_weight = <REAL_t>np.max(model.word_weights)
        else:
            self.params.ngram_vectors = NULL
            self.params.max_ngrams = 0
        self._keep = (model.word_weights, target, words, word_scales, ngrams, ngram_scales)
        self._locks = getattr(model, "_row_locks", None)
        self.sources = []
        self.prefill_rows = False

    def use_ring(self, Py_ssize_t n_rows):
        """Treat the target as a ring of windows for n_rows rows: row r goes to r % len(target).

        For training into windows of rows that are stored once finished, so only the
        window ring exists in float32. Requires prefill_rows, so every job writes its own
        contiguous rows; the caller makes sure a window is free before its rows are written.
        """
        if self.params.ring_rows:
            raise ValueError("the target is already a ring")
        if n_rows < 0:
            raise ValueError(f"n_rows must be non-negative, got {n_rows}")
        if self.target_rows < 1:
            raise ValueError("the target of a ring needs at least one row")
        self.params.ring_rows = <size_t>self.target_rows
        self.n_rows = n_rows

    def __reduce__(self):
        raise TypeError("KernelContext only exists during a train or infer call")


cdef class EncodedJob:
    """A batch of sentences encoded for the kernel.

    ``words[offsets[s]:offsets[s + 1]]`` are the tokens of sentence ``s``, which is added
    into row ``targets[s]``: vocabulary indices, or ``OOV_TOKEN`` for an unknown FastText
    word whose bytes are ``oov_bytes[oov_offsets[k]:oov_offsets[k + 1]]``, in order.
    ``n_items`` counts every input item of the job, including empty sentences.
    ``first_row`` is the index of the first item; ``contiguous`` says whether the items
    have the indices ``first_row`` to ``first_row + n_items - 1`` in order.
    """

    cdef readonly Py_ssize_t n_items
    cdef readonly np.int64_t first_row
    cdef readonly bint contiguous
    cdef readonly Py_ssize_t n_sentences
    cdef readonly Py_ssize_t eff_words
    cdef object words
    cdef object offsets
    cdef object targets
    cdef bytes oov_bytes
    cdef object oov_offsets


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


cdef class JobEncoder:
    """Read (tokens, index) tuples from an iterable and encode them into jobs.

    Batching and encoding happen in one pass, holding the GIL: a job takes items while
    it has fewer than ``batch_words`` items and its words stay within ``batch_words``; a
    longer item starts a job of its own. Empty sentences are skipped, as is every
    unknown word of a Word2Vec-style model. The number of non-empty sentences and of
    used tokens are the counts train reports.
    """

    cdef KernelContext ctx
    cdef object iterator
    cdef Py_ssize_t batch_words
    cdef object pending
    cdef bint has_pending
    cdef bint failed
    cdef _Buffer words, offsets, targets, oov_bytes, oov_offsets

    def __cinit__(self):
        memset(&self.words, 0, sizeof(_Buffer))
        memset(&self.offsets, 0, sizeof(_Buffer))
        memset(&self.targets, 0, sizeof(_Buffer))
        memset(&self.oov_bytes, 0, sizeof(_Buffer))
        memset(&self.oov_offsets, 0, sizeof(_Buffer))

    def __init__(self, KernelContext ctx, iterable, Py_ssize_t batch_words):
        if batch_words < 1:
            raise ValueError(f"batch_words must be positive, got {batch_words}")
        self.ctx = ctx
        self.iterator = iter(iterable)
        self.batch_words = batch_words
        self.has_pending = False
        self.failed = False

    def __dealloc__(self):
        free(self.words.data)
        free(self.offsets.data)
        free(self.targets.data)
        free(self.oov_bytes.data)
        free(self.oov_offsets.data)

    def __reduce__(self):
        raise TypeError("JobEncoder only exists during a train or infer call")

    def next_job(self):
        """Return the next EncodedJob, or None once the input is exhausted.

        After an exception the encoder refuses further calls: the items it had already
        read for the failed job are lost, so continuing would silently skip them.
        """
        if self.failed:
            raise RuntimeError("the JobEncoder failed before and cannot continue")
        try:
            return self._next_job()
        except BaseException:
            self.failed = True
            raise

    cdef object _next_job(self):
        cdef dict vocab = self.ctx.vocab
        cdef bint is_ft = self.ctx.params.is_ft
        cdef Py_ssize_t n_rows = self.ctx.n_rows
        cdef Py_ssize_t n_items = 0
        cdef Py_ssize_t n_words = 0
        cdef Py_ssize_t n_sentences = 0
        cdef Py_ssize_t length
        cdef Py_ssize_t utf8_size
        cdef np.int64_t first_row = -1, previous = -1
        cdef bint contiguous = True
        cdef const char *utf8
        cdef PyObject *hit

        self.words.size = 0
        self.offsets.size = 0
        self.targets.size = 0
        self.oov_bytes.size = 0
        self.oov_offsets.size = 0
        _push_i64(&self.offsets, 0)
        _push_i64(&self.oov_offsets, 0)

        while True:
            if self.has_pending:
                obj = self.pending
                self.pending = None
                self.has_pending = False
            else:
                try:
                    obj = next(self.iterator)
                except StopIteration:
                    break
            sentence = obj[0]
            length = len(sentence)
            # Compared without n_words + length, which could overflow for a huge length.
            if n_items and (
                n_items >= self.batch_words
                or n_words > self.batch_words
                or length > self.batch_words - n_words
            ):
                self.pending = obj
                self.has_pending = True
                break
            n_items += 1
            n_words = n_words + length if length <= PY_SSIZE_T_MAX - n_words else PY_SSIZE_T_MAX

            index = obj[1]
            # The kernel writes into target[index] through a raw pointer.
            if not 0 <= index < n_rows:
                raise IndexError(f"sentence index {index} is out of range for {n_rows} target rows")
            if n_items == 1:
                first_row = index
            elif index != previous + 1:
                contiguous = False
            previous = index
            # Truthiness, as in the numpy fallback.
            if not sentence:
                continue
            for token in sentence:
                hit = PyDict_GetItemWithError(vocab, token)
                if hit != NULL:
                    _push_u32(&self.words, <uINT_t>(<object>hit))
                elif is_ft:
                    # The UTF-8 bytes of "<token>", as fastText hashes them. A subclass of
                    # str is formatted like gensim does, which may call its __str__.
                    if type(token) is str:
                        utf8 = PyUnicode_AsUTF8AndSize(token, &utf8_size)
                        _push_bytes(&self.oov_bytes, "<", 1)
                        _push_bytes(&self.oov_bytes, utf8, utf8_size)
                        _push_bytes(&self.oov_bytes, ">", 1)
                    else:
                        formatted = ("<%s>" % token).encode("utf-8")
                        _push_bytes(&self.oov_bytes, formatted, len(formatted))
                    _push_i64(&self.oov_offsets, self.oov_bytes.size)
                    _push_u32(&self.words, OOV_TOKEN)
            _push_u32(&self.targets, <uINT_t>index)
            n_sentences += 1
            _push_i64(&self.offsets, self.words.size // sizeof(uINT_t))

        if not n_items:
            return None
        cdef EncodedJob job = EncodedJob.__new__(EncodedJob)
        job.n_items = n_items
        job.first_row = first_row
        job.contiguous = contiguous
        job.n_sentences = n_sentences
        job.eff_words = self.words.size // sizeof(uINT_t)
        job.words = _to_array(&self.words, np.uint32)
        job.offsets = _to_array(&self.offsets, np.int64)
        job.targets = _to_array(&self.targets, np.uint32)
        job.oov_bytes = PyBytes_FromStringAndSize(self.oov_bytes.data, self.oov_bytes.size)
        job.oov_offsets = _to_array(&self.oov_offsets, np.int64)
        return job


def encode_job(KernelContext ctx, batch):
    """Encode all (tokens, index) tuples of batch into one EncodedJob."""
    items = list(batch)
    job = JobEncoder(ctx, items, max(len(items), 1) + sum(len(obj[0]) for obj in items)).next_job()
    if job is None:
        job = EncodedJob.__new__(EncodedJob)
        job.words = np.empty(0, dtype=np.uint32)
        job.offsets = np.zeros(1, dtype=np.int64)
        job.targets = np.empty(0, dtype=np.uint32)
        job.oov_bytes = b""
        job.oov_offsets = np.zeros(1, dtype=np.int64)
    return job


cdef void _average_sentence(
    const KernelParams *p,
    const uINT_t *words,
    Py_ssize_t n_tokens,
    const unsigned char *oov_bytes,
    const np.int64_t *oov_offsets,
    Py_ssize_t *k,
    uINT_t target,
    REAL_t *mem,
    uINT_t *ngram_buffer,
    size_t ngram_capacity,
) noexcept nogil:
    """Add the average of one sentence into row target.

    The weighted word vectors of the sentence are summed into mem, then mem / length is
    added into the row, as in the numpy fallback. For an unknown FastText word, each
    ngram vector is added scaled by oov_weight / ngrams, where numpy sums the ngram
    vectors first and scales the sum: the same value within float32 rounding. k is the
    index of the next unknown word in oov_offsets and is advanced past this sentence.

    Word and ngram rows are read through p.add_row in their stored format; int8 rows
    fold their scale into the factor. While a row is added, the row PREFETCH_DISTANCE
    tokens (or ngrams) ahead is prefetched, so its cache misses overlap the work.
    """
    cdef int size = p.size
    cdef Py_ssize_t t
    cdef size_t j, ngrams, row
    cdef uINT_t word, ahead
    cdef REAL_t inv_count, inv_ngram, factor
    cdef PyThread_type_lock lock
    cdef const char *word_rows = <const char *>p.word_vectors
    cdef const char *ngram_rows = <const char *>p.ngram_vectors
    cdef size_t row_bytes = p.row_bytes

    memset(mem, 0, size * cython.sizeof(REAL_t))
    for t in range(n_tokens):
        word = words[t]
        if t + PREFETCH_DISTANCE < n_tokens:
            ahead = words[t + PREFETCH_DISTANCE]
            if ahead != OOV_TOKEN:
                fse_prefetch_row(word_rows + <size_t>ahead * row_bytes, row_bytes)
        if word == OOV_TOKEN:
            ngrams = ft_ngram_hashes_into(
                &oov_bytes[oov_offsets[k[0]]],
                <size_t>(oov_offsets[k[0] + 1] - oov_offsets[k[0]]),
                p.min_n, p.max_n, p.bucket, ngram_buffer, ngram_capacity,
            )
            k[0] += 1
            if ngrams == 0:
                # No ngrams: the word counts for the length but adds no vector.
                continue
            inv_ngram = (ONEF / <REAL_t>ngrams) * p.oov_weight
            for j in range(ngrams):
                if j + PREFETCH_DISTANCE < ngrams:
                    fse_prefetch_row(
                        ngram_rows + <size_t>ngram_buffer[j + PREFETCH_DISTANCE] * row_bytes, row_bytes
                    )
                row = <size_t>ngram_buffer[j]
                factor = inv_ngram * p.ngram_scales[row] if p.ngram_scales != NULL else inv_ngram
                p.add_row(mem, ngram_rows + row * row_bytes, factor, size)
        else:
            factor = p.word_weights[word]
            if p.word_scales != NULL:
                factor = factor * p.word_scales[word]
            p.add_row(mem, word_rows + <size_t>word * row_bytes, factor, size)

    # Every stored token counts for the length, also unknown words without ngrams.
    if n_tokens > 0:
        inv_count = ONEF / <REAL_t>n_tokens
        row = _target_row(p, target) * <size_t>size
        # Adding mem / length keeps many-to-one mappings correct: rows accumulate.
        if p.locks != NULL:
            lock = p.locks[target % N_ROW_LOCKS]
            PyThread_acquire_lock(lock, WAIT_LOCK)
            saxpy(&size, &inv_count, mem, &ONE, &p.target[row], &ONE)
            PyThread_release_lock(lock)
        else:
            saxpy(&size, &inv_count, mem, &ONE, &p.target[row], &ONE)


cdef void _average_job(
    const KernelParams *p,
    const uINT_t *words,
    const np.int64_t *offsets,
    const uINT_t *targets,
    Py_ssize_t n_sentences,
    const unsigned char *oov_bytes,
    const np.int64_t *oov_offsets,
    REAL_t *mem,
    uINT_t *ngram_buffer,
    size_t ngram_capacity,
) noexcept nogil:
    """Add the average of every sentence of an encoded job into its target row."""
    cdef Py_ssize_t s, k = 0
    for s in range(n_sentences):
        _average_sentence(
            p, &words[offsets[s]] if words != NULL else NULL, offsets[s + 1] - offsets[s],
            oov_bytes, oov_offsets, &k, targets[s], mem, ngram_buffer, ngram_capacity,
        )


cdef inline size_t _target_row(const KernelParams *p, size_t row) noexcept nogil:
    """The row of the target that holds sentence row ``row``: itself, or its ring slot."""
    return row % p.ring_rows if p.ring_rows else row


cdef void _prefill(const KernelParams *p, np.int64_t first_row, Py_ssize_t n_rows) noexcept nogil:
    """Set rows first_row to first_row + n_rows - 1 to EPS, as _fill_eps does in advance."""
    cdef size_t i, r, begin
    cdef size_t size = <size_t>p.size
    for r in range(<size_t>first_row, <size_t>first_row + <size_t>n_rows):
        begin = _target_row(p, r) * size
        for i in range(begin, begin + size):
            p.target[i] = FLT_EPSILON


cdef int _check_prefill(KernelContext ctx, np.int64_t first_row, Py_ssize_t n_items) except -1:
    """Raise unless rows first_row to first_row + n_items - 1 exist in the target."""
    if first_row < 0 or n_items < 0 or first_row > ctx.n_rows - n_items:
        raise ValueError(
            f"rows [{first_row}, {first_row + n_items}) are outside the {ctx.n_rows} target rows"
        )
    return 0


def run_job(KernelContext ctx, EncodedJob job, scratch):
    """Average an encoded job into the target of ctx without holding the GIL.

    scratch is the worker's private float32 buffer of at least the vector size. Returns
    the number of sentences and of tokens the job used. With ``ctx.prefill_rows``, the rows
    of the job are set to EPS first; their indices must be contiguous.
    """
    _check_array(scratch, "scratch", np.float32)
    if scratch.size < ctx.params.size:
        raise ValueError(f"scratch must hold at least {ctx.params.size} entries")
    cdef bint prefill = ctx.prefill_rows
    if ctx.params.ring_rows and not prefill:
        raise ValueError("a ring of windows needs prefill_rows")
    if prefill:
        if not job.contiguous:
            raise ValueError("the indices of the job are not contiguous, so its rows cannot be prefilled")
        _check_prefill(ctx, job.first_row, job.n_items)
    cdef uINT_t[::1] words = job.words
    cdef np.int64_t[::1] offsets = job.offsets
    cdef uINT_t[::1] targets = job.targets
    cdef np.int64_t[::1] oov_offsets = job.oov_offsets
    cdef const unsigned char *oov_bytes = job.oov_bytes
    cdef REAL_t *mem = <REAL_t *>(np.PyArray_DATA(scratch))
    cdef uINT_t *ngram_buffer = NULL
    cdef Py_ssize_t n_sentences = job.n_sentences
    # Typed memoryviews of length 0 have no element to take the address of.
    cdef const uINT_t *words_ptr = &words[0] if words.shape[0] else NULL
    cdef const uINT_t *targets_ptr = &targets[0] if targets.shape[0] else NULL

    # A word of b UTF-8 bytes has fewer than b * max_n ngrams, so the buffer needs at most
    # that many entries for the longest unknown word, even if batch_ngrams is larger.
    cdef Py_ssize_t longest = 0
    cdef Py_ssize_t i
    for i in range(oov_offsets.shape[0] - 1):
        longest = max(longest, oov_offsets[i + 1] - oov_offsets[i])
    capacity = min(int(ctx.params.max_ngrams), int(longest) * max(int(ctx.params.max_n), 1))
    if capacity > PY_SSIZE_T_MAX // sizeof(uINT_t):
        raise OverflowError(f"cannot allocate an ngram buffer for {capacity} ngrams")
    cdef size_t ngram_capacity = capacity
    if ngram_capacity:
        ngram_buffer = <uINT_t *>malloc(ngram_capacity * sizeof(uINT_t))
        if ngram_buffer == NULL:
            raise MemoryError("could not allocate the ngram buffer")
    try:
        with nogil:
            if prefill:
                _prefill(&ctx.params, job.first_row, job.n_items)
            _average_job(
                &ctx.params, words_ptr, &offsets[0], targets_ptr, n_sentences,
                oov_bytes, &oov_offsets[0], mem, ngram_buffer, ngram_capacity,
            )
    finally:
        free(ngram_buffer)
    return job.n_sentences, job.eff_words


def scan_indexed_sentences(sentences, double progress_per, progress):
    """Validate every (tokens, index) tuple and count them, like the Python scan loop.

    The checks and their messages are those of
    ``BaseSentence2VecModel._check_indexed_sent_valid``. ``progress(sentences, words)`` is
    called about every ``progress_per`` seconds. Returns the number of sentences and
    words, the number of empty sentences, the largest index, whether every index
    occurs at most once, and whether every index equals the position of its sentence.
    """
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t total_words = 0
    cdef Py_ssize_t empty_sentences = 0
    cdef object max_index = 0
    cdef bint unique_indices = True
    cdef bint identity = True
    cdef double last = time()
    # The bitmap of _index_tracker.IndexTracker with the same rules, inlined; it exists
    # while the indices are tracked, from the first index that is not its position.
    cdef bytearray bits = None
    cdef unsigned char* bit_data = NULL
    cdef Py_ssize_t bit_bytes = 0
    cdef object pending = None
    cdef long long pending_max = -1
    cdef long long idx
    cdef long long fold_bytes
    cdef Py_ssize_t min_bytes = 0
    cdef object max_supported = _index_tracker.MAX_INDEX

    for obj in sentences:
        if not isinstance(obj, tuple):
            raise TypeError(f"Passed {type(obj)}: {obj}. Iterable must contain tuple.")
        if len(<tuple>obj) != 2:
            raise ValueError(
                f"At {i}: Passed a tuple of length {len(obj)}. "
                "Each item must be a (words, index) tuple."
            )
        words = (<tuple>obj)[0]
        index = (<tuple>obj)[1]
        if not isinstance(words, list):
            raise TypeError(
                f"At {i}: Passed {type(words)}: {words}. tuple.words must contain list of str."
            )
        # Subclasses of list may override __iter__, which the Python check uses.
        for word in (<list>words if type(words) is list else words):
            if not isinstance(word, str):
                raise TypeError(
                    f"At {i}: Passed {type(words)}: {words}. tuple.words must contain list of str."
                )
        if type(index) is not int:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise TypeError(
                    f"At {i}: Passed {type(index)}: {index}. tuple.index must contain index"
                )
            index = int(index)
        if index < 0:
            raise ValueError(f"At {i}: Passed negative {index}")

        if (i & 4095) == 0 and time() - last > progress_per:
            last = time()
            progress(i, total_words)

        if index > max_index:
            max_index = index
        if index > max_supported:
            _index_tracker.check_index_range(index, i)
        if identity and index != i:
            identity = False
            bits = _index_tracker.new_bitmap(i)
            bit_data, bit_bytes = <unsigned char*><char*>bits, len(bits)
            pending = array("q")
            min_bytes = _index_tracker.BITMAP_MIN_BYTES
        if bits is not None:
            idx = index
            if (idx >> 3) < bit_bytes:
                if bit_data[idx >> 3] & (1 << (idx & 7)):
                    unique_indices = False
                    bits, bit_data, pending = None, NULL, None
                else:
                    bit_data[idx >> 3] |= 1 << (idx & 7)
            else:
                pending.append(idx)
                if idx > pending_max:
                    pending_max = idx
                # fold_size(final=False), in C: fold once the bitmap may double.
                fold_bytes = max(pending_max // 8 + 1, 2 * bit_bytes)
                if fold_bytes <= max(min_bytes, i + 1):
                    if not _index_tracker.fold(bits, pending, fold_bytes):
                        unique_indices = False
                        bits, bit_data, pending = None, NULL, None
                    else:
                        bit_data, bit_bytes = <unsigned char*><char*>bits, len(bits)
                    pending_max = -1
        total_words += len(words)
        if not len(words):
            empty_sentences += 1
        i += 1

    if bits is not None and len(pending):
        fold_bytes = _index_tracker.fold_size(bit_bytes, pending_max, i, True)
        if fold_bytes:
            unique_indices = _index_tracker.fold(bits, pending, fold_bytes)
        else:
            unique_indices = _index_tracker.sorted_unique(pending)
    return i, total_words, empty_sentences, max_index, unique_indices, identity


# ---------------------------------------------------------------------------------------
# Native text input (CorpusFile, TextList): tokenize UTF-8 bytes without the GIL.
#
# A sentence is a span of bytes: a line of a file without its "\n", or one string of a
# TextList. It is decoded strictly as UTF-8 and split like str.split(): at every code
# point for which str.isspace() is true. Tests compare the result with Python.
# ---------------------------------------------------------------------------------------

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


cdef struct VocabView:
    const unsigned char *keys
    const np.int64_t *key_offsets
    const np.uint32_t *key_lengths
    const uINT_t *values
    const uINT_t *slots
    const np.uint64_t *slot_hashes
    np.uint64_t mask


cdef inline uINT_t _vocab_lookup(const VocabView *v, const unsigned char *p, Py_ssize_t n) noexcept nogil:
    """Vocabulary index of the n bytes at p, or NOT_FOUND."""
    cdef np.uint64_t h = _fnv1a64(p, n)
    cdef np.uint64_t j = h & v.mask
    cdef uINT_t k
    while True:
        k = v.slots[j]
        if k == NOT_FOUND:
            return NOT_FOUND
        if (
            v.slot_hashes[j] == h
            and v.key_lengths[k] == n
            and memcmp(v.keys + v.key_offsets[k], p, n) == 0
        ):
            return v.values[k]
        j = (j + 1) & v.mask


cdef class VocabTable:
    """Immutable map from the UTF-8 bytes of a word to its vocabulary index.

    Built once per train or infer call from key_to_index while holding the GIL, then
    read without it: open addressing with linear probing, at most half full.
    """

    cdef bytes arena
    cdef object _arrays
    cdef VocabView view

    def __init__(self, dict key_to_index):
        cdef Py_ssize_t n = len(key_to_index)
        offsets_arr = np.empty(max(n, 1), dtype=np.int64)
        lengths_arr = np.empty(max(n, 1), dtype=np.uint32)
        values_arr = np.empty(max(n, 1), dtype=np.uint32)
        cdef np.int64_t[::1] offsets = offsets_arr
        cdef np.uint32_t[::1] lengths = lengths_arr
        cdef uINT_t[::1] values = values_arr
        cdef list parts = []
        cdef np.int64_t position = 0
        cdef Py_ssize_t k = 0
        for word, index in key_to_index.items():
            encoded = (<str?>word).encode("utf-8")
            parts.append(encoded)
            offsets[k] = position
            lengths[k] = len(encoded)
            values[k] = <uINT_t>index
            position += len(encoded)
            k += 1
        self.arena = b"".join(parts)

        cdef np.uint64_t size = 2
        while size < 2 * <np.uint64_t>n:
            size <<= 1
        self.view.mask = size - 1
        slots_arr = np.full(size, NOT_FOUND, dtype=np.uint32)
        hashes_arr = np.zeros(size, dtype=np.uint64)
        cdef uINT_t[::1] slots = slots_arr
        cdef np.uint64_t[::1] hashes = hashes_arr
        self._arrays = (offsets_arr, lengths_arr, values_arr, slots_arr, hashes_arr)
        self.view.keys = self.arena
        self.view.key_offsets = &offsets[0]
        self.view.key_lengths = &lengths[0]
        self.view.values = &values[0]
        self.view.slots = &slots[0]
        self.view.slot_hashes = &hashes[0]

        cdef np.uint64_t h, j
        for k in range(n):
            h = _fnv1a64(self.view.keys + offsets[k], lengths[k])
            j = h & self.view.mask
            while slots[j] != NOT_FOUND:
                j = (j + 1) & self.view.mask
            slots[j] = <uINT_t>k
            hashes[j] = h

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


cdef int _average_spans(
    const KernelParams *p,
    const VocabView *table,
    const unsigned char *data,
    Py_ssize_t data_size,
    const np.int64_t *item_offsets,
    Py_ssize_t first,
    Py_ssize_t last,
    np.int64_t first_row,
    Py_ssize_t n_rows,
    REAL_t *mem,
    _Buffer *words,
    _Buffer *oov_bytes,
    _Buffer *oov_offsets,
    _Buffer *ngrams,
    Py_ssize_t *eff_sentences,
    Py_ssize_t *eff_words,
    Py_ssize_t *error,
    Py_ssize_t *n_sentences,
) noexcept nogil:
    """Tokenize, look up and average every sentence of a chunk without the GIL."""
    cdef bint lines = item_offsets == NULL
    cdef Py_ssize_t pos = first, span_start, span_end, next_pos, i, token_start, k
    cdef Py_ssize_t longest, capacity
    cdef size_t per_word, needed
    cdef np.int64_t row = first_row
    cdef const unsigned char *newline
    cdef bint inside
    cdef int m
    cdef Py_ssize_t seen
    cdef uINT_t index
    cdef unsigned char open_mark = 60, close_mark = 62  # "<" and ">"

    while pos < last:
        if lines:
            newline = <const unsigned char *>memchr(data + pos, 10, data_size - pos)
            span_start = pos
            span_end = (newline - data) if newline != NULL else data_size
            next_pos = span_end + 1 if newline != NULL else data_size
        else:
            span_start = item_offsets[pos]
            span_end = item_offsets[pos + 1]
            next_pos = pos + 1
        if row < 0 or row >= n_rows:
            return NATIVE_BAD_ROW

        words.size = 0
        oov_bytes.size = 0
        oov_offsets.size = 0
        if _push_i64_nogil(oov_offsets, 0) != 0:
            return NATIVE_NO_MEMORY
        longest = 0
        seen = 0
        inside = False
        token_start = span_start
        i = span_start
        while i <= span_end:
            if i < span_end:
                m = _utf8_len(data, i, span_end)
                if m == 0:
                    error[0] = i
                    return NATIVE_BAD_UTF8
            else:
                m = 0
            if i == span_end or _is_space(data, i, m):
                if inside:
                    seen += 1
                    index = _vocab_lookup(table, data + token_start, i - token_start)
                    if index != NOT_FOUND:
                        if _push_u32_nogil(words, index) != 0:
                            return NATIVE_NO_MEMORY
                    elif p.is_ft:
                        # The UTF-8 bytes of "<token>", as fastText hashes them.
                        if (
                            _push_bytes_nogil(oov_bytes, &open_mark, 1) != 0
                            or _push_bytes_nogil(oov_bytes, data + token_start, i - token_start) != 0
                            or _push_bytes_nogil(oov_bytes, &close_mark, 1) != 0
                            or _push_i64_nogil(oov_offsets, oov_bytes.size) != 0
                            or _push_u32_nogil(words, OOV_TOKEN) != 0
                        ):
                            return NATIVE_NO_MEMORY
                        longest = max(longest, i - token_start + 2)
                    inside = False
                if i == span_end:
                    break
            elif not inside:
                inside = True
                token_start = i
            i += m

        # A sentence with tokens counts even if none is known, as in the job path.
        if seen:
            eff_sentences[0] += 1
        if words.size:
            # A word of b bytes has fewer than b * max_n ngrams; batch_ngrams is at most
            # MAX_NGRAMS_LIMIT, so capacity * 4 cannot overflow.
            per_word = max(<size_t>p.max_n, 1)
            needed = <size_t>longest * per_word if <size_t>longest <= SIZE_MAX // per_word else SIZE_MAX
            capacity = <Py_ssize_t>min(p.max_ngrams, needed)
            if capacity * <Py_ssize_t>sizeof(uINT_t) > ngrams.capacity:
                if _grow_nogil(ngrams, capacity * sizeof(uINT_t) - ngrams.size) != 0:
                    return NATIVE_NO_MEMORY
            k = 0
            eff_words[0] += words.size // sizeof(uINT_t)
            _average_sentence(
                p, <const uINT_t *>words.data, words.size // sizeof(uINT_t),
                <const unsigned char *>oov_bytes.data, <const np.int64_t *>oov_offsets.data,
                &k, <uINT_t>row, mem, <uINT_t *>ngrams.data, <size_t>capacity,
            )
        row += 1
        pos = next_pos
    n_sentences[0] = row - first_row
    return NATIVE_OK


def run_chunk(KernelContext ctx, VocabTable table, const unsigned char[::1] data, offsets,
              Py_ssize_t first, Py_ssize_t last, np.int64_t first_row, scratch,
              Py_ssize_t n_items=-1):
    """Average the sentences first to last - 1 of a native source without the GIL.

    Positions are bytes in line mode (offsets is None) and string numbers otherwise; the
    first sentence goes into row first_row. Returns the number of non-empty sentences and
    of used tokens, like run_job. With ``ctx.prefill_rows``, n_items is the number of
    sentences of the chunk, and their rows are set to EPS first.
    """
    _check_array(scratch, "scratch", np.float32)
    if scratch.size < ctx.params.size:
        raise ValueError(f"scratch must hold at least {ctx.params.size} entries")
    _check_span(data, offsets, first, last)
    if first_row < 0:
        raise ValueError(f"first_row must be non-negative, got {first_row}")
    cdef bint prefill = ctx.prefill_rows
    if ctx.params.ring_rows and not prefill:
        raise ValueError("a ring of windows needs prefill_rows")
    if prefill:
        _check_prefill(ctx, first_row, n_items)
    cdef Py_ssize_t n = data.shape[0]
    cdef const unsigned char *p = &data[0] if n else NULL
    cdef np.int64_t[::1] item_offsets
    cdef const np.int64_t *item_ptr = NULL
    if offsets is not None:
        item_offsets = offsets
        item_ptr = &item_offsets[0]
    cdef REAL_t *mem = <REAL_t *>(np.PyArray_DATA(scratch))
    cdef _Buffer words, oov_bytes, oov_offsets, ngrams
    memset(&words, 0, sizeof(_Buffer))
    memset(&oov_bytes, 0, sizeof(_Buffer))
    memset(&oov_offsets, 0, sizeof(_Buffer))
    memset(&ngrams, 0, sizeof(_Buffer))
    cdef Py_ssize_t eff_sentences = 0, eff_words = 0, error = -1, n_sentences = 0
    # With prefill, no sentence may go past the rows that were set to EPS.
    cdef Py_ssize_t n_rows = first_row + n_items if prefill else ctx.n_rows
    cdef int status
    try:
        with nogil:
            if prefill:
                _prefill(&ctx.params, first_row, n_items)
            status = _average_spans(
                &ctx.params, &table.view, p, n, item_ptr, first, last, first_row, n_rows, mem,
                &words, &oov_bytes, &oov_offsets, &ngrams, &eff_sentences, &eff_words, &error,
                &n_sentences,
            )
    finally:
        free(words.data)
        free(oov_bytes.data)
        free(oov_offsets.data)
        free(ngrams.data)
    if status == NATIVE_NO_MEMORY:
        raise MemoryError("could not grow the buffers of a native chunk")
    if status == NATIVE_BAD_UTF8:
        raise UnicodeDecodeError(
            "utf-8", bytes(data[error : error + 1]), 0, 1,
            f"invalid UTF-8 at byte {error}; the input changed after the scan",
        )
    if status == NATIVE_BAD_ROW:
        if prefill:
            raise ValueError(f"the chunk holds more than its {n_items} sentences")
        raise IndexError(f"a sentence of the chunk maps past the {n_rows} target rows")
    if prefill and n_sentences != n_items:
        raise ValueError(f"the chunk holds {n_sentences} sentences instead of {n_items}")
    return eff_sentences, eff_words


def _check_array(array, name, dtype):
    """Raise unless array is a C-contiguous ndarray of dtype, because the kernel takes raw pointers."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise TypeError(f"{name} must be C-contiguous")


def _word_format(model):
    """The storage format of the model's word vectors: float32, float16 or int8."""
    fmt = getattr(model, "wv_dtype", "float32")
    if fmt not in _FORMATS:
        raise ValueError(f"unknown word vector format {fmt!r}")
    return fmt


def _check_inputs(model, target, memory=None):
    """Validate every array the kernel reads or writes through a raw pointer."""
    _check_array(target, "target", np.float32)
    _check_array(model.word_weights, "word_weights", np.float32)
    if not target.flags.writeable:
        raise ValueError("target must be writeable")
    # The kernel strides through all matrices by the sentence vector size.
    size = model.sv.vector_size
    fmt = _word_format(model)
    code_type = _CODE_TYPES[fmt]
    word_vectors, word_scales, ngram_vectors, ngram_scales = model._word_arrays()
    if fmt == "float32":
        word_name = "wv.vectors_vocab" if model.is_ft else "wv.vectors"
        ngram_name = "wv.vectors_ngrams"
    else:
        word_name, ngram_name = "wv_codes", "ngram_codes"
    _check_array(word_vectors, word_name, code_type)
    if model.is_ft:
        _check_array(ngram_vectors, ngram_name, code_type)
        if ngram_vectors.shape != (model.wv.bucket, size):
            raise ValueError(
                f"{ngram_name} must have shape ({model.wv.bucket}, {size}), "
                f"got {ngram_vectors.shape}"
            )
        if model.batch_ngrams < 1:
            raise ValueError(f"batch_ngrams must be positive, got {model.batch_ngrams}")
    # int8 codes need one float32 scale per row, the other formats none.
    for codes, scales, name in (
        (word_vectors, word_scales, "wv_scales"),
        (ngram_vectors if model.is_ft else None, ngram_scales, "ngram_scales"),
    ):
        if fmt != "int8":
            if scales is not None:
                raise ValueError(f"{name} must be None for {fmt} word vectors")
            continue
        if codes is None:
            continue
        _check_array(scales, name, np.float32)
        if scales.shape != (codes.shape[0],):
            raise ValueError(f"{name} must have shape ({codes.shape[0]},), got {scales.shape}")
    if word_vectors.ndim != 2 or word_vectors.shape[1] != size:
        raise ValueError(f"word vectors must have shape (n, {size}), got {word_vectors.shape}")
    # The encoder takes word indices from key_to_index and the kernel reads their rows.
    if len(model.wv.key_to_index) > word_vectors.shape[0]:
        raise ValueError(
            f"the vocabulary has {len(model.wv.key_to_index)} words, but there are only "
            f"{word_vectors.shape[0]} word vectors"
        )
    if model.word_weights.shape != (word_vectors.shape[0],):
        raise ValueError(
            f"word_weights must have one weight per word vector ({word_vectors.shape[0]}), "
            f"got shape {model.word_weights.shape}"
        )
    if target.ndim != 2 or target.shape[1] != size:
        raise ValueError(f"target must have shape (n, {size}), got {target.shape}")
    # Row and word indices are stored as uint32; the two largest values are markers.
    if target.shape[0] >= OOV_TOKEN or word_vectors.shape[0] >= OOV_TOKEN:
        raise ValueError(f"the kernel supports fewer than {OOV_TOKEN} rows and words")
    if memory is not None:
        _check_array(memory[0], "memory[0]", np.float32)
        if memory[0].size < size:
            raise ValueError(f"memory[0] must hold at least {size} entries, got {memory[0].size}")


def train_average_cy(model, indexed_sentences, target, memory):
    """Training on a sequence of sentences and update the target ndarray.

    Called internally from :meth:`~fse.models.average.Average._do_train_job`. Encodes the
    sentences and averages them like the training workers do.

    Parameters
    ----------
    model : :class:`~fse.models.base_s2v.BaseSentence2VecModel`
        The BaseSentence2VecModel model instance.
    indexed_sentences : iterable of tuple
        The sentences used to train the model.
    target : ndarray
        The target ndarray. We use the index from indexed_sentences
        to write into the corresponding row of target.
    memory : ndarray
        Private memory for each working thread; ``memory[0]`` is the float32 buffer
        the sentence sums are built in.

    Returns
    -------
    int, int
        Number of effective sentences (non-zero) and effective words in the vocabulary used
        during training the sentence embedding.
    """
    _check_inputs(model, target, memory)
    ctx = KernelContext(model, target)
    return run_job(ctx, encode_job(ctx, indexed_sentences), memory[0])


def init():
    return 1

MAX_WORDS_IN_BATCH = MAX_WORDS
MAX_NGRAMS_IN_BATCH = MAX_NGRAMS
FAST_VERSION = init()


# ---------------------------------------------------------------------------------------
# Reading float16 and int8 sentence vectors: dot products on the stored rows.
# ---------------------------------------------------------------------------------------

# Every float16 bit pattern as float32, so a stored float16 value is one table lookup.
cdef REAL_t[65536] _HALF_TABLE
with np.errstate(invalid="ignore"):
    # The bit patterns include NaN payloads, whose conversion numpy reports.
    _half_values = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
for _i in range(65536):
    _HALF_TABLE[_i] = _half_values[_i]
    fse_half_table[_i] = _half_values[_i]
del _half_values, _i

fse_select_kernels(0)


def _kernel_info():
    """The instruction set of the row kernels in use: "avx2", "neon" or "portable"."""
    return fse_kernel_isa.decode("ascii")


def _use_portable_kernels(bint portable):
    """Switch to the kernels of a CPU without AVX2 (True) or back to the fastest (False).

    For tests of the fallback; affects kernel contexts created afterwards.
    """
    fse_select_kernels(portable)


cdef inline REAL_t _dot_int8(const signed char *row, const REAL_t *w, Py_ssize_t dim) noexcept nogil:
    """Dot product of a row of int8 codes with float32 weights, in four float32 sums."""
    cdef REAL_t a0 = 0, a1 = 0, a2 = 0, a3 = 0
    cdef Py_ssize_t j = 0
    while j + 4 <= dim:
        a0 += <REAL_t>row[j] * w[j]
        a1 += <REAL_t>row[j + 1] * w[j + 1]
        a2 += <REAL_t>row[j + 2] * w[j + 2]
        a3 += <REAL_t>row[j + 3] * w[j + 3]
        j += 4
    while j < dim:
        a0 += <REAL_t>row[j] * w[j]
        j += 1
    return (a0 + a1) + (a2 + a3)


cdef inline REAL_t _dot_half(const unsigned short *row, const REAL_t *w, Py_ssize_t dim) noexcept nogil:
    """Dot product of a row of float16 values with float32 weights, in four float32 sums."""
    cdef REAL_t a0 = 0, a1 = 0, a2 = 0, a3 = 0
    cdef Py_ssize_t j = 0
    while j + 4 <= dim:
        a0 += _HALF_TABLE[row[j]] * w[j]
        a1 += _HALF_TABLE[row[j + 1]] * w[j + 1]
        a2 += _HALF_TABLE[row[j + 2]] * w[j + 2]
        a3 += _HALF_TABLE[row[j + 3]] * w[j + 3]
        j += 4
    while j < dim:
        a0 += _HALF_TABLE[row[j]] * w[j]
        j += 1
    return (a0 + a1) + (a2 + a3)


cdef double _dot_wide(
    const void *codes, bint half, size_t offset, const REAL_t *w, Py_ssize_t dim, double scale
) noexcept nogil:
    """The scaled dot product of one row in double, for rows whose float32 sum overflows.

    An int8 code times a large weight can exceed float32 although the scaled result does
    not; summing in double and scaling first gives the representable result.
    """
    cdef double total = 0
    cdef Py_ssize_t j
    for j in range(dim):
        if half:
            total += <double>_HALF_TABLE[(<const unsigned short *>codes)[offset + j]] * w[j]
        else:
            total += <double>(<const signed char *>codes)[offset + j] * w[j]
    return total * scale


cdef void _score(
    const void *codes, bint half, Py_ssize_t dim, Py_ssize_t k, const REAL_t *weights_t,
    const REAL_t *scales, const REAL_t *factors, const REAL_t *bias, REAL_t *out,
    Py_ssize_t begin, Py_ssize_t end,
) noexcept nogil:
    """out[i, c] = dot(codes[i], weights_t[c]) * scale_i * factor_i + bias[c]."""
    cdef Py_ssize_t i, c
    cdef REAL_t scale, dot
    cdef double wide
    cdef size_t offset
    for i in range(begin, end):
        scale = (scales[i] if scales != NULL else ONEF) * (factors[i] if factors != NULL else ONEF)
        offset = <size_t>i * <size_t>dim
        for c in range(k):
            if half:
                dot = _dot_half(&(<const unsigned short *>codes)[offset], &weights_t[<size_t>c * <size_t>dim], dim)
            else:
                dot = _dot_int8(&(<const signed char *>codes)[offset], &weights_t[<size_t>c * <size_t>dim], dim)
            dot = dot * scale
            if not isfinite(dot):
                # Rare: the unscaled sum left float32; NaN or infinite inputs stay so.
                wide = _dot_wide(codes, half, offset, &weights_t[<size_t>c * <size_t>dim], dim, scale)
                dot = <REAL_t>wide
            out[<size_t>i * <size_t>k + c] = dot + (bias[c] if bias != NULL else 0)


def score_rows(codes, scales, weights_t, bias, factors, out, Py_ssize_t begin, Py_ssize_t end):
    """``out[i] = (codes[i] @ weights_t.T) * scales[i] * factors[i] + bias`` for rows begin to end.

    codes are stored float16 or int8 sentence vectors of shape (n, dim); weights_t is
    float32 of shape (k, dim), one row of weights per output; scales and factors are
    float32 of shape (n,) or None; bias is float32 of shape (k,) or None; out is float32 of
    shape (n, k). Sums run in float32 over the stored values in double, without the GIL and without
    converting a block first.
    """
    if not isinstance(codes, np.ndarray) or codes.ndim != 2:
        raise TypeError("codes must be a 2-dimensional numpy array")
    if codes.dtype != np.int8 and codes.dtype != np.float16:
        raise TypeError(f"codes must have dtype int8 or float16, got {codes.dtype}")
    if not codes.flags.c_contiguous:
        raise ValueError("codes must be C-contiguous")
    cdef Py_ssize_t n = codes.shape[0], dim = codes.shape[1]
    _check_array(weights_t, "weights_t", np.float32)
    if weights_t.ndim != 2 or weights_t.shape[1] != dim:
        raise ValueError(f"weights_t must have shape (k, {dim}), got {weights_t.shape}")
    cdef Py_ssize_t k = weights_t.shape[0]
    _check_array(out, "out", np.float32)
    if not out.flags.writeable:
        raise ValueError("out must be writeable")
    if out.ndim != 2 or out.shape[0] != n or out.shape[1] != k:
        raise ValueError(f"out must have shape ({n}, {k}), got {out.shape}")
    for array, name, size in ((scales, "scales", n), (factors, "factors", n), (bias, "bias", k)):
        if array is not None:
            _check_array(array, name, np.float32)
            if array.ndim != 1 or array.shape[0] != size:
                raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not 0 <= begin <= end <= n:
        raise ValueError(f"rows [{begin}, {end}) are outside the {n} rows")
    # Writing out while reading an overlapping input would change later results.
    for array, name in ((codes, "codes"), (weights_t, "weights_t"), (scales, "scales"),
                        (factors, "factors"), (bias, "bias")):
        if array is not None and np.may_share_memory(out, array):
            raise ValueError(f"out must not overlap {name}")
    cdef const REAL_t *w = <const REAL_t *>np.PyArray_DATA(weights_t)
    cdef const REAL_t *s = <const REAL_t *>np.PyArray_DATA(scales) if scales is not None else NULL
    cdef const REAL_t *f = <const REAL_t *>np.PyArray_DATA(factors) if factors is not None else NULL
    cdef const REAL_t *b = <const REAL_t *>np.PyArray_DATA(bias) if bias is not None else NULL
    cdef REAL_t *o = <REAL_t *>np.PyArray_DATA(out)
    cdef const void *data = np.PyArray_DATA(codes)
    cdef bint half = codes.dtype == np.float16
    with nogil:
        _score(data, half, dim, k, w, s, f, b, o, begin, end)
